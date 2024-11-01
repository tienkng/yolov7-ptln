import os
import yaml
import torch
import logging
import argparse
import numpy as np
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from engine import LitYOLO
from models.yolo import Model as YOLO
from models.model_util import attempt_download
from utils.torch_utils import intersect_dicts, torch_distributed_zero_first, select_device
from utils.general import check_img_size, check_file, set_logging, colorstr, labels_to_class_weights, increment_path, init_seeds
from utils.plots import plot_evolution
from utils.autoanchor import check_anchors
from data.dataloader import create_dataloader
    

logger = logging.getLogger(__name__)


    
def main(opt, tb_writer=None):
    set_logging()
    opt.cfg = check_file(opt.cfg)  # check file
    device = select_device(opt.device)
    init_seeds(2 + opt.global_rank)
    
    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps
    hyp.update(vars(opt))
    
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict
    
    for k, v in hyp.items():
        print(k, "\t", v)
    
    wandb_logger = WandbLogger(project=opt.name, log_model="all")
    
    dist = True if len(opt.device) > 1 else False
    plots = not opt.evolve  # create plots
    cuda = device.type != 'cpu'
    num_classes = 1 if opt.single_cls else int(data_dict['nc'])  # number of classes
    names = ['item'] if opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    assert len(names) == num_classes, '%g names found for nc=%g dataset in %s' % (len(names), num_classes, opt.data)  # check
    train_path = data_dict['train']
    test_path = data_dict['val']
    
    pretrained = opt.weights.endswith('.pt')
    if pretrained:
        with torch_distributed_zero_first(opt.global_rank):
            attempt_download(opt.weights)  # download if not found locally
        
        ckpt = torch.load(opt.weights, map_location=device)  # load checkpoint
        model = YOLO(hyp['cfg'] or ckpt['model'].yaml, ch=3, class_num=num_classes, anchors=hyp.get('anchors'))
        exclude = ['anchor'] if (opt.cfg or hyp.get('anchors')) and not opt.resume else []  # exclude keys
        state_dict = ckpt['model'].float().state_dict()  # to FP32
        state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(state_dict, strict=False)  # load
        logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), opt.weights))  # report
    else:
        model = YOLO(opt.cfg, ch=3, class_num=num_classes, anchors=hyp.get('anchors')).to(device)  # create
        
    # Freeze model
    freeze = [f'model.{x}.' for x in (opt.freeze if len(opt.freeze) > 1 else range(opt.freeze[0]))]  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False
            
    # Update Optimzier
    norm_batch_size = 64
    accumulate = max(round(norm_batch_size / opt.batch_size), 1)
    hyp['weight_decay'] *= opt.batch_size * accumulate / norm_batch_size # scale_weight_decay
    logger.info(f"------ Scaled weight_decay = {hyp['weight_decay']} -----")
    
    # Create EMAModel
    # ema = ModelEMA(model) if opt.global_rank else None
    
    # Image sizes
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    nl = model.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
    imgsz, imgsz_test = [check_img_size(x, gs) for x in opt.img_size]  # verify imgsz are gs-multiples
    
    # SyncBatchNorm
    if opt.sync_bn and opt.global_rank != -1 and cuda:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        logger.info("\t\tUsing SyncBatchNorm()\n")
    
    # Resume -> Not Implement yet
    
    # Create dataloader
    train_dataloader, dataset = create_dataloader(train_path, imgsz, opt.batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache_images, rect=opt.rect, rank=opt.global_rank,
                                            world_size=opt.world_size, workers=opt.workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '))
    mlc = np.concatenate(dataset.labels, 0)[:, 0].max()  # max label class
    nb = len(train_dataloader)  # number of batches
    assert mlc < num_classes, 'Label class %g exceeds nc=%g in %s. Possible class labels are 0-%g' % (mlc, num_classes, opt.data, num_classes - 1)

    # Process 0
    if opt.global_rank in [-1, 0]:
        test_dataloader = create_dataloader(test_path, imgsz_test, opt.batch_size * 2, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache_images and not opt.notest, rect=True, rank=-1,
                                       world_size=opt.world_size, workers=opt.workers,
                                       pad=0.5, prefix=colorstr('val: '))[0]

        if not opt.resume:
            labels = np.concatenate(dataset.labels, 0)
            c = torch.tensor(labels[:, 0])  # classes
            # cf = torch.bincount(c.long(), minlength=nc) + 1.  # frequency
            # model._initialize_biases(cf.to(device))
            if plots:
                #plot_labels(labels, names, save_dir, loggers)
                if tb_writer:
                    tb_writer.add_histogram('classes', c, 0)

            # Anchors
            if not opt.noautoanchor:
                check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)
            model.half().float()  # pre-reduce anchor precision
            
    # Model parameters
    hyp['box'] *= 3. / nl  # scale to layers
    hyp['cls'] *= num_classes / 80. * 3. / nl  # scale to classes and layers
    hyp['obj'] *= (imgsz / 640) ** 2 * 3. / nl  # scale to image size and layers
    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = num_classes  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.class_weights = labels_to_class_weights(dataset.labels, num_classes).to(device) * num_classes  # attach class weights
    model.names = names
    
    # Wrapper by Pytorch Lightning
    lit_yolo = LitYOLO(cfg=hyp, model=model, dist=dist)
    
    # Create callback functions
    model_checkpoint = ModelCheckpoint(save_top_k=3,
                        monitor="train/loss",
                        mode="min", dirpath="output/",
                        filename="sample-{epoch:02d}",
                        save_weights_only=True)
    
    
    opt.device = [int(x) for x in opt.device]
    # Create Trainer
    trainer = Trainer(
        max_epochs=opt.epochs,
        accelerator=opt.accelerator,
        devices=opt.device,
        callbacks=[model_checkpoint],
        strategy='ddp_find_unused_parameters_true' if dist else 'auto',
        log_every_n_steps=opt.log_steps,
        logger=wandb_logger,
        precision=16
    )
    
    if opt.do_train:
        logger.info("*** Start training ***")
        trainer.fit(
            model=lit_yolo, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=test_dataloader if opt.do_eval else None
        )
        
        # Saves only on the main process    
        saved_ckpt_path = f'{opt.save_dir}/weights'
        os.makedirs(saved_ckpt_path, exist_ok=True)
        saved_ckpt_path = f'{saved_ckpt_path}/best.pt'
        trainer.save_checkpoint(saved_ckpt_path)
        
    if opt.do_eval:
        logger.info("\n\n*** Evaluate ***")
        trainer.devices = 0
        trainer.test(lit_yolo, dataloaders=test_dataloader, ckpt_path="best")
        
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--accelerator', default='auto', help='cpu, gpu, tpu or auto')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    parser.add_argument('--log-steps', type=int, default=20, help='Loging step')
    parser.add_argument('--do-train', action='store_true', help='Do training')
    parser.add_argument('--do-eval', action='store_true', help='Do eval')
    opt = parser.parse_args()
    
    # check config
    opt.notest, opt.nosave = True, True  # only test/save final epoch
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    opt.global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else -1
    opt.data, opt.cfg, opt.hyp = check_file(opt.data), check_file(opt.cfg), check_file(opt.hyp)  # check files
    assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
    opt.img_size.extend([opt.img_size[-1]] * (2 - len(opt.img_size)))  # extend to 2 sizes (train, test)
    opt.name = 'evolve' if opt.evolve else opt.name
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    yaml_file = Path(opt.save_dir) / 'hyp_evolved.yaml'  # save best result here

    set_logging(opt.global_rank)
    
    main(opt)
    
    # Write mutation results
    # print_mutation(hyp.copy(), results, yaml_file, opt.bucket)
            
    # Plot results
    # plot_evolution(yaml_file)
    # print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
    #         f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')