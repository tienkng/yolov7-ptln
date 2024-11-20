import os
import torch
import numpy as np
from pathlib import Path
from lightning.pytorch import LightningModule 
from lightning.pytorch.loggers import WandbLogger

from models.model_util import create_optimizer, create_scheduler
from utils.loss import ComputeLoss
from utils.general import non_max_suppression, scale_coords, xywh2xyxy, box_iou, increment_path, check_img_size, xyxy2xywh, coco80_to_coco91_class, fitness
from utils.plots import plot_images, output_to_target
from utils.metrics import ConfusionMatrix, ap_per_class
from utils.torch_utils import time_synchronized
from threading import Thread


class LitYOLO(LightningModule):
    def __init__(
        self,
        cfg,
        model,
        loss_fn=None,
        dist:bool=False,
    ):
        super(LitYOLO, self).__init__()
        self.cfg = cfg
        self.dist = dist
        self.model = model
        self.loss_fn = loss_fn if loss_fn else ComputeLoss(self.model)
        self.confusion_matrix = ConfusionMatrix(nc=self.model.nc)        
        self.bbox_interval = (self.cfg['epochs'] // 10) if self.cfg['epochs'] > 10 else 1
        self.best_fitness = 0
        
    def on_train_batch_start(self, batch, batch_idx):
        self.mloss = torch.zeros(4)
    
    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        imgs = imgs.float() / 255.0 # uint8 to float32, 0-255 to 0.0-1.0
        pred = self.model(imgs)
        loss, loss_item = self.loss_fn(pred, targets)
        
        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)
        
        if self.mloss.device != loss_item.device:
            self.mloss = self.mloss.to(loss_item)
        self.mloss = (self.mloss * batch_idx + loss_item) / (batch_idx + 1) # update mean losses
        
        for idx, x in enumerate(['box', 'obj', 'cls']):
            self.log(
                f'train/{x}',
                self.mloss[idx],
                on_epoch=True, 
                on_step=False,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )
        
        return loss
    
    def init(self):
        imgsz = 640
        gs = max(int(self.model.stride.max()), 32)  # grid size (max stride)
        self.imgsz = check_img_size(imgsz, s=gs)  # check img_size
        
        self.nc = self.model.nc # number of classes
        self.iouv = torch.linspace(0.5, 0.95, 10).to(self.device)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
    
        self.seen = 0
        self.names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        
        self.coco91class = coco80_to_coco91_class()
        self.t0, self.t1 = 0., 0.
        self.loss = torch.zeros(3, device=self.device)
        self.jdict, self.stats, self.wandb_images = [], [], []
        self.save_txt = False
        self.save_conf = False
        
        # Directories
        if 'save_dir' in self.cfg:
            self.save_dir = Path(self.cfg['save_dir'], exist_ok=self.cfg['exist_ok'])
        else:
            print("\n\nAuto increase path\n\n")
            self.save_dir = Path(increment_path(Path(self.cfg['project']) / self.cfg['name'], exist_ok=self.cfg['exist_ok']))  # increment run
            
        (self.save_dir / 'labels' if self.cfg['save_txt'] else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        self.save_json = False
        self.is_coco = False
        if self.cfg['data'].endswith('coco.yaml'):
            self.is_coco = True 
            self.save_json = True
        
        self.eval_idx = []
        
        # Logging
        self.log_imgs = 0
        if self._trainer.logger and isinstance(self._trainer.logger, WandbLogger):
            self.log_imgs = min(16, 100)
    
    def on_validation_epoch_start(self):
        self.init() 
    
    def validation_step(self, batch, batch_idx, conf_thres=0.001, iou_thres=0.6, save_hybrid=False, plots=True):
        img, targets, paths, shapes = batch

        # uint8 to fp16/32
        param_dtype = next(self.model.parameters()).dtype
        if param_dtype == torch.float16:
            img = img.half()
        elif param_dtype == torch.float32:
            img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        device = img.device
        
        with torch.no_grad():
            # Run model
            out, train_out = self.model(img, augment=False)  # inference and training outputs

            # Compute loss
            if self.loss_fn:
                self.loss += self.loss_fn([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            self.seen += 1

            if len(pred) == 0:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            # Append to text file -> test
            if self.save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                    with open(self.save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=device)
            
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                
                 # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    self.confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
                    
            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))        
        self.eval_idx.append(batch_idx)

             
    def on_validation_epoch_end(self, plots=True, v5_metric=False):
        # Compute statistics
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(self.stats) and self.stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*self.stats, plot=plots, v5_metric=v5_metric, save_dir=self.save_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()

        # log
        loss = (self.loss.cpu() / len(self.eval_idx)).tolist()
        for idx, name in enumerate(['box_loss', 'obj_loss', 'cls_loss']):
            self.log(
                f"val/{name}", loss[idx],
                on_epoch=True, 
                on_step=False,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )
            
        fi = fitness(np.array([mp, mr, map50, map]).reshape(1, -1))[0]  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
                
        for _, (name, value) in enumerate(zip(['mP', 'mR', 'mAP@.5', 'mAP@0.5:0.95', 'fi'], [mp, mr, map50, map, fi])):
            self.log(
                f'metrics/{name}',
                value,
                on_epoch=True, 
                on_step=False,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )
                
        return sum(loss) / len(loss)
    
    def on_test_epoch_start(self):
        print("\n\n\nDebug Test \n\n\n")
        self.init()
    
    def test_step(self, batch, batch_idx, conf_thres=0.25, iou_thres=0.7, save_hybrid=False, plots=True):
        img, targets, paths, shapes = batch

        # uint8 to fp16/32
        param_dtype = next(self.model.parameters()).dtype
        if param_dtype == torch.float16:
            img = img.half()
        elif param_dtype == torch.float32:
            img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        device = img.device
        
        with torch.no_grad():
            # Run model
            t = time_synchronized()
            out, train_out = self.model(img, augment=False)  # inference and training outputs
            self.t0 += time_synchronized() - t

            # Compute loss
            if self.loss_fn:
                self.loss += self.loss_fn([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls

            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            t = time_synchronized()
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            self.t1 += time_synchronized() - t
            
        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path = Path(paths[si])
            self.seen += 1

            if len(pred) == 0:
                if nl:
                    self.stats.append((torch.zeros(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            
            # Append to text file -> test
            if self.save_txt:
                gn = torch.tensor(shapes[si][0])[[1, 0, 1, 0]]  # normalization gain whwh
                for *xyxy, conf, cls in predn.tolist():
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
                    with open(self.save_dir / 'labels' / (path.stem + '.txt'), 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], self.niou, dtype=torch.bool, device=device)
            
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                
                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                if plots:
                    self.confusion_matrix.process_batch(predn, torch.cat((labels[:, 0:1], tbox), 1))

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        detected_set = set()
                        for j in (ious > self.iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > self.iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
                    
            # Append statistics (correct, conf, pcls, tcls)
            self.stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        
        # Plot images -> test
        if plots and batch_idx < 3:
            f = self.save_dir / f'test_batch{batch_idx}_labels.jpg'  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, self.names), daemon=True).start()
            f = self.save_dir / f'test_batch{batch_idx}_pred.jpg'  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, self.names), daemon=True).start()
            
    def on_test_epoch_end(self, plots=True, v5_metric=False, verbose=False):
        self.stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        if len(self.stats) and self.stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*self.stats, plot=plots, v5_metric=v5_metric, save_dir=self.save_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(self.stats[3].astype(np.int64), minlength=self.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)

        # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95'))
        print(pf % ('all', self.seen, nt.sum(), mp, mr, map50, map))
        
        # Print results per class
        if (verbose or (self.nc < 50)) and self.nc > 1 and len(self.stats):
            for i, c in enumerate(ap_class):
                print(pf % (self.names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))

        # Print speeds -> test
        t = tuple(x / self.seen * 1E3 for x in (self.t0, self.t1, self.t0 + self.t1)) + (self.imgsz, self.imgsz, self.cfg['batch_size'])  # tuple
        # if not training:
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

        # Plots -> test
        if plots:
            self.confusion_matrix.plot(save_dir=self.save_dir, names=list(self.names.values()))
            if self._trainer.logger and isinstance(self._trainer.logger, WandbLogger):
                for f in sorted(self.save_dir.glob('test*.jpg')):
                    self._trainer.logger.log_image(key='samples', images=[str(f)], caption=[f.name])
            
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = create_optimizer(self.model, self.cfg)
        scheduler = create_scheduler(optimizer, self.cfg)
        
        return [optimizer], [scheduler]