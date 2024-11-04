import os
import torch
import numpy as np
from pathlib import Path
from lightning.pytorch import LightningModule 
from threading import Thread


from models.model_util import create_optimizer, create_scheduler
from utils.loss import ComputeLoss
from utils.general import non_max_suppression, scale_coords, xywh2xyxy, box_iou, increment_path
from utils.plots import plot_images, output_to_target
from utils.metrics import ConfusionMatrix, ap_per_class



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
        self.validation_step_outputs = []
    
    def on_train_batch_start(self, batch, batch_idx):
        self.mloss = torch.zeros(4)
        
    
    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        imgs = imgs.float() / 255.0 # uint8 to float32, 0-255 to 0.0-1.0
        pred = self.model(imgs)
        loss, loss_item = self.loss_fn(pred, targets)
        
        # mem = '%.3g' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
        # self.log('train/GPU_MEM', float(mem), on_epoch=True, prog_bar=True, logger=True, sync_dist=self.dist)
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
    
    def on_validation_epoch_start(self, save_txt=False):
        self.seen = 0
        self.stats = []
        self.iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        
        self.val_idx = 0

        # plot
        self.plot = True
        self.names = {k: v for k, v in enumerate(self.model.names if hasattr(self.model, 'names') else self.model.module.names)}
        self.save_dir = Path(self.cfg.get("save_dir"), exist_ok=True)  # increment run
        (self.save_dir / 'labels' if save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # metric
        self.v5_metric = False
        # s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
        # p, r, f1, mp, mr, map50, map= 0., 0., 0., 0., 0., 0., 0.,
        self.loss = torch.zeros(3)
    
    def validation_step(self, batch, batch_idx, conf_thres=0.001, iou_thres=0.6, save_hybrid=False, plots=True):
        img, targets, paths, shapes = batch
        
        img = img.to(non_blocking=True)
        half = img.device.type != 'cpu'
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        
        device = img.device
        self.loss = self.loss.to(device)
        self.iouv = self.iouv.to(device)
        
        with torch.no_grad():
            # Run model
            out, train_out = self.model(img, augment=False)
            self.loss += self.loss_fn([x.float() for x in train_out], targets)[1][:3]  # box, obj, cls
            self.val_idx += 1 
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            
        for si, pred in enumerate(out):
            labels = targets[targets[:,0] == si, 1:]
            nl = len(labels)
            tcls = labels[:,0].tolist() if nl else [] # target class
            # path = Path(paths[si])
            self.seen += 1
            
            if len(pred) == 0:
                if nl:
                    self.stats.append(torch.zero(0, self.niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls)
                continue
            
            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
        
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
            
        # Plot images
        if plots and batch_idx < 3:
            f = os.path.join(self.save_dir,f'test_batch{batch_idx}_labels.jpg')  # labels
            Thread(target=plot_images, args=(img, targets, paths, f, self.names), daemon=True).start()
            f = os.path.join(self.save_dir,f'test_batch{batch_idx}_pred.jpg')  # predictions
            Thread(target=plot_images, args=(img, output_to_target(out), paths, f, self.names), daemon=True).start()

    def on_validation_epoch_end(self, plots=True, verbose=False):
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*self.stats)]  # to numpy
        
        if len(stats) and stats[0].any():
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, v5_metric=self.v5_metric, save_dir=self.save_dir, names=self.names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=self.model.nc)  # number of targets per class
        else:
            nt = torch.zeros(1)
            
        # Print results
        pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
        print(pf % ('all', self.seen, nt.sum(), mp, mr, map50, map))

        # Print results per class
        if (verbose or self.model.nc < 50) and self.model.nc > 1 and len(stats):
            for i, c in enumerate(ap_class):
                print(pf % (self.names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))
                
        # Plots
        # if self.plots:
        #     self.confusion_matrix.plot(save_dir=self.cfg.save_dir, names=list(self.names.values()))
        #     if wandb_logger and wandb_logger.wandb:
        #         val_batches = [wandb_logger.wandb.Image(str(f), caption=f.name) for f in sorted(self.cfg.save_dir.glob('test*.jpg'))]
        #         wandb_logger.log({"Validation": val_batches})
        # if wandb_images:
        #     wandb_logger.log({"Bounding Box Debugger/Images": wandb_images})

        # Return results
        # self.model.float()  # for training
        # s = f"\n{len(list(self.cfg.get("save_dir").glob('labels/*.txt')))} labels saved to {self.cfg.get("save_dir") / 'labels'}" if self.cfg.save_txt else ''
        # print(f"Results saved to {self.cfg.save_dir}{s}")
        maps = np.zeros(self.model.nc) + map
        for i, c in enumerate(ap_class):
            maps[c] = ap[i]
        
        loss = (self.loss.cpu() / self.val_idx).tolist()

        for idx, name in enumerate(['box_loss', 'obj_loss', 'cls_loss']):
            self.log(
                f"val/{name}", loss[idx],
                on_epoch=True, 
                on_step=False,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )

        for _, (name, value) in enumerate(zip(['mp', 'mr', 'map50', 'map'], [mp, mr, map50, map])):
            self.log(
                f'val/{name}',
                value,
                on_epoch=True, 
                on_step=False,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            ) 
        return sum(loss) / len(loss)
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = create_optimizer(self.model, self.cfg)
        scheduler = create_scheduler(optimizer, self.cfg)
        
        return [optimizer], [scheduler]
    
    