import torch
from lightning.pytorch import LightningModule

from models.model_util import create_optimizer, create_scheduler
from utils.loss import ComputeLoss
from utils.general import non_max_suppression



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
        self.mloss = torch.zeros(4)  # mean losses
        self.model = model
        self.loss_fn = loss_fn if loss_fn else ComputeLoss(self.model)
        
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

    def on_train_epoch_end(self):
        self.mloss = torch.zeros(4)  # mean losses
    
    def validation_step(self, batch, batch_idx, conf_thres=0.001, iou_thres=0.6):
        imgs, targets, paths, _ = batch
        imgs = imgs.to(non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width
        augment = False
        with torch.no_grad():
            # Run model
            out, train_out = self.model(img, augment=augment)
            
            loss += self.loss_fn([x.float() for x in train_out], targets)
            
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height])
            lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
           
            out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=lb, multi_label=True)
            
            
    
    def on_validation_epoch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = create_optimizer(self.model, self.cfg)
        scheduler = create_scheduler(optimizer, self.cfg)
        
        return [optimizer], [scheduler]
    
    