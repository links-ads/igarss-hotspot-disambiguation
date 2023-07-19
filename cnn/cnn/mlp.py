from typing import Callable, List, Optional, Type, Union
import warnings

import torch

warnings.filterwarnings("ignore")

import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
import torchvision.models as models
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchmetrics.classification import BinaryF1Score
from dataset import HotspotSatelliteFeaturesDataset
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.ops import MLP

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class MLPClassifier(pl.LightningModule):
    optimizers = {"adam": Adam, "AdamW": AdamW, "sgd": SGD}
    schedulers = {"cos": CosineAnnealingLR, "cosWR": CosineAnnealingWarmRestarts, "step": StepLR, "plateau": ReduceLROnPlateau}

    def __init__(
        self,
        train_paths: list,
        val_paths: list,
        test_paths=None,
        optimizer="adam",
        scheduler="plateau",
        lr=1e-3,
        batch_size=16,
        num_channels=32,
        num_workers = 1
    ):
        super().__init__()

        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.lr = lr
        self.batch_size = batch_size
        self.class_weights = torch.tensor([0.8])
        self.optimizer = self.optimizers[optimizer]
        self.scheduler = self.schedulers[scheduler]
        self.num_workers = num_workers
        
        if scheduler == "step":
            self.scheduler_params = {
                "step_size" : 10,
                "gamma" : 0.1,
                "last_epoch": - 1,
                "verbose": False
            }
            
        elif scheduler == "cos":
            self.scheduler_params = {
                "T_max":100000,
                "eta_min": 0,
                "last_epoch": -1,
                "verbose": False
            }
        elif scheduler == "cosWR":
            self.scheduler_params = {
                "T_0" : 50000,
                "T_mult": 1,
                "eta_min": 0,
                "last_epoch": - 1,
                "verbose": False
            }
        elif scheduler == "plateau":
            self.scheduler_params = dict(
                mode='min', 
                factor=0.1,
                patience=10, 
                threshold=0.0001, 
                threshold_mode='rel', 
                cooldown=0, 
                min_lr=0, 
                eps=1e-08, 
                verbose=False
            )
        else:
            self.scheduler = None
            self.scheduler_params = None

        # instantiate loss criterion
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights) 
        
        # create accuracy metric
        self.acc = Accuracy(
            task="binary"
        )
        self.f1 = BinaryF1Score()
        
       
        
        self.NUM_CHANNELS = num_channels
        self.mlp = MLP(self.NUM_CHANNELS, hidden_channels=[round(self.NUM_CHANNELS/2),round(self.NUM_CHANNELS/4),1], bias=False)
        

    def forward(self, X):
        return self.mlp(X)

    def configure_optimizers(self):
        opt = self.optimizer(self.parameters(), lr=self.lr)

        if self.scheduler is None:
            return opt
        
        if self.scheduler == "plateau":
            metric = "train_loss"
        else:
            metric = "val_loss"

        sch = self.scheduler(opt, **self.scheduler_params)
        return {"optimizer": opt , 
                "lr_scheduler": {
                    "scheduler": sch,
                    "monitor": metric
                }}

    def _step(self, batch):

        x = batch['feat']
        y = batch['label']

        preds = self(x)
        preds = preds.flatten()
        y = y.float()

        loss = self.loss_fn(preds, y)

        acc = self.acc((sigmoid(preds) > 0.5).long(), y)
        f1 = self.f1((sigmoid(preds) > 0.5).long(), y)
        return loss, acc, f1

    def _dataloader(self, data_path, shuffle=False):
        # values here are specific to pneumonia dataset and should be updated for custom data
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize((500, 500)),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.48232,), (0.23051,)),
        #     ]
        # )

        # img_folder = ImageFolder(data_path, transform=transform)
        dataset = HotspotSatelliteFeaturesDataset(catalogs=data_path)  
        

        # 
        if self.batch_size > len(dataset):
            return DataLoader(dataset, batch_size=len(dataset), shuffle=shuffle, drop_last=True, num_workers=self.num_workers), len(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True, num_workers=self.num_workers), len(dataset)

    def train_dataloader(self):
        dl, size = self._dataloader(self.train_paths, shuffle=True)
        print(f"TRAIN DATASET: Loaded {size} samples")
        return dl

    def training_step(self, batch, batch_idx):

        loss, acc, f1 = self._step(batch)
        # perform logging
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_f1", f1, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def val_dataloader(self):
        dl, size = self._dataloader(self.val_paths)
        print(f"TRAIN DATASET: Loaded {size} samples")
        return dl

    def validation_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1", f1, on_epoch=True, prog_bar=True, logger=True)

    def test_dataloader(self):
        dl, size = self._dataloader(self.test_paths)
        print(f"TRAIN DATASET: Loaded {size} samples")
        return dl

    def test_step(self, batch, batch_idx):
        loss, acc, f1 = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)
        self.log("test_f1", f1, on_step=True, prog_bar=True, logger=True)


# class KFoldTrainingPool(FitLoop):

#     def __init__(self, split_dataloaders:list):
#         super(FitLoop).__init__()
#         self.split_dataloaders = split_dataloaders

#     def run(self, dataloader):
#         for k in range(len(self.split_dataloaders)):
#             test_dl = k
#             train_dl = [i for i in range(len(self.split_dataloaders)) if i != k]
#             for d in train_dl:
#                 for i, batch in enumerate(self.split_dataloaders[d]):
#                     self.advance(batch, i)
#             # test iteration
#             for i, batch in enumerate(self.split_dataloaders[test_dl]):
#                 self.advance(batch, i)



