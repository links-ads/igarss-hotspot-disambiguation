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
from dataset import HotspotSatelliteDataset
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from torchvision.ops import MLP

# Here we define a new class to turn the ResNet model that we want to use as a feature extractor
# into a pytorch-lightning module so that we can take advantage of lightning's Trainer object.
# We aim to make it a little more general by allowing users to define the number of prediction classes.
class ResNetClassifier(pl.LightningModule):
    resnets = {
        18: models.resnet18,
        34: models.resnet34,
        50: models.resnet50,
        101: models.resnet101,
        152: models.resnet152,
    }
    optimizers = {"adam": Adam, "AdamW": AdamW, "sgd": SGD}
    schedulers = {"cos": CosineAnnealingLR, "cosWR": CosineAnnealingWarmRestarts, "step": StepLR, "plateau": ReduceLROnPlateau}

    def __init__(
        self,
        num_classes,
        resnet_version,
        train_paths: list,
        val_paths: list,
        test_paths=None,
        optimizer="adam",
        scheduler="plateau",
        lr=1e-3,
        batch_size=16,
        transfer=False,
        tune_fc_only=False,
        include_lc = True,
        crop_size = 32,
        num_channels=32,
        external_features_as_channels=False,
        num_workers = 1,
        use_mlp = False,
        class_weight = 1.0,
        return_logits=False
    ):
        super().__init__()

        self.num_classes = num_classes
        self.train_paths = train_paths
        self.val_paths = val_paths
        self.test_paths = test_paths
        self.lr = lr
        self.batch_size = batch_size
        self.class_weights = torch.tensor([class_weight])
        self.include_lc = include_lc
        self.crop_size = crop_size
        self.optimizer = self.optimizers[optimizer]
        self.scheduler = self.schedulers[scheduler]
        self.num_workers = num_workers
        self.external_features_as_channels = external_features_as_channels
        self.use_mlp = use_mlp
        
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
        self.loss_fn = (
            nn.BCEWithLogitsLoss(pos_weight=self.class_weights) if num_classes == 1 else nn.CrossEntropyLoss()
        )
        # create accuracy metric
        self.acc = Accuracy(
            task="binary" if num_classes == 1 else "multiclass", num_classes=num_classes
        )
        self.f1 = BinaryF1Score()
        # Using a pretrained ResNet backbone
        self.resnet_model = self.resnets[resnet_version](pretrained=transfer)
        # Replace old FC layer with Identity so we can train our own
        linear_size = list(self.resnet_model.children())[-1].in_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        
        self.NUM_CHANNELS = num_channels
        if include_lc:
            self.NUM_CHANNELS +=1

        self.resnet_model.conv1 = nn.Conv2d(self.NUM_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if self.use_mlp:
            self.mlp = MLP(512, hidden_channels=[256,128,1])
            # add new_forward function to the resnet instance as a class method
            bound_method = self.__resnet_end_to_end_forward.__get__(self.resnet_model, self.resnet_model.__class__)
            setattr(self.resnet_model, 'forward', bound_method)

        if return_logits:
            bound_method = self.__resnet_end_to_end_forward.__get__(self.resnet_model, self.resnet_model.__class__)
            setattr(self.resnet_model, 'forward', bound_method)
        if tune_fc_only:  # option to only tune the fully-connected layers
            for child in list(self.resnet_model.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

    @staticmethod
    def __resnet_end_to_end_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


    def forward(self, X):
        x = self.resnet_model(X)
        if self.use_mlp:
            x = self.mlp(x)
        return x

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

        x = batch['img']
        y = batch['label']

        preds = self(x)

        if self.num_classes == 1:
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
        dataset = HotspotSatelliteDataset(catalogs=data_path, include_lc=self.include_lc, external_features_as_channels=self.external_features_as_channels, crop_size=self.crop_size)  
        

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




class ResNetClassifierE2E(ResNetClassifier):
    def __init__(self, 
                 num_classes, 
                 resnet_version, 
                 train_paths: list, 
                 val_paths: list, 
                 test_paths=None, 
                 optimizer="adam", 
                 scheduler="plateau", 
                 lr=0.001, 
                 batch_size=16, 
                 transfer=False, 
                 tune_fc_only=False, 
                 include_lc=True, 
                 crop_size=32,
                 num_channels=32,
                 external_features_as_channels=False,
                 num_workers = 1,
                 external_features = 13,
                 class_weight = 1.0):
        super().__init__(num_classes, resnet_version, train_paths, val_paths, test_paths, optimizer, scheduler, lr, batch_size, transfer, tune_fc_only, include_lc, crop_size, num_channels, external_features_as_channels, num_workers, class_weight=class_weight)

        
        # add new_forward function to the resnet instance as a class method
        bound_method = self.__resnet_end_to_end_forward.__get__(self.resnet_model, self.resnet_model.__class__)
        setattr(self.resnet_model, 'forward', bound_method)

        # add a fully connected layer compatible with the new features
        linear_size = list(self.resnet_model.children())[-1].in_features + external_features
        # replace final layer for fine tuning
        self.resnet_model.fc = nn.Linear(linear_size, num_classes)
        self.BN = torch.nn.BatchNorm1d(linear_size)
        self.mlp = MLP(in_channels=linear_size, hidden_channels=[256, 128, 1])

    
    @staticmethod
    def __resnet_end_to_end_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # concatenate x and f
        # x = torch.concatenate((x,f), axis=-1)
        # x = self.fc(x)

        return x

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
        dataset = HotspotSatelliteDataset(catalogs=data_path, include_lc=self.include_lc, external_features=True)
        if self.batch_size > len(dataset):
            return DataLoader(dataset, batch_size=len(dataset), shuffle=shuffle, drop_last=True, num_workers=self.num_workers), len(dataset)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=True, num_workers=self.num_workers), len(dataset)

    def forward(self, X, F):
        x = self.resnet_model(X)
        x = torch.concatenate((x,F), axis=-1)
        x = self.BN(x)
        x = self.mlp(x)
        return x


    def _step(self, batch):
        
        x = batch['img']
        f = batch['feat']
        y = batch['label']

        preds = self(x, f)

        if self.num_classes == 1:
            preds = preds.flatten()
            y = y.float()

        loss = self.loss_fn(preds, y)

        acc = self.acc((sigmoid(preds) > 0.5).long(), y)
        f1 = self.f1((sigmoid(preds) > 0.5).long(), y)
        return loss, acc, f1




