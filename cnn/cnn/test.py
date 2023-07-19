import pytorch_lightning as pl
from argparse import ArgumentParser
import pytorch_lightning as pl
from resnet import ResNetClassifierE2E, ResNetClassifier
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

parser = ArgumentParser()
# Required arguments
parser.add_argument(
    "--file", help="""Name of ckpt file.""", type=str
)
parser.add_argument(
        "--dataset_path", help="""Path to data folder.""", type=Path
)
parser.add_argument("--fold", help="""Fold to test on.""", type=int)
args = parser.parse_args()
# "logs/E2E_CHANNELS_MLP_OVERSAMPLINGmodelAdamW-fold-k=0-epoch=0-val_loss=0.12-val_acc=0.95.ckpt"
logger = TensorBoardLogger("lightning_logs_correct", name=f"TEST-{args.file.split('.')[0]}")
        
trainer_args = {
           "accelerator": "gpu",
            "devices": [0],
            "strategy": "auto",
            "max_epochs": 20,
            "precision": 32,
            "logger": logger
        }
k = args.fold
val = [args.dataset_path / f"K{k}.csv"]
train = [args.dataset_path / f"K{i}.csv" for i in range(3) if i != k]
test = [args.dataset_path / "KTEST.csv"]
model = ResNetClassifier(1, 18, train_paths=train, val_paths=val, test_paths=test, external_features_as_channels=True, num_channels=32+13, use_mlp=True)
trainer = pl.Trainer(**trainer_args)
        
trainer.test(model, ckpt_path=args.file)