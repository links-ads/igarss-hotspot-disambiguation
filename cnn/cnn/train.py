from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import torch 
import datetime
import rasterio
from resnet import ResNetClassifier, ResNetClassifierE2E
from lightning.pytorch.loggers import TensorBoardLogger
pl.seed_everything(42, workers=True)

if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    parser.add_argument(
        "--model",
        help="""Choose one of the predefined ResNet models provided by torchvision. e.g. 50""",
        type=int,
    )
    parser.add_argument(
        "--num_classes", help="""Number of classes to be learned.""", type=int
    )
    parser.add_argument("--num_epochs", help="""Number of Epochs to Run.""", type=int)
    parser.add_argument(
        "--dataset_path", help="""Path to data folder.""", type=Path
    )
   
    # Optional arguments
    parser.add_argument(
        "-amp",
        "--mixed_precision",
        help="""Use mixed precision during training. Defaults to False.""",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        help="""PyTorch optimizer to use. Defaults to adam.""",
        default="adam",
    )
    parser.add_argument(
        "-sc",
        "--scheduler",
        help="""PyTorch scheduler to use. Defaults to plateau.""",
        default="plateau",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        help="Adjust learning rate of optimizer.",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="""Manually determine batch size. Defaults to 16.""",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--crop_size",
        help="""Tile crop size. Defaults to 32.""",
        type=int,
        default=32,
    )
    parser.add_argument(
        "-tr",
        "--transfer",
        help="""Determine whether to use pretrained model or train from scratch. Defaults to False.""",
        action="store_true",
    )
    parser.add_argument(
        "-to",
        "--tune_fc_only",
        help="Tune only the final, fully connected layers. Defaults to False.",
        action="store_true",
    )
    parser.add_argument(
        "--exclude_lc",
        help="Wheter exclude land cover data from train. Defaults to False.",
        action="store_true",
    )
    parser.add_argument(
        "-s", "--save_path", help="""Path to save model trained model checkpoint."""
    )
    parser.add_argument(
        "-g", "--gpus", help="""Enables GPU acceleration.""", type=int, default=None
    )
    parser.add_argument(
        "-n", "--name", help="""Experiment name.""", type=str, default="resnet18"
    )
    parser.add_argument(
        "-w", "--num_workers", help="""Num workers for dataloader. Defaults to 64""", type=int, default=64
    )
    parser.add_argument(
        "--end_to_end",
        help="Train End 2 End solution with features pointwise. Defaults to False.",
        action="store_true",
    )
    parser.add_argument(
        "--end_to_end_as_channel",
        help="Train End 2 End solution with features added as channelsto image. Defaults to False.",
        action="store_true",
    )
    parser.add_argument(
        "--use_mlp",
        help="Use MLP. Defaults to True.",
        action="store_true",
    )
    parser.add_argument(
        "--class_weight",
        help="Class weights. Default 1.0",
        type=float,
        default=1.0
    )
    args = parser.parse_args()

    num_splits = 3
    test = [args.dataset_path / "KTEST.csv"]
    # we work on splits already saved on csv files
    # pl.utilities.seed.seed_everything(42, workers=True)
    for k in range(num_splits):
        val = [Path("data/folds/filtered_by_features") / f"K{k}.csv"]
        train = [args.dataset_path / f"K{i}.csv" for i in range(num_splits) if i != k]
        # # Instantiate Model
        num_channels = 32
        if args.end_to_end:
            model = ResNetClassifierE2E(
                    num_classes=args.num_classes,
                    resnet_version=args.model,
                    train_paths=train,
                    val_paths=val,
                    test_paths=test,
                    optimizer=args.optimizer,
                    scheduler=args.scheduler,
                    lr=args.learning_rate,
                    batch_size=args.batch_size,
                    transfer=args.transfer,
                    tune_fc_only=args.tune_fc_only,
                    include_lc=not args.exclude_lc,
                    crop_size=args.crop_size,
                    num_channels=num_channels,
                    num_workers=args.num_workers,
                    external_features_as_channels=False,
                    class_weight=args.class_weight
                )
        else:
            if args.end_to_end_as_channel:
                num_channels = 32 + 13
            model = ResNetClassifier(
                num_classes=args.num_classes,
                resnet_version=args.model,
                train_paths=train,
                val_paths=val,
                test_paths=test,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                transfer=args.transfer,
                tune_fc_only=args.tune_fc_only,
                include_lc=not args.exclude_lc,
                crop_size=args.crop_size,
                num_workers=args.num_workers,
                num_channels=num_channels,
                external_features_as_channels=args.end_to_end_as_channel,
                use_mlp=args.use_mlp,
                class_weight=args.class_weight
            )
            
        
        save_path = args.save_path if args.save_path is not None else "./models"
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path,
            filename= f"{args.name}" + "model"+args.optimizer+"-fold-{k}-{epoch}-{val_loss:.2f}-{val_acc:0.2f}",
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            save_last=True,
        )

        stopping_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
        logger = TensorBoardLogger("lightning_logs_correct", name=f"{args.name}-{args.optimizer}-{args.scheduler}-fold-{k}-{datetime.datetime.now()}")
        # Instantiate lightning trainer and train model
        trainer_args = {
            "accelerator": "gpu" if args.gpus else "cpu",
            "devices": [0] if args.gpus else 1,
            "strategy": "dp" if args.gpus and args.gpus > 1 else "auto",
            "max_epochs": args.num_epochs,
            "callbacks": [checkpoint_callback],
            "precision": 16 if args.mixed_precision else 32,
            "logger": logger
        }
        

        trainer = pl.Trainer(**trainer_args)
        
        trainer.fit(model)
        trainer.test(model)
        
        # Save trained model weights
        torch.save(trainer.model.resnet_model.state_dict(), save_path + f"{args.name}-{args.optimizer}-{args.scheduler}-fold-{k}-{datetime.datetime.now()}-trained_model.pt")