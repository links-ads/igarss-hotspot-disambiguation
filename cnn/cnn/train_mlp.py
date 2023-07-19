from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import torch 
import datetime
from mlp import MLPClassifier
from lightning.pytorch.loggers import TensorBoardLogger


if __name__ == "__main__":
    parser = ArgumentParser()
    # Required arguments
    
    
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
    
    
    args = parser.parse_args()

    num_splits = 3
    test = [args.dataset_path / "KTEST.csv"]
    # we work on splits already saved on csv files
    for k in range(num_splits):
        val = [args.dataset_path / f"K{k}.csv"]
        train = [args.dataset_path / f"K{i}.csv" for i in range(num_splits) if i != k]
        # # Instantiate Model
        num_channels = 41
        model = MLPClassifier(
                train_paths=train,
                val_paths=val,
                test_paths=test,
                optimizer=args.optimizer,
                scheduler=args.scheduler,
                lr=args.learning_rate,
                batch_size=args.batch_size,
                num_channels=num_channels,
                num_workers=args.num_workers
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