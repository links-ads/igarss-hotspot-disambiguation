import pytorch_lightning as pl
from argparse import ArgumentParser
import pytorch_lightning as pl
from resnet import ResNetClassifierE2E, ResNetClassifier
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from dataset import HotspotSatelliteDataset
import pandas as pd
from tqdm import tqdm  
from torch.utils.data import DataLoader

parser = ArgumentParser()
# Required arguments
parser.add_argument(
    "--file", help="""Name of ckpt file.""", type=str
)
parser.add_argument(
        "--dataset_path", help="""Path to data folder.""", type=Path
)
args = parser.parse_args()

ds = HotspotSatelliteDataset(catalogs=[args.dataset_path / f"K{i}.csv" for i in range(3)])

model = ResNetClassifier.load_from_checkpoint(args.file, 
                                                num_classes=1,
                                                resnet_version=18,
                                                train_paths=[],
                                                val_paths= [],return_logits=True)
# trainer = pl.Trainer(**trainer_args)
        
# trainer.test(model, ckpt_path=args.file)

model.eval()
# create dataloader
dl = DataLoader(dataset=ds, batch_size=128, shuffle=False,  drop_last=True, num_workers=16)

results = []
ext_feat_df = pd.DataFrame(columns=["hotspot_id", "feat"])

for step, batch in enumerate(tqdm(dl)):
    x = batch["img"].to("cuda")
    h_id = batch["h_id"]
    res = model(x).squeeze().detach().cpu().numpy().tolist()
    data = {"hotspot_id": h_id, "feat": res}
    ext_feat_df = ext_feat_df.append(data, ignore_index=True)
    if step % 10:
        ext_feat_df.to_csv("data/extracted_s3_features.csv", index=False)

    
