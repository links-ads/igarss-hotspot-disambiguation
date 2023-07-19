import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
from pathlib import Path 
import os
import json
from tqdm import tqdm

PATH = Path("data/cache/files_mapping")
COLUMNS = ["hotspot_id", "list_id", "S2_float32_L2A", "S3_float32_OLCI", "S3_float32_SLSTR_reflectance", "S3_float32_SLSTR_brightness_temperature"]




for file in tqdm(os.listdir(PATH)):
    if os.path.isfile(PATH / file):
        data = json.load(open(PATH / file))
        list_id = file.split(".")[0]
        if os.path.isfile(f"data/cache/catalogs/catalog_{list_id}.csv"):
            continue
        df = pd.DataFrame(columns=COLUMNS, index=["hotspot_id"])
    for id,files in tqdm(data.items()):
        row = dict(
            hotspot_id = id,
            list_id=list_id, 
            S2_float32_L2A=None,
            S3_float32_OLCI=None, 
            S3_float32_SLSTR_reflectance=None, 
            S3_float32_SLSTR_brightness_temperature=None
        )
        for f in files:
            # print(f)
            row[f["source"]] = f["files"]


        df = df.append(row, ignore_index=True)
    df.to_csv(f"data/cache/catalogs/catalog_{list_id}.csv", index=False)
    



