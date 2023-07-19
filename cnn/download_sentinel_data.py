import argparse
from pathlib import Path
import os
import json
from datetime import datetime
from datetime import timedelta
from shapely.geometry import Point
import shapely.wkt
import logging
from tqdm import tqdm
from config.base import prepare_logging
import pandas as pd

from sentinelDownloader.download import ImageRetriever
from hotspotDB.database import HotspotDatabase
from config.schema import Configuration, GeoApiConfig
import threading
BASE = Path("data")
CACHE = BASE / "cache"
CFG = Configuration()
DB = HotspotDatabase(CFG.hotspotdb)
DATA_COLLECTIONS = ["SENTINEL2_L2A", "SENTINEL3_OLCI", "SENTINEL3_SLSTR", "SENTINEL3_SLSTR"]
EVALSCRIPTS = ["S2_float32_L2A.js", "S3_float32_OLCI.js", "S3_float32_SLSTR_reflectance.js", "S3_float32_SLSTR_brightness_temperature.js"]
prepare_logging(log_level=CFG.log_level, log_format=CFG.log_format, suppress=CFG.log_suppress)
LOG = logging.getLogger(__name__)
COLUMNS = ["hotspot_id", "list_id", "S2_float32_L2A", "S3_float32_OLCI", "S3_float32_SLSTR_reflectance", "S3_float32_SLSTR_brightness_temperature"]


class SentinelDownloader(threading.Thread):
    def __init__(self, config: GeoApiConfig, db: HotspotDatabase) -> None:
        super().__init__()
        self.cfg = config
        self.list = pd.read_csv(BASE / config.list).values.tolist()
        self.downloader = ImageRetriever(config)
        self.DB = db
        self.evalscripts = EVALSCRIPTS
        self.data_collections = DATA_COLLECTIONS
        self.cached_ids = {}
        self.files = {}
        self.thread_id = config.id
        for evalscript in self.evalscripts:
            source_id = evalscript.replace(".js", "")
            self.cached_ids[source_id] = []
            if os.path.isfile(CACHE / f"ids_{self.thread_id}_{source_id}.json"):
                self.cached_ids[source_id] = json.load(open(CACHE / f"ids_{self.thread_id}_{source_id}.json"))


    def run(self):
        df = pd.DataFrame(columns=COLUMNS, index=["hotspot_id"])
        for item in self.list:
            h_id, h_time, h_point = self.DB.get_hotspot([item[0]])[0]
            row = dict(
                        hotspot_id = h_id,
                        list_id=self.thread_id, 
                        S2_float32_L2A=None,
                        S3_float32_OLCI=None, 
                        S3_float32_SLSTR_reflectance=None, 
                        S3_float32_SLSTR_brightness_temperature=None
                    )
            self.files[h_id] = []
            geom = shapely.wkt.loads(h_point)
            if h_time.year < 2015:
                continue
            start = h_time - timedelta(hours=12)
            end = h_time + timedelta(hours=12)
            for evalscript, data_collection in zip(self.evalscripts, self.data_collections):
                try:
                    source_id = evalscript.replace(".js", "")
                    if h_id in self.cached_ids[source_id]:
                        continue
                    fs, ds = self.downloader.retrieve_images(evalscript=evalscript, data_collection=data_collection, geometry=geom, min_dims=(32,32), resolution=300, start_date=start, end_date=end, cache=True)
                    
                    self.files[h_id].append({"source": data_collection, "files": fs, "acquistion_dates": ds})
                    row[data_collection] = fs
                    self.cached_ids[source_id].append(h_id)
                    LOG.info(f"[{self.thread_id} - {h_id} - {source_id}] Stored in {fs}")
                except Exception as e:
                    LOG.error(f"[{self.thread_id} - {h_id} - {source_id}] Error: {e}")
            df = df.append(row, ignore_index=True)
        
        json.dump(self.cached_ids[source_id], open(CACHE / f"ids_files_{self.thread_id}_{source_id}.json", "w"))
        
        json.dump(self.files, open(CACHE / f"files_{self.thread_id}.json", "w"))
        df.to_csv(CACHE / f"catalogs/catalog_{self.thread_id}.csv", index=False)

# def parse_args():
#     parser= argparse.ArgumentParser()
#     parser.add_argument('--source', '-s', choices=["modis", "vnp14", "vnp14img"], type=str, help="Data source [modis, vnp14, vnp14img]")
#     return parser.parse_args()


def main():

    # args = parse_args()
    # source = args.source
    # output_folder = Path(CFG.geoapi.cache_dir)
    # output_folder.mkdir(exist_ok=True)
    # hotspot_ids = DB.get_hotspot_data(f"historic_{source}")
    

    # files={}
    # cached_ids = {}

    
    # for evalscript, data_collection in zip(EVALSCRIPTS, DATA_COLLECTIONS):
    #     source_id = evalscript.replace(".js", "")
    #     cached_ids[source_id] = []
    #     if os.path.isfile(CACHE / f"ids_{source}_{source_id}.json"):
    #         cached_ids[source_id] = json.load(open(CACHE / f"ids_{source}_{source_id}.json"))

    
    # for h_id, h_time, h_point in tqdm(hotspot_ids):
    #     geom = shapely.wkt.loads(h_point)
    #     if h_time.year < 2015:
    #         continue
    #     start = h_time - timedelta(hours=12)
    #     end = h_time + timedelta(hours=12)
    #     for evalscript, data_collection in zip(EVALSCRIPTS, DATA_COLLECTIONS):
    #         try:
    #             source_id = evalscript.replace(".js", "")
    #             if h_id in cached_ids[source_id]:
    #                 continue
    #             fs, ds = IMG_DOWNLOADER.retrieve_images(evalscript=evalscript, data_collection=data_collection, geometry=geom, min_dims=(32,32), resolution=60, start_date=start, end_date=end, cache=True)
    #             files[h_id] = {"source": data_collection, "files": fs, "acquistion_dates": ds}
    #             cached_ids[source_id].append(h_id)
    #             LOG.info(f"[{source} - {h_id} - {source_id}] Stored in {fs}")
    #         except Exception as e:
    #             LOG.error(f"[{source} - {h_id} - {source_id}] Error: {e}")
    #             pass
    # json.dump(cached_ids[source_id], open(CACHE / f"ids_files_{source}_{source_id}.json", "w"))
    # json.dump(files, open(CACHE / f"files_{source}.json", "w"))

    threads = []
    for sh_cfg in CFG.geoapis:
        threads.append(SentinelDownloader(sh_cfg, DB))
    
    for th in threads:
            th.start()
    for th in threads:
        th.join()    
    return


if __name__ == "__main__":
    main()
