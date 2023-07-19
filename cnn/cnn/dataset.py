import pandas as pd
from torch.utils.data import Dataset
import rasterio
from torchvision.transforms import CenterCrop, Normalize, Compose
import numpy as np
import torch
from typing import List, Optional
from shapely import Point
from rasterio.transform import AffineTransformer
from rasterio.windows import Window


class HotspotSatelliteDataset(Dataset):
    def __init__(self, catalogs: List[str], include_S2:bool = False, include_lc=True, crop_size=32, external_features=False, external_features_as_channels = False) -> None:

        super(HotspotSatelliteDataset).__init__()

        self.crop_size = crop_size
        self.include_lc = include_lc
        self.lc_file = "data/LC/ann-9c-lc2018.tiff"
        self.basepath = "data/data/"
        self.featpath = "data/dataset_500k.csv"
        self.external_features = external_features
        self.external_features_as_channels = external_features_as_channels


        # Merge catalogs
        dfs = [pd.read_csv(catalog) for catalog in catalogs]
        dfs = [df.set_index('hotspot_id') for df in dfs]
        self.catalog = pd.concat(dfs)

        self.category_list = ['S3_float32_OLCI','S3_float32_SLSTR_reflectance','S3_float32_SLSTR_brightness_temperature']
        if include_S2:
            self.category_list.append('S2_float32_L2A')

        if self.external_features or self.external_features_as_channels:
            self.features_list = ['daynight', 'week_sin', 'week_cos', 'frp', 't_21', 't_31', 't_m13', 't_m15', 't_i4', 't_i5', 'count_12h', 'count_24h', 'count_36h']
            self.features_dataset = pd.read_csv(self.featpath)
            self.FEAT_MEAN = [0.48950179966185026,
                            -0.12152517228040097,
                            -0.27011707828630216,
                            11.786910950449661,
                            28.525963052475444,
                            26.36693787795492,
                            28.52670698185816,
                            26.306758015496797,
                            268.34770719968463,
                            240.58613335131264,
                            2.8997334218469613,
                            3.947883452446399,
                            4.542381777256838]
            self.FEAT_STD = [0.4998902941612639,
                            0.7500115021483671,
                            0.5913992185528857,
                            49.49397087775578,
                            91.3575612323558,
                            84.34277613985807,
                            91.76688846104926,
                            84.50596489896633,
                            125.9911745040769,
                            112.37712189052024,
                            4.0297914915436,
                            5.3539329267138225,
                            6.43907373539704]
            self.transform_norm_feat= Compose([Normalize(self.FEAT_MEAN, self.FEAT_STD)])

        self.IMG_MEANS = [0.27508422090348406, 0.26320713002213203, 0.24018299935894125,  0.21779024293917984,
        0.21098833309526324,  0.20834972626068543,  0.2178419050120043, 
        0.23421741304091712, 0.0,
        0.23744300718799402,  0.23966075079577784,  0.2561276654471364, 
        0.3079022514751455,  0.09147371212127992, 0.15945767185533005, 
        0.2786307577864053,  0.3138295998935744,  0.3320055142727458, 
        0.3332900938172262,  0.24618422394770345, 0.10947266567836743,
        0.0,0.0,0.0,0.0, 0.09810729308845423, 0.23525031929406434, 
        282.63800234382194, 281.7805197635508,  280.94142792097006, 285.4887733896844,
        281.79883226902774, 51.93786965676766]

        self.IMG_STDS = [0.12598678373147143, 0.12775161103438984, 0.13182377684310373, 
                    0.13583471311826467,  0.13481306786455036, 
                    0.13243560762505993, 0.14484251807502851,
                    0.15953063241621113, 1.0,
                    0.16242989013175313,  0.1637262060715902,
                    0.15734837152594128,  0.15840197628803812, 
                    0.05901783211882027, 0.09166365233021094, 
                    0.14440586722802007,  0.15805961758878387,
                    0.15907776419854294, 0.15888801865402202,
                    0.12900596635872155,  0.09249286493209616,
                    1.0,1.0,1.0,1.0,
                    0.6215010943046356, 1.526181215340772, 13.670012569505534, 15.584643484269382,
                    15.573991734944194, 18.09143521176135,  15.561235612514862, 131.846300661227]
        self.transform_norm = Compose([
            Normalize(self.IMG_MEANS, self.IMG_STDS),
            CenterCrop(size=self.crop_size)
        ])
            
    def __len__(self):
        return len(self.catalog)


    def read_landcover(self, point: Point, width, height):

        with rasterio.open(self.lc_file) as src:
            x,y = AffineTransformer(src.transform).rowcol(point.x, point.y)
            return src.read(window=Window(x, y, width=width, height=height))


    def get_hotspot_id(self, idx):
        return self.catalog.iloc[idx]._name

    def __getitem__(self, idx):
        # get S2 and S3 files path
        label = int(self.catalog.iloc[[idx]]["is_positive"].values[0])

        images = []
        
        for cat in self.category_list:
            # read img
            with rasterio.Env(CPL_DEBUG=False):
                images.append(rasterio.open(f"{self.basepath}/{eval(self.catalog.iloc[[idx]][cat].values[0])}").read())
      
        if self.include_lc:
            lat = self.catalog.iloc[[idx]]["lat"].values[0]
            lon = self.catalog.iloc[[idx]]["lon"].values[0]
            
            images.append(self.read_landcover(Point(lon, lat), images[0].shape[2], images[0].shape[1]))
        
        if self.external_features_as_channels:
            for f in self.features_list:
                images.append(np.ones((1,images[0].shape[1],images[0].shape[2]),dtype=np.float32)*self.features_dataset[self.features_dataset["hotspot_id"] == self.catalog.iloc[idx]._name][f].values[0])

        # read and stack on the same array
        img = np.vstack(images)
        img = torch.from_numpy(img)
        # remove NaN values
        img[torch.isnan(img)] = 0
        
        # mean = img.mean([1,2])
        # std = img.std([1,2])
        # std[std==0]=1

        
        
        img = self.transform_norm(img)
        h_id = self.catalog.iloc[idx]._name
        if not self.external_features or self.external_features_as_channels:
            return dict(img=img, label=label, h_id=h_id)
        
        feat = []
        
        for f in self.features_list:
            feat.append(self.features_dataset[self.features_dataset["hotspot_id"] == self.catalog.iloc[idx]._name][f].values[0])
            
        feat = self.transform_norm_feat(torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0).permute((2,0,1))).squeeze().squeeze()

        return dict(feat=feat, img=img, label=label, h_id=h_id)
    

    



class HotspotSatelliteFeaturesDataset(Dataset):
    def __init__(self, catalogs: List[str]) -> None:

        super(HotspotSatelliteDataset).__init__()

        
        self.lc_file = "data/LC/ann-9c-lc2018.tiff"
        self.featpath = "data/dataset_500k.csv"
        dfs = [pd.read_csv(catalog) for catalog in catalogs]
        dfs = [df.set_index('hotspot_id') for df in dfs]
        self.catalog = pd.concat(dfs)

        self.features_list = [  'frp', 't_21', 't_31',
                                't_m13', 't_m15', 't_i4',
                                't_i5',  'daynight', 'week_sin', 'week_cos',
                                'LC', 'count_12h', 'count_24h', 'count_36h',
                                'S3_float32_OLCI_0', 'S3_float32_OLCI_1', 'S3_float32_OLCI_2',
                                'S3_float32_OLCI_3', 'S3_float32_OLCI_4', 'S3_float32_OLCI_5',
                                'S3_float32_OLCI_6', 'S3_float32_OLCI_7', 'S3_float32_OLCI_9',
                                'S3_float32_OLCI_10', 'S3_float32_OLCI_11', 'S3_float32_OLCI_12',
                                'S3_float32_OLCI_13', 'S3_float32_OLCI_14', 'S3_float32_OLCI_15',
                                'S3_float32_OLCI_16', 'S3_float32_OLCI_17', 'S3_float32_OLCI_18',
                                'S3_float32_OLCI_19', 'S3_float32_OLCI_20',
                                'S3_float32_SLSTR_reflectance_4', 'S3_float32_SLSTR_reflectance_5',
                                'S3_float32_SLSTR_brightness_temperature_0',
                                'S3_float32_SLSTR_brightness_temperature_1',
                                'S3_float32_SLSTR_brightness_temperature_2',
                                'S3_float32_SLSTR_brightness_temperature_3',
                                'S3_float32_SLSTR_brightness_temperature_4'
                            ]
            
        self.features_dataset = pd.read_csv(self.featpath)
            
        # DA RICALCOLARE
        self.FEAT_MEAN = [11.786910950449661, 28.525963052475444, 26.36693787795492, 28.52670698185816, 26.306758015496797, 268.34770719968463, 240.58613335131264, 0.48950179966185026, -0.12152517228040097, -0.27011707828630216, 51.93786965676766, 2.8997334218469613, 3.947883452446399, 4.542381777256838, 0.27508422090348406, 0.26320713002213203, 0.24018299935894125, 0.21779024293917984, 0.21098833309526324, 0.20834972626068543, 0.2178419050120043, 0.23421741304091712, 0.23744300718799402, 0.23966075079577784, 0.2561276654471364, 0.3079022514751455, 0.09147371212127992, 0.15945767185533005, 0.2786307577864053, 0.3138295998935744, 0.3320055142727458, 0.3332900938172262, 0.24618422394770345, 0.10947266567836743, 0.09810729308845423, 0.23525031929406434, 282.63800234382194, 281.7805197635508, 280.94142792097006, 285.4887733896844, 281.79883226902774]
        self.FEAT_STD = [49.49397087775578, 91.3575612323558, 84.34277613985807, 91.76688846104926, 84.50596489896633, 125.9911745040769, 112.37712189052024, 0.4998902941612639, 0.7500115021483671, 0.5913992185528857, 131.846300661227, 4.0297914915436, 5.3539329267138225, 6.43907373539704, 0.12598678373147143, 0.12775161103438984, 0.13182377684310373, 0.13583471311826467, 0.13481306786455036, 0.13243560762505993, 0.14484251807502851, 0.15953063241621113, 0.16242989013175313, 0.1637262060715902, 0.15734837152594128, 0.15840197628803812, 0.05901783211882027, 0.09166365233021094, 0.14440586722802007, 0.15805961758878387, 0.15907776419854294, 0.15888801865402202, 0.12900596635872155, 0.09249286493209616, 0.6215010943046356, 1.526181215340772, 13.670012569505534, 15.584643484269382, 15.573991734944194, 18.09143521176135, 15.561235612514862]
        
        self.transform_norm_feat= Compose([Normalize(self.FEAT_MEAN, self.FEAT_STD)])
    
    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        # get S2 and S3 files path
        label = int(self.catalog.iloc[[idx]]["is_positive"].values[0])
        
        feat = []
        
        for f in self.features_list:
            feat.append(self.features_dataset[self.features_dataset["hotspot_id"] == self.catalog.iloc[idx]._name][f].values[0])
            
        feat = self.transform_norm_feat(torch.FloatTensor(feat).unsqueeze(0).unsqueeze(0).permute((2,0,1))).squeeze().squeeze()

        return dict(feat=feat, label=label)
    

    

