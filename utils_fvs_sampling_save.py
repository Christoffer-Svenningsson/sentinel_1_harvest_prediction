from enum import Enum, auto
import re
import numpy as np
import os
import pickle
from collections import namedtuple
from decimal import Decimal as D
from decimal import getcontext, ROUND_HALF_UP
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from util_paths import NPY_TEST_DIFFERENT_SIZE_PATH
from util_paths import DATASETS_PATH

# ----------------- Fvs and Sampling
class FeatureVector(Enum):
    All = auto()
    FeatureImportance = auto()
    VH_VV = auto()
    Weather = auto()
    Topology = auto()
    Height = auto()

def is_part_of_main_experiment(feature: FeatureVector) -> bool:
    return (
        feature is FeatureVector.All
        or feature is FeatureVector.FeatureImportance
        or feature is FeatureVector.VH_VV
        or feature is FeatureVector.Weather
        or feature is FeatureVector.Topology
        or feature is FeatureVector.Height
    )

class Sampling(Enum):
    Grid_3x3_50m = auto()
    Rfi_Grid_50m = auto()
    Nearest_50m = auto()
    Nearest_22m = auto()
    Nearest_12m = auto()
    
sampling_2_datasetpath_and_name = {
    Sampling.Grid_3x3_50m: {
        "path": os.path.join(DATASETS_PATH, "dataset_y_2017_2020_alot2.feather"), 
        "name": "alot2"
    },
    Sampling.Rfi_Grid_50m: {
        "path": os.path.join(DATASETS_PATH, "dataset_y_2017_2020_rfi_filtered.feather"),
        "name": "rfi_filtered"
    },
    Sampling.Nearest_50m: {
        "path": os.path.join(DATASETS_PATH, "dataset_y_2017_2020_alot2.feather"), 
        "name": "alot2"
    },
    Sampling.Nearest_22m: {
        "path": os.path.join(DATASETS_PATH, "dataset_y_2017_2020_22m_hd.feather"),
        "name": "hd_22m"
    },
    Sampling.Nearest_12m: {
        "path": os.path.join(DATASETS_PATH, "dataset_y_2017_2020_12m_hd.feather"),
        "name": "hd_12m"
    }
}

def get_fvs(sampling: Sampling):
    if sampling == Sampling.Grid_3x3_50m:
        return [feature for feature in list(FeatureVector) if is_part_of_main_experiment(feature) and feature != FeatureVector.Weather]
    
    elif sampling == Sampling.Rfi_Grid_50m:
        return [FeatureVector.All]
    
    elif sampling == Sampling.Nearest_50m:
        return [feature for feature in list(FeatureVector) if is_part_of_main_experiment(feature)]
    
    return [feature for feature in list(FeatureVector) if is_part_of_main_experiment(feature) and feature != FeatureVector.FeatureImportance]
        

# ----------------- bands

class Indices(Enum):
    VH_VV_Pure = auto()
    VH_VV_Ratios = auto()
    All = auto()
    
def is_vh_vv(col: str)-> bool:
    return (
        "vh_week" in col
        or "vv_week" in col
    ) and "/" not in col

def is_vh_vv_and_ratios(col: str)-> bool:
    return (
        "vh" in col
        or "vv" in col
    )

def is_vh_vv_rvi(col: str)-> bool:
    return (
        "vh" in col
        or "vv" in col
        or "rvi" in col
    )
    
def is_indices(col: str)-> bool:
    return (
        "rvi" in col 
        or "mi" in col
    )

def is_rvi(col: str)-> bool:
    return "rvi" in col

def is_sentinel_data(col: str)-> bool:
    return is_vh_vv_and_ratios(col) or is_indices(col)

# ----------------- topology

neighbours_col_names = [
    "c",
    "n",
    "ne",
    "e",
    "se",
    "s",
    "sw",
    "w",
    "nw"
]


topology_classes = [
    "flat",
    "peak",
    "ridge",
    "shoulder",
    "spur",
    "slope",
    "hollow",
    "footslope",
    "valley",
    "pit"
]

def is_topology(col: str) -> bool:
    pattern = r"(\D+)_(\D+)"
    res = re.search(pattern, col)
    
    if res is None:
        return False
    
    n, t = res.groups()
    return n in neighbours_col_names and t in topology_classes

def is_int_topology(col: str) -> bool:
    res = re.search('(\D+)_inttopology',col)
    if res is None:
        return False
    return res.groups()[0] in neighbours_col_names

def is_height(col: str)-> bool:
    res = re.search(r'(\D+)_height', col)
    if res == None:
        return False
    return res.groups()[0] in neighbours_col_names

# ----------------- weather

weather_col_names = [
    "acc_sun_wpm2",
    "acc_rain_sum",
    "avg_percent_humidity",
    "avg_wind_speed",
    "avg_temp"
]
   
def is_weather(col: str) -> bool:
    pattern = r"week_\d+_(.+)"
    res = re.search(pattern, col)
    
    if res is None:
        return False
    
    p = res.groups()
    return p[0] in weather_col_names

# ----------------- scaling

def is_minmax_scale(col: str)-> bool:
    return is_vh_vv_and_ratios(col) or is_rvi(col) or is_weather(col) or is_height(col)

def is_standard_scale(col: str)-> bool:
    return "mi" in col

def normalize(X_train, X_test=None):
    cols_minmax = [col for col in X_train.columns if is_minmax_scale(col)]
    if len(cols_minmax) > 0:
        minmax_scaler = MinMaxScaler().fit(X_train[cols_minmax])
        # Warnings here, unclear why. The dataframe gets updated values
        X_train.loc[:, cols_minmax] = minmax_scaler.transform(X_train[cols_minmax])
        if X_test is not None:
            X_test.loc[:, cols_minmax] = minmax_scaler.transform(X_test[cols_minmax])
        
    cols_standard = [col for col in X_train.columns if is_standard_scale(col)]
    if len(cols_standard) > 0:
        standard_scaler = StandardScaler().fit(X_train[cols_standard])
        # Warnings here, unclear why. The dataframe gets updated values
        X_train.loc[:, cols_standard] = standard_scaler.transform(X_train[cols_standard])
        if X_test is not None:
            X_test.loc[:, cols_standard] = standard_scaler.transform(X_test[cols_standard])
        
# ----------------- Random

def is_in_all(col: str) -> bool:
    return is_vh_vv(col) or is_weather(col) or is_topology(col) or is_height(col)

def is_nearest_sampling(col: str)-> bool:
    if is_vh_vv(col) or is_topology(col) or is_height(col):
        return col[0] == "c" 
    
    return is_weather(col) or col == "harvest" 

def my_round(nbr, digits=2):
    ctx = getcontext()
    ctx.rounding = ROUND_HALF_UP
    return round(D(nbr), digits)


Result = namedtuple("Result", ["rmses", "accs", "f1s", "model_type", "fv_type", "year"])
AvgScore = namedtuple("AvgScore", ["rmse_mean", "accs_mean", "f1s_mean"])


def save(info, path, name, f_format):
    if not os.path.exists(path):
        os.makedirs(path)
        
    file = os.path.join(path, f"{name}.{f_format}")
    i = 1
    
    while os.path.exists(file):
        file = os.path.join(path, f"{name}-{i}.{f_format}")
        i += 1
        #print(f"File already exists, trying adding i = {i} to ending.")
    
    with open(file, "wb") as f:    
        if f_format == "npy":
            np.save(f, info)
            
        else:
            pickle.dump(info, f)

