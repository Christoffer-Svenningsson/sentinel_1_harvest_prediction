import pyproj
import shutil
import glob
import os
import re
import gzip

SWEDEN = "resources/swedish_border.geojson"

HARVEST_17_20_AOI_FILE = "resources/aoi_year_17_to_20.geojson"

HARVEST_20_AOI_FILE = "/mimer/NOBACKUP/groups/snic2022-23-428/shared_oliver_christoffer/aois/aoi_2020.geojson"

SENTINEL_1_11x11_SAR2SAR_V3_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/sentinel_1_data/despeckle_sar2sarV3_res_11m"

SENTINEL_1_11x11_PATH = "raw_sentinel_11m"

SHARED_FOLDER_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/shared_oliver_christoffer"

DATASETS_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/shared_oliver_christoffer/datasets"

FIG_TEST_DIFFERENT_SIZE_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/Oliver/figures"

NPY_TEST_DIFFERENT_SIZE_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/Oliver/npy"

FEATURES_SELECTED_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/Oliver/selected_features"

RESULTS_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/Oliver/results"

EXPERIMENTS_PATH = "/mimer/NOBACKUP/groups/snic2022-23-428/Oliver/experiments"

UTM33N = pyproj.CRS("EPSG:32633")

def find_eo_patches(year):
    paths = glob.glob(os.path.join(year, '*'))
    pattern = r'.*eopatch_(\d+)'
    patch_ids = []
    for p in paths:
        m = re.search(pattern, p)
        patch_ids.append(int(m.group(1)))
    return patch_ids, paths

def extract_eo_patch_data(patch_ids, year_path):
    #Extract fetched data from zipped folders
    for idx in patch_ids:
        eopatch_path = os.path.join(year_path, f'eopatch_{idx}')
        if os.path.exists(os.path.join(eopatch_path, "data", "IW.npy")):
            print(f"Skipping EO patch {idx}, already unzipped!")
            continue

        paths = glob.glob(os.path.join(eopatch_path, '*'))
        paths = [path for path in paths if not os.sep + "data" in path]
        paths += [os.path.join(eopatch_path,"data" + os.sep + "IW.npy.gz")]    
        for path in paths:
            with gzip.open(path,'rb') as f_in:
                with open(path.replace('.gz', ''), 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
