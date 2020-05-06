import os
import sys
from glob import glob
import pandas as pd
# !pip install multiprocess
from p_tqdm import p_umap

import src.utils as utils
from src.features.smali import SmaliApp, HINProcess
# from src.features.app_features import FeatureBuilder


def is_large_dir(app_dir, size_in_bytes=1e8):
    if utils.get_tree_size(app_dir) > size_in_bytes:
        return True
    return False


def process_app(app_dir, out_dir):
    if is_large_dir(app_dir):
        print(f'Error {app_dir} too big')
        return None
    try:
        app = SmaliApp(app_dir)
        out_path = os.path.join(out_dir, app.package + '.csv')
        app.info.to_csv(out_path, index=None)
        package = app.package
        del app
    except Exception as e:
        print(f'Error extracting {app_dir}')
        print(e)
        return None
    return package, out_path


def extract_save(in_dir, out_dir, class_i, nproc):
    app_dirs = glob(os.path.join(in_dir, '*/'))
    assert len(app_dirs) > 0, in_dir

    print(f'Extracting features for {class_i}')

    meta = p_umap(process_app, app_dirs, [out_dir for i in range(len(app_dirs))], num_cpus=nproc, file=sys.stdout)
    meta = [i for i in meta if i is not None]
    packages = [t[0]for t in meta]
    csv_paths = [t[1]for t in meta]
    return packages, csv_paths


def build_features(**config):
    """Main function of data ingestion. Runs according to config file"""
    # Set number of process, default to 2
    nproc = config['nproc'] if 'nproc' in config.keys() else 2
    test_size = config['test_size'] if 'test_size' in config.keys() else 0.67

    csvs = []
    apps_meta = []

    for cls_i in utils.ITRM_CLASSES_DIRS.keys():
        raw_dir = utils.RAW_CLASSES_DIRS[cls_i]
        itrm_dir = utils.ITRM_CLASSES_DIRS[cls_i]

        # Look for processed csv files, skip extract step
        csv_paths = glob(f'{itrm_dir}/*.csv')
        if len(csv_paths) > 0:
            print('Found previously generated CSV files')
            packages = [p[:-4] for p in csv_paths]
        else:
            print('Previous extracted CSV files not found/complete')
            packages, csv_paths = extract_save(raw_dir, itrm_dir, cls_i, nproc)

        # Sort meta by package name for consistent index
        di = dict(zip(packages, csv_paths))
        for package, csv_path in sorted(di.items()):
            apps_meta.append((cls_i, package, csv_path,))
            csvs.append(csv_path)

    print('Total number of csvs:', len(csvs))
    hin = HINProcess(csvs, utils.PROC_DIR, nproc=nproc, test_size=test_size)
    hin.run()

    meta = pd.DataFrame(
        apps_meta,
        columns=['label', 'package', 'csv_path']
    )

    meta_train = meta.iloc[hin.tr_apps, :]
    meta_train.index = [f'app_{i}' for i in range(len(meta_train))]
    meta_train.to_csv(os.path.join(utils.PROC_DIR, 'meta_tr.csv'))

    meta_tst = meta.iloc[hin.tst_apps, :]
    meta_tst.index = [f'app_{i + len(meta_train)}' for i in range(len(meta_tst))]
    meta_tst.to_csv(os.path.join(utils.PROC_DIR, 'meta_tst.csv'))
