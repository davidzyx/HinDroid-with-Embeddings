import os
import shutil
from collections import defaultdict
import numpy as np

RAW_DIR = None
ITRM_DIR = None
PROC_DIR = None
RAW_CLASSES_DIRS = None
ITRM_CLASSES_DIRS = None


def prep_dir(**cfg):
    """Prepare necessary directory structure inside data_dir"""

    global RAW_DIR
    global ITRM_DIR
    global PROC_DIR
    global RAW_CLASSES_DIRS
    global ITRM_CLASSES_DIRS

    if not os.path.exists(cfg['data_dir']):
        os.mkdir(cfg['data_dir'])

    RAW_DIR = os.path.join(cfg['data_dir'], cfg['data_subdirs']['raw'])
    ITRM_DIR = os.path.join(cfg['data_dir'], cfg['data_subdirs']['interim'])
    PROC_DIR = os.path.join(cfg['data_dir'], cfg['data_subdirs']['processed'])
    data_classes = cfg['data_classes']

    RAW_CLASSES_DIRS = {
        class_i: os.path.join(RAW_DIR, class_i)
        for class_i in data_classes.keys()
    }
    ITRM_CLASSES_DIRS = {
        class_i: os.path.join(ITRM_DIR, class_i)
        for class_i in data_classes.keys()
    }

    dir_ls = [RAW_DIR, ITRM_DIR, PROC_DIR] + \
        list(RAW_CLASSES_DIRS.values()) + \
        list(ITRM_CLASSES_DIRS.values())

    for dir_i in dir_ls:
        if not os.path.exists(dir_i):
            os.mkdir(dir_i)


def clean_raw(**cfg):
    data_dir = cfg['data_dir']
    raw_dir = os.path.join(data_dir, cfg['data_subdirs']['raw'])
    shutil.rmtree(raw_dir, ignore_errors=True)


def clean_features(**cfg):
    data_dir = cfg['data_dir']
    itrm_dir = os.path.join(data_dir, cfg['data_subdirs']['interim'])
    shutil.rmtree(itrm_dir, ignore_errors=True)


def clean_processed(**cfg):
    data_dir = cfg['data_dir']
    itrm_dir = os.path.join(data_dir, cfg['data_subdirs']['processed'])
    shutil.rmtree(itrm_dir, ignore_errors=True)


class UniqueIdAssigner():
    class MapGetter():
        def __init__(self, mapping):
            self.mapping = mapping

        def __getitem__(self, key):
            if type(self.mapping) == list:
                return self.mapping[key]
            if key not in self.mapping.keys():
                raise KeyError(key)
            return self.mapping[key]

    def __init__(self):
        self.assigner = defaultdict(lambda: len(self.assigner))
        self.value_by_id = []
        self.id_of = self.MapGetter(self.assigner)
        self.value_of = self.MapGetter(self.value_by_id)

    def add(self, *values):
        uids = [self.assigner[v] for v in values]
        self.value_by_id.clear()
        self.value_by_id.extend(self.assigner.keys())
        return uids

    def __getitem__(self, k):
        return self.value_by_id[k]

    def __len__(self):
        return len(self.assigner)


# From https://www.python.org/dev/peps/pep-0471/#examples
def get_tree_size(path):
    """Return total size of files in given path and subdirs."""
    total = 0
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            total += get_tree_size(entry.path)
        else:
            total += entry.stat(follow_symlinks=False).st_size
    return total


# From https://stackoverflow.com/a/47171600
def replace_with_dict(arr, dic):
    """Replace an array's values with dictionary"""
    # Extract out keys and values
    k = np.array(list(dic.keys()))
    v = np.array(list(dic.values()))

    # Get argsort indices
    sidx = k.argsort()

    ks = k[sidx]
    vs = v[sidx]
    return vs[np.searchsorted(ks, arr)]