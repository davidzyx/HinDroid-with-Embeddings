import sys
import json

from src.utils import prep_dir, clean_raw, clean_features, clean_processed
from src.data.get_data import get_data
from src.features.build_features import build_features, reduce_apis
from src.models import hindroid
from src.features.n2v import node2vec_main
from src.features.w2v import word2vec_main
from src.features.m2v import metapath2vec_main
import src.features.metaPrediction

DATA_PARAMS = 'config/data-params.json'
TEST_PARAMS = 'config/test-params.json'


def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param


def main(targets):

    if 'test-project' in targets:
        cfg = load_params(TEST_PARAMS)
        prep_dir(**cfg)
        get_data(**cfg)
        build_features(**cfg)
        reduce_apis()
        node2vec_main(**cfg)
        word2vec_main(**cfg)
        metapath2vec_main(**cfg)
        return

    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
    elif 'data-test' in targets:
        cfg = load_params(TEST_PARAMS)
    else:
        return

    prep_dir(**cfg)

    if 'clean' in targets:
        clean_raw(**cfg)
        clean_features(**cfg)
        clean_processed(**cfg)
        return

    if 'ingest' in targets:
        get_data(**cfg)

    if 'process' in targets:
        build_features(**cfg)

    if 'reduce' in targets:
        reduce_apis()

    if 'node2vec' in targets:
        node2vec_main(**cfg)

    if 'word2vec' in targets:
        word2vec_main(**cfg)

    if 'model' in targets:
        hindroid.run(**cfg)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
