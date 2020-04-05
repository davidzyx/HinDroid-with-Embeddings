import sys
import json

from src import *


DATA_PARAMS = 'config/data-params.json'
TEST_PARAMS = 'config/test-params.json'


def load_params(fp):
    with open(fp) as fh:
        param = json.load(fh)

    return param


def main(targets):

    if 'data' in targets:
        cfg = load_params(DATA_PARAMS)
    elif 'data-test' in targets:
        cfg = load_params(TEST_PARAMS)
    else:
        return

    if 'clean' in targets:
        return

    if 'ingest' in targets:
        pass

    if 'process' in targets:
        pass

    if 'model' in targets:
        pass

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
