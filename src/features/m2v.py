import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import networkx as nx
import numpy as np
import pandas as pd
import random
import collections
import src.features.metaPrediction as mp

import src.utils as utils

# import logging
# logger = logging.getLogger('debug')
# hdlr = logging.FileHandler('./debug.log', mode='w')
# formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
# hdlr.setFormatter(formatter)
# logger.addHandler(hdlr) 
# logger.setLevel(logging.INFO)

class Metapath2Vec():
    def __init__(self, A_tr, B_tr, P_tr):
        assert 'csr_matrix' in str(type(A_tr))
        self.A_tr_csr = A_tr
        self.A_tr_csc = A_tr.tocsc(copy=True)
        self.B_tr = B_tr.tocsr()
        self.P_tr = P_tr.tocsr()
        
    def chose_idx(self, row):
        next_index = random.choice(np.nonzero(row)[1])
        return next_index
    
    def A_api(self, appNumber, path):
        '''
        Find the next api using matrix A given app
        '''
        row = self.A_tr_csr[appNumber]
        next_index = self.chose_idx(row)
        path.append('api_%d'%next_index)
        return next_index

    def A_app(self, next_index, path):
        '''
        Find the next app using matrix A given api
        '''
#         logger.info('next_index: ' + str(next_index))
        row = self.A_tr_csc[:, next_index]
        next_index = random.choice(np.nonzero(row)[0])
        path.append('app_%d'%next_index)
        return next_index

    def B_api(self, next_index, path):
        row = self.B_tr[next_index]
        next_index = self.chose_idx(row)
        path.append('api_%d'%next_index)
        return next_index

    def P_api(self, next_index, path):
        row = self.P_tr[next_index]
        next_index = self.chose_idx(row)
        path.append('api_%d'%next_index)
        return next_index

    def metapath2vec(self, metapaths, appNumber):
        try:
            path = []
            # We have to start with an app
            path.append('app_%s'%appNumber)
            next_index = -1
            for i in metapaths:
                if(i == 'A'):
                    prefix = (path[-1].split('_')[0])
                    if(prefix == 'app'):
                        next_index = self.A_api(appNumber, path)
                    else:
                        next_index = self.A_app(next_index, path)
                if(i == 'B'):
                    next_index = self.B_api(next_index, path)
                if(i == 'P'):
                    next_index = self.P_api(next_index, path)   
#             logger.info(path)
#             logger.info(isinstance(path, collections.Iterable))
            return path
        except:
#             logger.info('rerun')
            return self.metapath2vec(metapaths, appNumber)
        
    def create_corpus(self, meta, save_fp, metapath2vec_epoch, suffix=''):
#         fp = '/datasets/dsc180a-wi20-public/Malware/group_data/group_01/metapath_corpus'
        corpus_name = 'meta_%s%s.cor'%(meta, suffix)
        metapaths = list('%s'%meta)

        f = open(os.path.join(save_fp, corpus_name), "w")

        for appNum in range(self.A_tr_csr.shape[0]):
        #     logger.info(appNum)
            for times in range(metapath2vec_epoch):
#                 logger.info(times)
                path = []
                f.write(' '.join(self.metapath2vec(metapaths, appNum)) + '\n')    
        f.close()
        
def metapath2vec_main(**cfg):
    path = os.path.join(cfg['data_dir'], cfg['data_subdirs']['processed'])
    outdir = os.path.join(path, 'walks')
    
#     path = '/datasets/dsc180a-wi20-public/Malware/group_data/group_01/pipeline_output_new'
    A = sparse.load_npz(os.path.join(path, 'A_reduced_tr.npz'))
    B_tr = sparse.load_npz(os.path.join(path, 'B_reduced_tr.npz')).tocsr()
    P_tr = sparse.load_npz(os.path.join(path, 'P_reduced_tr.npz')).tocsr()
    model = Metapath2Vec(A, B_tr, P_tr)

    save_fp = path #'/datasets/dsc180a-wi20-public/Malware/group_data/group_01/metapath_corpus_new'
    metas = ['ABA']
    for meta in metas:
        model.create_corpus(meta, outdir, cfg['metapath2vec_epoch'], '_reduced')
        
    A_tst = sparse.load_npz(os.path.join(path, 'A_reduced_tst.npz'))
    model = Metapath2Vec(A_tst, B_tr, P_tr)
    for meta in metas:
        model.create_corpus(meta, outdir, cfg['metapath2vec_epoch'], '_reduced_tst')

    print('prediction')
    mp.prediction('ABA', outdir, path, A.shape[0], True ,cfg['hindroid_reduced'])

   