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

    def metapath2vec(self, metapaths, path, appNumber):
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
        return path