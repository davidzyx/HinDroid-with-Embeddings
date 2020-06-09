import pandas as pd
import numpy as np
from scipy import sparse
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm
import time

import src.utils as utils


class HinDroid():
    def __init__(self, B_mat, P_mat, metapaths):
        self.B_mat = B_mat
        self.P_mat = P_mat
        self.metapaths = metapaths
        self.kernels = self.construct_kernels(metapaths)
        self.svms = [SVC(kernel='precomputed') for mp in metapaths]

    def _kernel_func(self, metapath):
        B_mat = self.B_mat
        P_mat = self.P_mat
        if metapath == 'AA':
            f = lambda X, Y: np.dot(X, Y.T)
        elif metapath == 'ABA':
            f = lambda X, Y: np.dot(X, B_mat).dot(Y.T)
        elif metapath == 'APA':
            f = lambda X, Y: np.dot(X, P_mat).dot(Y.T)
        elif metapath == 'APBPA':
            f = lambda X, Y: np.dot(X, P_mat).dot(B_mat).dot(P_mat).dot(Y.T)
        else:
            raise NotImplementedError

        return lambda X, Y: f(X, Y).todense()

    def construct_kernels(self, metapaths):
        kernels = []
        for mp in metapaths:
            kernels.append(self._kernel_func(mp))
        return kernels

    def _evaluate(self, X_train, X_test, y_train, y_test):
        results = []
        for mp, kernel, svm in zip(self.metapaths, self.kernels, self.svms):
            print(f'Evaluating {mp}...', end='', file=sys.stderr, flush=True)
            gram_train = kernel(X_train, X_train)
            svm.fit(gram_train, y_train)
            train_acc = svm.score(gram_train, y_train)

            gram_test = kernel(X_test, X_train)
            y_pred = svm.predict(gram_test)
            test_acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            results.append(pd.Series({
                'train_acc': train_acc, 'test_acc': test_acc, 'f1': f1,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
            }))
            print('done', file=sys.stderr)

        return results

    def evaluate(self, X, y, test_size=0.33):
        X = sparse.csr_matrix(X, dtype='uint32')
        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_size)

        results = self._evaluate(X_train, X_test, y_train, y_test)
        results = [res.rename(mp) for res, mp in zip(results, self.metapaths)]
        results = pd.DataFrame(results)
        results.index.name = 'metapath'
        return results


class HinDroidNew():
    def __init__(self, matrices, metapaths):
        self.A_tr = matrices['A_tr']
        self.A_tst = matrices['A_tst']
        self.B_tr = matrices['B_tr']
        # self.B_tst = matrices['B_tst']
        self.P_tr = matrices['P_tr']
        # self.P_tst = matrices['P_tst']
        self.metapaths = metapaths
        # self.kernels = self.construct_kernels(metapaths)
        # self.svms = [SVC(kernel='precomputed') for mp in metapaths]
    
    def evaluate(self, y_train, y_test):
        self.A_tr = sparse.csr_matrix(self.A_tr, dtype='uint32')
        self.A_tst = sparse.csr_matrix(self.A_tst, dtype='uint32')

        results = []
        tr_pred = {}
        tst_pred = {}
        for path in self.metapaths:
            print(path)
            now = time.time()

            print('Calculating gram matrix for train')
            if path is 'AA':
                gram_train = np.dot(self.A_tr, self.A_tr.T).todense()
            elif path is 'APA':
                gram_train = np.dot(np.dot(self.A_tr, self.P_tr), self.A_tr.T).todense()
            elif path is 'ABA':
                gram_train = np.dot(np.dot(self.A_tr, self.B_tr), self.A_tr.T).todense()
            elif path is 'APBPA':
                gram_train = (self.A_tr * self.P_tr * self.B_tr * self.P_tr * self.A_tr.T).todense()
            elif path is 'ABPBA':
                gram_train = (self.A_tr * self.B_tr * self.P_tr * self.B_tr * self.A_tr.T).todense()
            else:
                raise NotImplementedError()

            print('Fitting SVM')
            svm = SVC(kernel='precomputed')
            svm.fit(gram_train, y_train)

            print(time.time() - now)

            train_predicted = svm.predict(gram_train)
            train_acc = accuracy_score(y_train, train_predicted)
            tr_pred[path] = train_predicted

            del gram_train

            print('Calculating gram matrix for test')
            if path is 'AA':
                gram_test = np.dot(self.A_tst, self.A_tr.T).todense()
            elif path is 'APA':
                gram_test = np.dot(np.dot(self.A_tst, self.P_tr), self.A_tr.T).todense()
            elif path is 'ABA':
                gram_test = np.dot(np.dot(self.A_tst, self.B_tr), self.A_tr.T).todense()
            elif path is 'APBPA':
                gram_test = (self.A_tst * self.P_tr * self.B_tr * self.P_tr * self.A_tr.T).todense()
            elif path is 'ABPBA':
                gram_test = (self.A_tst * self.B_tr * self.P_tr * self.B_tr * self.A_tr.T).todense()
            else:
                raise NotImplementedError()

            print('Predicting SVM')
            y_pred = svm.predict(gram_test)
            del gram_test
            tst_pred[path] = y_pred

            test_acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()


            result = pd.Series({
                'train_acc': train_acc, 'test_acc': test_acc, 'f1': f1,
                'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
            })
            print(result)
            print()
            results.append(result)

        return results, tr_pred, tst_pred




def run(**config):
    PROC_DIR = utils.PROC_DIR

    if config['hindroid_reduced']:
        A_tr, A_tst, B_tr, P_tr = [
            sparse.load_npz(os.path.join(PROC_DIR, mat))
            for mat in [
                'A_reduced_tr.npz', 'A_reduced_tst.npz',
                'B_reduced_tr.npz', 'P_reduced_tr.npz'
            ]
        ]
    else:
        A_tr, A_tst, B_tr, P_tr = [
            sparse.load_npz(os.path.join(PROC_DIR, mat))
            for mat in ['A_tr.npz', 'A_tst.npz', 'B_tr.npz', 'P_tr.npz']
        ]

    meta_tr_fp = os.path.join(PROC_DIR, 'meta_tr.csv')
    meta_tst_fp = os.path.join(PROC_DIR, 'meta_tst.csv')
    meta_tr = pd.read_csv(meta_tr_fp, index_col=0)
    meta_tst = pd.read_csv(meta_tst_fp, index_col=0)
    tr_labels = (meta_tr.label == 'class1').astype(int).values
    tst_labels = (meta_tst.label == 'class1').astype(int).values

    metapaths = ['AA', 'APA', 'ABA', 'APBPA', 'ABPBA']
    matrices = {
        'A_tr': A_tr, 'A_tst': A_tst,
        'B_tr': B_tr, 'P_tr': P_tr
    }
    hin = HinDroidNew(matrices, metapaths)
    results, tr_pred, tst_pred = hin.evaluate(tr_labels, tst_labels)

    for mtpath, preds in tr_pred.items():
        meta_tr[mtpath] = preds
    
    for mtpath, preds in tst_pred.items():
        meta_tst[mtpath] = preds

    meta_tr.to_csv(meta_tr_fp)
    meta_tst.to_csv(meta_tst_fp)
