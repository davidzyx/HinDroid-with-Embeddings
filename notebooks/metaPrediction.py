from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
    
A_shape = 1335
    
def check_tst_file(CORPUS_TEST):
    # check if the app number in the test corpus starts from 1335
    f = open(CORPUS_TEST).readlines()
    app_num = int(f[0].split()[0].split('_')[1])
    if(app_num < A_shape):
        print('changing')
        walks = []
        for line in f:
            walk = line.strip().split(' ')
            walks.append([
                f"app_{int(node.split('_')[-1]) + 1335}"
                if node.startswith('app') else node
                for node in walk
            ])


        f = open(CORPUS_TEST, "w")
        for walk in walks:
            f.write(' '.join(walk) + '\n')
        f.close()
    else:
        print('changed')
        return
    
def prediction(metapath, reduced=False):
    fp = '/datasets/dsc180a-wi20-public/Malware/group_data/group_01/metapath_corpus'
    if(not reduced):
        CORPUS = os.path.join(fp, 'meta_%s.cor'%metapath)
        CORPUS_TEST = os.path.join(fp, 'meta_%s_tst.cor'%metapath)
    else:
        CORPUS = os.path.join(fp, 'meta_%s_reduced.cor'%metapath)
        CORPUS_TEST = os.path.join(fp, 'meta_%s_reduced_tst.cor'%metapath)
    print(CORPUS, CORPUS_TEST)
    check_tst_file(CORPUS_TEST)
    
    from gensim import utils
    import gensim.models

    class MyCorpus(object):
        """An interator that yields sentences (lists of str)."""
        def __init__(self, CORPUS, CORPUS_TEST):
            self.lines = open(CORPUS).readlines()
    #         print(len(self.lines))
            self.lines += open(CORPUS_TEST).readlines()  # !!! Test
    #         print(len(self.lines))

        def __iter__(self):
            corpus_path = CORPUS
            for line in tqdm(self.lines):
                # assume there's one document per line, tokens separated by whitespace
                yield line.strip().split(' ')
    print('Creating model...')
    sentences = MyCorpus(CORPUS, CORPUS_TEST)
    WINDOW = len(metapath) // 2
    model = gensim.models.Word2Vec(sentences=sentences, min_count=10, size=500, window=WINDOW)
    
    path = '/datasets/dsc180a-wi20-public/Malware/group_data/group_01/pipeline_output'
    meta_tr = pd.read_csv(os.path.join(path, 'meta_tr.csv'), index_col=0)
    meta_tst = pd.read_csv(os.path.join(path, 'meta_tst.csv'), index_col=0)

    y_train = meta_tr.label == 'class1'
    y_val = y_train[1100:]
    y_train = y_train[:1100]
    
    y_test = meta_tst.label == 'class1'

    app_vec = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr))])
    app_val = app_vec[1100:]
    app_vec = app_vec[:1100]
    app_vec_tst = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr), len(meta_tr) + len(meta_tst))])
    
    # select best parameters
    print('Selecting best parameters...')
    param_grid = {'C': [1,10,100,1000,10000], 'kernel': ('linear', 'poly'), 'degree': np.arange(20)+1}
    svc = SVC(gamma='auto')
    clf = GridSearchCV(svc, param_grid, cv=5, return_train_score=True, iid=False, n_jobs=-1)
    best = clf.fit(app_val, y_val).best_params_
#     best = clf.fit(app_vec, y_train).best_params_
    print(best)
    
    print('Training...')
    svm = SVC(kernel='poly', C=best['C'], degree=best['degree'], gamma='auto')
#     svm = SVC(kernel='linear')
    svm.fit(app_vec, y_train)
#     svm.fit(app_val, y_val)
    
    y_pred = svm.predict(app_vec_tst)
    print('train_acc: ', svm.score(app_vec, y_train))
    print('test_acc: ', svm.score(app_vec_tst, y_test))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print('tn', 'fp', 'fn', 'tp')
    print(tn, fp, fn, tp)
    print('')
    return model, y_train, y_test, app_vec, app_vec_tst, y_pred, svm
    