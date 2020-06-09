from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

from sklearn.manifold import TSNE                   # final reduction
from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
from plotly import tools
import plotly.express as px

    
    
def check_tst_file(CORPUS_TEST, A_shape):
    # check if the app number in the test corpus starts from 1335
    f = open(CORPUS_TEST).readlines()
    app_num = int(f[0].split()[0].split('_')[1])
    if(app_num < A_shape):
        print('changing')
        walks = []
        for line in f:
            walk = line.strip().split(' ')
            walks.append([
                f"app_{int(node.split('_')[-1]) + A_shape}"
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
    
def prediction(metapath, meta_fp, labels_fp, shape, test, reduced=False):
    if(not reduced):
        CORPUS = os.path.join(meta_fp, 'meta_%s.cor'%metapath)
        CORPUS_TEST = os.path.join(meta_fp, 'meta_%s_tst.cor'%metapath)
        graph_title = metapath + " two dimensional embeddings"
    else:
        CORPUS = os.path.join(meta_fp, 'meta_%s_reduced.cor'%metapath)
        CORPUS_TEST = os.path.join(meta_fp, 'meta_%s_reduced_tst.cor'%metapath)
        graph_title = metapath + "reduced two dimensional embeddings"

    print(CORPUS, CORPUS_TEST)
    check_tst_file(CORPUS_TEST, shape)
    
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
    
#     path = '/datasets/dsc180a-wi20-public/Malware/group_data/group_01/pipeline_output'
    meta_tr = pd.read_csv(os.path.join(labels_fp, 'meta_tr.csv'), index_col=0)
    meta_tst = pd.read_csv(os.path.join(labels_fp, 'meta_tst.csv'), index_col=0)
    
    
    y_train = meta_tr.label == 'class1'
    spliter = (len(y_train)//2)
    y_val = y_train[spliter:]
    y_train = y_train[:spliter]
    
    y_test = meta_tst.label == 'class1'

    app_vec = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr))])
    app_val = app_vec[spliter:]
    app_vec = app_vec[:spliter]
    app_vec_tst = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr), len(meta_tr) + len(meta_tst))])
    
    # select best parameters
    if(test):
        y_train = meta_tr.label == 'class1'
        y_test = meta_tst.label == 'class1'
        app_vec = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr))])
        app_vec_tst = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr), len(meta_tr) + len(meta_tst))])
        svm = LogisticRegression(random_state=0)
        svm.fit(app_vec, y_train)
        
    else:
        print('Selecting best parameters...')
        param_grid = {'C': [1,10,100,1000,10000], 'kernel': ('linear', 'poly'), 'degree': np.arange(20)+1}
        svc = SVC(gamma='auto')
        clf = GridSearchCV(svc, param_grid, cv=spliter+1, return_train_score=True, iid=False, n_jobs=-1)
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
    
    # Graph predictions
    x_vals, y_vals, labels = reduce_dimensions(model)    
    df_dict = {'x_vals': x_vals, 'y_vals': y_vals, 'labels': labels}
    df = pd.DataFrame(df_dict)
    graph_labels = {0: 'train_benign', 1: 'train_malware', 2: 'test_benign', 3: 'test_malware'}
    df = df.replace({"labels": graph_labels})
    plot_with_plotly(df, graph_title)
    
    print('')
    
    
    return model, y_train, y_test, app_vec, app_vec_tst, y_pred, svm

def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    for word in model.wv.vocab:
        if 'app' in word:
#             labels.append(label_classifier(int(word.split('_')[1])))

            vectors.append(model.wv[word])
            labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, [label_classifier(i) for i in labels]


def plot_with_plotly(df, graph_title='Graph'):
    
    fig = px.scatter(df, x="x_vals", y="y_vals", color='labels', title=graph_title)
    fig.show()
#     data = [trace]

#     if plot_in_notebook:
#         init_notebook_mode(connected=True)
#         iplot(data, filename='word-embedding-plot')
#     else:
#         plot(data, filename='word-embedding-plot.html')

def label_classifier(app_number):
    labels_fp = '/datasets/dsc180a-wi20-public/Malware/group_data/group_01/pipeline_output_new'
    meta_tr = pd.read_csv(os.path.join(labels_fp, 'meta_tr.csv'), index_col=0)
    meta_tst = pd.read_csv(os.path.join(labels_fp, 'meta_tst.csv'), index_col=0)
    
    number = int(app_number.split('_')[1])
    if number > 1335 - 1:
        if meta_tst.loc[app_number]['label'] == 'class1':
            return 3  #test_malware
        else:
            return 2  #test_benign
    else:
        if meta_tr.loc[app_number]['label'] == 'class1':
            return 1 #train_malware
        else:
            return 0 #train_benign
    