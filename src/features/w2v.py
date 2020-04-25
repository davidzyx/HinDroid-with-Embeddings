from gensim import utils
import gensim.models
import numpy as np
from scipy import sparse

from sklearn.manifold import TSNE                   # final reduction

class word2vec(object):
    def __init__(self, matrix_path, corpus_path):
        self.matrix_path = matrix_path
        self.corpus_path = corpus_path
                
    def generate_matrix(self):
        
        train_A = sparse.load_npz(self.matrix_path + 'train_A.npz').tocsr()
        train_B = sparse.load_npz(self.matrix_path + 'train_B.npz').tocsc()
        train_P = sparse.load_npz(self.matrix_path + 'train_P.npz').tocsc()
        test_A = sparse.load_npz(self.matrix_path + 'test_A.npz').tocsr()
        train_A_csc = train_A.tocsc(copy=True)
        
        return train_A, train_B, train_P, test_A, train_A_csc 
        
    def create_model(self):
        sentences = MyCorpus(self.corpus_path)
        model = gensim.models.Word2Vec(sentences=sentences, min_count=1, size=200)
        return model
    
    def ABPBA(self):
        while True:
            app_i = np.random.choice(np.arange(A.shape[0]))
            api_i = np.random.choice(np.nonzero(A_csr[app_i])[1])
            api_bi = np.random.choice(np.nonzero(B[:, api_i])[0])
            api_p = np.random.choice(np.nonzero(P[:, api_bi])[0])
            api_bj = np.random.choice(np.nonzero(B[:, api_p])[0])
            app_j = np.random.choice(np.nonzero(A_csc[:, api_bj])[0])

            yield f'app_{app_i} api_{api_i} api_{api_bi} api_{api_p} api_{api_bj} app_{app_j}'


    def ABA(self):
        while True:
            app_i = np.random.choice(np.arange(A.shape[0]))
            api_i = np.random.choice(np.nonzero(A_csr[app_i])[1])
            api_bi = np.random.choice(np.nonzero(B[:, api_i])[0])
            app_j = np.random.choice(np.nonzero(A_csc[:, api_bi])[0])

            yield f'app_{app_i} api_{api_i} api_{api_bi} app_{app_j}'
    
    def plot(model):
        x_vals, y_vals, labels = reduce_dimensions(model)

        plot_with_plotly(x_vals, y_vals, labels)

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
    
    def __iter__(self):
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield line.strip().split(' ')


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    for word in model.wv.vocab:
        if 'app' in word:
            if int(word.split('_')[1]) > 332:
                labels.append(1)
            else:
                labels.append(0)

            vectors.append(model.wv[word])
            # labels.append(word)

    # convert both lists into numpy vectors for reduction
    vectors = np.asarray(vectors)
    labels = np.asarray(labels)

    # reduce using t-SNE
    vectors = np.asarray(vectors)
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    y_vals = [v[1] for v in vectors]
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go

    trace = go.Scatter(x=x_vals, y=y_vals, mode='markers',
                       text=labels, marker=dict(size=5, color=labels))

    data = [trace]

    if plot_in_notebook:
        init_notebook_mode(connected=True)
        iplot(data, filename='word-embedding-plot')
    else:
        plot(data, filename='word-embedding-plot.html')


def plot_with_matplotlib(x_vals, y_vals, labels):
    import matplotlib.pyplot as plt
    import random

    random.seed(0)

    plt.figure(figsize=(12, 12))
    plt.scatter(x_vals, y_vals)

    #
    # Label randomly subsampled 25 data points
    #
    indices = list(range(len(labels)))
    selected_indices = random.sample(indices, 25)
    for i in selected_indices:
        plt.annotate(labels[i], (x_vals[i], y_vals[i]))
    plt.show()