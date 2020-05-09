from gensim import utils
import gensim.models
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from sklearn.manifold import TSNE                   # final reduction

class word2vec(object):
    def __init__(self, matrix_path, corpus_path):
        self.matrix_path = matrix_path
        self.corpus_path = corpus_path
                
    def load_matrix(self):
        
        self.train_A = sparse.load_npz(self.matrix_path + '/train_A.npz').tocsr()
        self.train_B = sparse.load_npz(self.matrix_path + '/train_B.npz').tocsc()
        self.train_P = sparse.load_npz(self.matrix_path + '/train_P.npz').tocsc()
        self.test_A = sparse.load_npz(self.matrix_path + '/test_A.npz').tocsr()
        self.train_A_csc = self.train_A.tocsc(copy=True)        
    
    def generate_corpus(self, metapath, num_doc, walk_length):
        if metapath == "ABPBA":
            corpus_function = self.ABPBA(length=walk_length)
            
        if metapath == "ABA":
            corpus_function = self.ABA(length=walk_length)
                
        if metapath == "APA":
            corpus_function = self.APA(length=walk_length)
            
        f = open(self.corpus_path, 'w')
        for _ in tqdm(range(num_doc)):
            f.write(next(corpus_function) + '\n')
        f.close()
           
    def create_model(self):
        sentences = MyCorpus(self.corpus_path)
        self.model = gensim.models.Word2Vec(sentences=sentences)
    
    def ABPBA(self, length=5000):
        while True:

            app = np.random.choice(np.arange(self.train_A.shape[0]))

            path = f'app_{app}'

            for i in range(length):

                api_i = np.random.choice(np.nonzero(self.train_A[app])[1])
                api_bi = np.random.choice(np.nonzero(self.train_B[:, api_i])[0])
                api_p = np.random.choice(np.nonzero(self.train_P[:, api_bi])[0])
                api_bj = np.random.choice(np.nonzero(self.train_B[:, api_p])[0])
                app = np.random.choice(np.nonzero(self.train_A_csc[:, api_bj])[0])

                path += f' api_{api_i} api_{api_bi} api_{api_p} api_{api_bj} app_{app}'

            yield path
            
    def ABA(self, length=5000):
        while True:

            app = np.random.choice(np.arange(self.train_A.shape[0]))

            path = f'app_{app}'

            for i in range(length):

                api_i = np.random.choice(np.nonzero(self.train_A[app])[1])
                api_b = np.random.choice(np.nonzero(self.train_B[:, api_i])[0])
                app = np.random.choice(np.nonzero(self.train_A_csc[:, api_b])[0])

                path += f' api_{api_i} api_{api_b} app_{app}'

            yield path

    def APA(self, length=5000):
        while True:
            app = np.random.choice(np.arange(self.train_A.shape[0]))

            path = f'app_{app}'

            for i in range(length):

                api_i = np.random.choice(np.nonzero(self.train_A[app])[1])
                api_p = np.random.choice(np.nonzero(self.train_P[:, api_i])[0])
                app = np.random.choice(np.nonzero(self.train_A_csc[:, api_p])[0])

                path += f' api_{api_i} api_{api_p} app_{app}'

            yield path
    
    def plot_embeddings(self):
        x_vals, y_vals, labels = reduce_dimensions(self.model)

        plot_with_plotly(x_vals, y_vals, labels)
        
    # Predict embeddings for application in testing set
    # Populate the embeddings into the model
    def predict_embeddings(self):
        X = []
        Y = []
        train_labels = []
        for j in range(self.train_A.shape[0]):
            indexes = np.nonzero((self.train_A[j]).toarray()[0])[0]
            all_api = self.model.wv.vocab.keys()
            matrix = np.zeros(100)
            for i in indexes:
                element = 'api_' + str(i)
                if element in all_api:
                    matrix += self.model.wv[element]
            matrix /= len(all_api)
            X.append(matrix)
            Y.append(self.model.wv['app_' + str(j)])
            train_labels.append('app_' + str(j))
                                   
        test_X = []
        test_labels = []
        for j in range(self.test_A.shape[0]):
            indexes = np.nonzero((self.test_A[j]).toarray()[0])[0]
            all_api = self.model.wv.vocab.keys()
            matrix = np.zeros(100)
            for i in indexes:
                element = 'api_' + str(i)
                if element in all_api:
                    matrix += self.model.wv[element]
            matrix /= len(all_api)
            test_X.append(matrix)
            test_labels.append('app_' + str(j + self.train_A.shape[0]))

        regressor = LinearRegression()
        regressor.fit(X, Y)
        
        embeddings = regressor.predict(test_X)
        
        for i in range(len(test_labels)):
            self.model.wv[test_labels[i]] = embeddings[i] 
        
        self.train_embeddings = Y
        self.test_embeddings = embeddings
        self.train_labels = [app_label(i) for i in train_labels]
        self.test_labels = [app_label(i) for i in test_labels]
        
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
    
    def __iter__(self):
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield line.strip().split(' ')

# Helper function in reduce dimensions for colors on the graph
def label_classifier(app_number):
    app_number = int(app_number.split('_')[1])
    if app_number > 332 and app_number < 666:
#         return "train_malware"
        return 0
    
    elif app_number <= 332:
#         return "train_benign"
        return 1
    
    elif app_number >= 666 and app_number < 833:
#         return "test_benign"
        return 2
    
    else:
        return 3
#         return "test_malware"

# benign - 0, malware - 1
def app_label(app_number):
    app_number = int(app_number.split('_')[1])
    if app_number > 332 and app_number < 666:
        return 1
    
    elif app_number <= 332:
        return 0
    
    elif app_number >= 666 and app_number < 833:
        return 0
    
    else:
        return 1
            
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
    return x_vals, y_vals, labels


def plot_with_plotly(x_vals, y_vals, labels, plot_in_notebook=True):
    from plotly.offline import init_notebook_mode, iplot, plot
    import plotly.graph_objs as go
    from plotly import tools

    trace = go.Scatter(x=x_vals, y=y_vals, mode='markers', text=labels, 
                       marker=dict(size=5, color=[label_classifier(i) for i in labels]))
    
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
    
def cosine(u, v):
    return np.dot(u, v)/(np.linalg.norm(u)* np.linalg.norm(v))