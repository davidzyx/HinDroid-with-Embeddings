from gensim import utils
import gensim.models
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesRegressor,
                              AdaBoostRegressor)
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import confusion_matrix

from sklearn.manifold import TSNE                   # final reduction

from plotly.offline import init_notebook_mode, iplot, plot
import plotly.graph_objs as go
from plotly import tools
import plotly.express as px

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt
import os

import src.utils as utils

class word2vec(object):
    def __init__(self, matrix_path, corpus_path):
        self.matrix_path = matrix_path
        self.corpus_path = corpus_path
                
    def load_matrix(self):
        
#         self.train_A = sparse.load_npz(self.matrix_path + '/train_A.npz').tocsr()
#         self.train_B = sparse.load_npz(self.matrix_path + '/train_B.npz').tocsc()
#         self.train_P = sparse.load_npz(self.matrix_path + '/train_P.npz').tocsc()
#         self.test_A = sparse.load_npz(self.matrix_path + '/test_A.npz').tocsr()
        self.train_A = sparse.load_npz(self.matrix_path + '/A_reduced_tr.npz').tocsr()
        self.train_B = sparse.load_npz(self.matrix_path + '/B_reduced_tr.npz').tocsc()
        self.train_P = sparse.load_npz(self.matrix_path + '/P_reduced_tr.npz').tocsc()
        self.test_A = sparse.load_npz(self.matrix_path + '/A_reduced_tst.npz').tocsr()
        self.train_A_csc = self.train_A.tocsc(copy=True) 
        self.num_train = self.train_A.shape[0]
        
        self.train_label = pd.read_csv(self.matrix_path + '/meta_tr.csv', index_col=None).rename(columns={'Unnamed: 0':'app_id'}).set_index('app_id')['label'].to_dict()
        self.test_label = pd.read_csv(self.matrix_path + '/meta_tst.csv', index_col=None).rename(columns={'Unnamed: 0':'app_id'}).set_index('app_id')['label'].to_dict()
    
    def generate_corpus(self, metapath, num_doc, walk_length):
        if metapath == "ABPBA":
            corpus_function = self.ABPBA(length=walk_length)
            
        if metapath == "ABA":
            corpus_function = self.ABA(length=walk_length)
                
        if metapath == "APA":
            corpus_function = self.APA(length=walk_length)
            
        if metapath == "APBPA":
            corpus_function = self.APBPA(length=walk_length)
            
        f = open(self.corpus_path, 'w')
        for _ in tqdm(range(num_doc)):
            f.write(next(corpus_function) + '\n')
        f.close()
           
    def create_model(self):
        sentences = MyCorpus(self.corpus_path)
        self.model = gensim.models.Word2Vec(sentences=sentences, size=256, sg=1, negative=5, window=7, iter=3)
#         self.model = gensim.models.Word2Vec(sentences=sentences, size=100)
    
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
    
    def APBPA(self, length=5000):
        while True:

            app = np.random.choice(np.arange(self.train_A.shape[0]))

            path = f'app_{app}'

            for i in range(length):

                api_i = np.random.choice(np.nonzero(self.train_A[app])[1])
                api_pi = np.random.choice(np.nonzero(self.train_P[:, api_i])[0])
                api_b = np.random.choice(np.nonzero(self.train_B[:, api_pi])[0])
                api_pj = np.random.choice(np.nonzero(self.train_P[:, api_b])[0])
                app = np.random.choice(np.nonzero(self.train_A_csc[:, api_pj])[0])

                path += f' api_{api_i} api_{api_pi} api_{api_b} api_{api_pj} app_{app}'

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
        x_vals, y_vals, labels = reduce_dimensions(self)
        
        df_dict = {'x_vals': x_vals, 'y_vals': y_vals, 'labels': labels}
        df = pd.DataFrame(df_dict)
        graph_labels = {0: 'train_benign', 1: 'train_malware', 2: 'test_benign', 3: 'test_malware'}
        df = df.replace({"labels": graph_labels})
        graph_title = self.corpus_path.split('/')[-1].split('_')[0] + " two dimensional embeddings"
        plot_with_plotly(df, graph_title)
        
    # Predict embeddings for application in testing set
    # Populate the embeddings into the model
    def predict_embeddings(self):
        X = []
        Y = []
        train_labels = []
        missed_app = []
        for j in range(self.train_A.shape[0]):
            if ('app_' + str(j)) not in self.model.wv:
                missed_app.append(j)
            else:
                indexes = np.nonzero((self.train_A[j]).toarray()[0])[0]
                all_api = self.model.wv.vocab.keys()
                matrix = np.zeros(256)
                for i in indexes:
                    element = 'api_' + str(i)
                    if element in all_api:
                        matrix += self.model.wv[element]
                matrix /= len(all_api)
                X.append(matrix)
                Y.append(self.model.wv['app_' + str(j)])
                train_labels.append('app_' + str(j))
                
        regressor = DecisionTreeRegressor(max_depth=None).fit(X, Y)
            
        # Keep track of which app didnt get selected into corpus
        temp_X = []
        temp_labels = []
        for j in missed_app:
            indexes = np.nonzero((self.train_A[j]).toarray()[0])[0]
            all_api = self.model.wv.vocab.keys()
            matrix = np.zeros(256)
            for i in indexes:
                element = 'api_' + str(i)
                if element in all_api:
                    matrix += self.model.wv[element]
            matrix /= len(all_api)
            temp = regressor.predict([matrix])
            self.model.wv['app_' + str(j)] = temp
                                   
        test_X = []
        test_labels = []
        for j in range(self.test_A.shape[0]):
            indexes = np.nonzero((self.test_A[j]).toarray()[0])[0]
            all_api = self.model.wv.vocab.keys()
            matrix = np.zeros(256)
            for i in indexes:
                element = 'api_' + str(i)
                if element in all_api:
                    matrix += self.model.wv[element]
            matrix /= len(all_api)
            test_X.append(matrix)
            test_labels.append('app_' + str(j + self.train_A.shape[0]))
    
        embeddings = regressor.predict(test_X)
        
        for i in range(len(test_labels)):
            self.model.wv[test_labels[i]] = embeddings[i] 
        
        
        self.train_embeddings = Y
        self.test_embeddings = embeddings
        self.train_labels = [self.read_label(i) for i in train_labels]
        self.test_labels = [self.read_label(i) for i in test_labels]
        
        
    def read_label(self, app_number):
        number = int(app_number.split('_')[1])
        
        if number > self.num_train - 1:
            if self.test_label[app_number] == 'class1':
                return 1
            else:
                return 0
            
        else:
            if self.train_label[app_number] == 'class1':
                return 1
            else:
                return 0
            
    def train_nn(self, num_epoch=10000):
        train_X = torch.tensor(self.train_embeddings).float()
        test_X = torch.tensor(self.test_embeddings).float()
        
        train_Y = torch.tensor(self.train_labels).float()
        test_Y = torch.tensor(self.test_labels).float()
        
        net = Net(train_X.shape[1])
        criterion = nn.BCELoss()
        # criterion = torch.nn.MSELoss(reduction='mean')
        # optimizer = torch.optim.Adamax(net.parameters(), lr=0.0001)
        optimizer = optim.Adam(net.parameters(), lr=0.0001)
        
        y_pred = None
        y_test_pred = None

        for epoch in range(num_epoch):  # loop over the dataset multiple times

            running_loss = 0.0

            y_pred = net(train_X)
            y_pred = torch.squeeze(y_pred)

            train_loss = criterion(y_pred, train_Y)

            if epoch % 1000 == 0:
                train_acc = calculate_accuracy(train_Y, y_pred)

                y_test_pred = net(test_X)
                y_test_pred = torch.squeeze(y_test_pred)

                test_loss = criterion(y_test_pred, test_Y)

                test_acc = calculate_accuracy(test_Y, y_test_pred)
                print(
                    f'''epoch {epoch}
                    Train set - loss: {round_tensor(train_loss)}, accuracy: {round_tensor(train_acc)}
                    Test  set - loss: {round_tensor(test_loss)}, accuracy: {round_tensor(test_acc)}
                ''')

            optimizer.zero_grad()

            train_loss.backward()

            optimizer.step()

        print('Finished Training')
        
        self.nn_train_pred = y_pred
        self.nn_test_pred = y_test_pred
        
    def evaluate(self):
        cm = confusion_matrix(torch.tensor(self.test_labels).float().numpy()*1, self.nn_test_pred.ge(.5).view(-1).detach().numpy()*1)
        df_cm = pd.DataFrame(cm, index=['benign', 'malware'], columns=['benign', 'malware'])

        hmap = sns.heatmap(df_cm, annot=True, fmt="d")
        hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
        hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        graph_title = self.corpus_path.split('/')[-1].split('_')[0] + " confusion matrix"
        plt.title(graph_title, fontsize=20)
        
    def save_result(self):
        train_df = pd.DataFrame.from_dict(self.train_label, orient='index',columns=['labels'])
        test_df = pd.DataFrame.from_dict(self.test_label, orient='index',columns=['labels'])

        train_df[self.corpus_path.split('/')[-1].split('.cor')[0]] = self.nn_train_pred.ge(.5).view(-1).detach().numpy()*1
        test_df[self.corpus_path.split('/')[-1].split('.cor')[0]] = self.nn_test_pred.ge(.5).view(-1).detach().numpy()*1

        train_directory = './data/processed/meta_tr_w2v_' + self.corpus_path.split('/')[-1].split('.cor')[0] + '.csv'
        test_directory = './data/processed/meta_tst_w2v_' + self.corpus_path.split('/')[-1].split('.cor')[0] + '.csv'
        
        train_df.to_csv(train_directory, index=True)
        print("Saved: " + train_directory)
        
        test_df.to_csv(test_directory, index=True)
        print("Saved: " + test_directory)
            
        
class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
    
    def __iter__(self):
        for line in open(self.corpus_path):
            # assume there's one document per line, tokens separated by whitespace
            yield line.strip().split(' ')
            
class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_features, 5, bias=False)
        self.fc2 = nn.Linear(5, 3, bias=False)
        self.fc3 = nn.Linear(3, 1, bias=False)
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))

# class Net(nn.Module):
#     def __init__(self, n_features):
#         super(Net, self).__init__()
#         self.classifier = nn.Sequential(
#             nn.Linear(n_features, 512),
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout(0.2),
#             nn.Linear(512, 512),
#             nn.LeakyReLU(inplace=True),
#             nn.Linear(512,1)
#         )
#     def forward(self, x):
#         return torch.sigmoid(self.classifier(x))

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)

# Helper function in reduce dimensions for colors on the graph
def label_classifier(model, app_number):
    number = int(app_number.split('_')[1])
    if number > model.num_train - 1:
        if model.test_label[app_number] == 'class1':
            return 3  #test_malware
        else:
            return 2  #test_benign
            
    else:
        if model.train_label[app_number] == 'class1':
            return 1 #train_malware
        else:
            return 0 #train_benign
                        
def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    vectors = []  # positions in vector space
    labels = []  # keep track of words to label our data again later
    for word in model.model.wv.vocab:
        if 'app' in word:
#             labels.append(label_classifier(int(word.split('_')[1])))

            vectors.append(model.model.wv[word])
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
    return x_vals, y_vals, [label_classifier(model, i) for i in labels]


def plot_with_plotly(df, graph_title):
    
    fig = px.scatter(df, x="x_vals", y="y_vals", color="labels", title=graph_title)
    fig.show()
#     data = [trace]

#     if plot_in_notebook:
#         init_notebook_mode(connected=True)
#         iplot(data, filename='word-embedding-plot')
#     else:
#         plot(data, filename='word-embedding-plot.html')
    
def cosine(u, v):
    return np.dot(u, v)/(np.linalg.norm(u)* np.linalg.norm(v))


def word2vec_main(**cfg):
    matrix_path = utils.PROC_DIR
    corpus_path = os.path.join(utils.PROC_DIR, 'walks', 'word2vec_ABPBA_reduced.cor')

    model = word2vec(matrix_path, corpus_path)
    model.load_matrix()
    model.generate_corpus("ABPBA", 100, 50)
    model.create_model()
    model.predict_embeddings()
    model.plot_embeddings()
    model.train_nn(num_epoch=cfg['word2vec_epoch'])
    model.evaluate()
    model.save_result()