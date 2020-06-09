from tqdm import tqdm
import numpy as np
from scipy import sparse
import os
import gensim.models
import pandas as pd

import src.utils as utils

from sklearn.ensemble import RandomForestRegressor

from src.features.w2v import reduce_dimensions, plot_with_plotly

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

class Node2Vec():
    def __init__(self, indir, n=1, p=2, q=1, walk_length=100, test=False, test_offset=0):
        self.indir = indir
        self.offset = test_offset

        outdir = os.path.join(indir, 'walks')
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        fp = os.path.join(
            outdir, f'node2vec_n={n}_p={p}_q={q}_wl={walk_length}.cor'
        )
        if test:
            fp = os.path.join(
                outdir, f'node2vec_n={n}_p={p}_q={q}_wl={walk_length}_test.cor'
            )
        self.corpus_path = fp

        self.n = n
        self.p = p 
        self.q = q 
        self.walk_length = walk_length
    
    def get_api_neighbors_A(self, app):
        """Get all API neighbors of an APP from A matrix"""
        assert app.startswith('app_')
        app_id = int(app.split('_')[1])
        neighbor_ids = np.nonzero(self.A_tr_csr[app_id])[1]
        return np.array([f'api_{s}' for s in neighbor_ids])

    def get_app_neighbors_A(self, api):
        """Get all APP neighbors of an API from A matrix"""
        assert api.startswith('api_')
        api_id = int(api.split('_')[1])
        neighbor_ids = np.nonzero(self.A_tr_csc[:, api_id])[0]
        return np.array([f'app_{s}' for s in neighbor_ids])

    def get_api_neighbors_B(self, api):
        """Get all API neighbors of an API from B matrix"""
        assert api.startswith('api_')
        api_id = int(api.split('_')[1])
        neighbor_ids = np.nonzero(self.B_tr[:, api_id])[0]
        ls = [f'api_{s}' for s in neighbor_ids]
        ls.remove(api)
        return np.array(ls)

    def get_api_neighbors_P(self, api):
        """Get all API neighbors of an API from P matrix"""
        assert api.startswith('api_')
        api_id = int(api.split('_')[1])
        neighbor_ids = np.nonzero(self.P_tr[:, api_id])[0]
        ls = [f'api_{s}' for s in neighbor_ids]
        ls.remove(api)
        return np.array(ls)

    def all_neighbors_from_api(self, api):
        """Get all API neighbors of an APP from all matrices (B and P)"""
        assert api.startswith('api_')
        api_id = int(api.split('_')[1])
        nbr_apis = np.concatenate([
            self.get_api_neighbors_B(api),
            self.get_api_neighbors_P(api)
        ])
        nbr_apis = np.unique(nbr_apis)
        nbr_apps = self.get_app_neighbors_A(api)
        # weights later? no
        return nbr_apis, nbr_apps
    
    def perform_one_walk_full(self, p=1, q=1, walk_length=20, app=None):
        path = []

        if app is None:
            app = 'app_' + str(np.random.choice(np.arange(self.A_tr_csr.shape[0])))
        prev_nbrs = self.get_api_neighbors_A(app)
        curr_node = np.random.choice(prev_nbrs)
        prev_node = app
        path.append(app)
        path.append(curr_node)

        for i in range(walk_length - 2):
            if curr_node.startswith('api_'):
                nbr_apis, nbr_apps = self.all_neighbors_from_api(curr_node)
                curr_nbrs = np.concatenate([nbr_apis, nbr_apps])
            elif curr_node.startswith('app_'):
                curr_nbrs = self.get_api_neighbors_A(curr_node)
            else: raise AssertionError

            alpha_1 = np.intersect1d(prev_nbrs, curr_nbrs, assume_unique=True)
            alpha_p = prev_node
            alpha_q = np.setdiff1d(
                np.setdiff1d(curr_nbrs, alpha_1, assume_unique=True),
                [alpha_p], assume_unique=True
            )
            alphas = [*alpha_1, alpha_p, *alpha_q]
            assert len(alpha_1) + len(alpha_q) + 1 == len(curr_nbrs)

            probs_q = np.full(len(alpha_q), 1/q/len(alpha_q)) if len(alpha_q) else []
            probs_1 = np.full(len(alpha_1), 1/len(alpha_1)) if len(alpha_1) else []
            probs = [*probs_1, 1/p, *probs_q]
            probs = np.array(probs) / sum(probs)

            new_node = np.random.choice(alphas, p=probs)
            path.append(new_node)
            prev_node = curr_node
            prev_nbrs = curr_nbrs
            curr_node = new_node

        return path
    
    def perform_one_walk_metapath(self, p=1, q=1, walk_length=20, app=None, metapath='APA'):
        path = []

        if metapath == 'APA':
            path_stages = ['A', 'P']

        if app is None:
            app = 'app_' + str(np.random.choice(range(self.A_tr_csr.shape[0])))
        prev_nbrs = self.get_api_neighbors_A(app)
        curr_node = np.random.choice(prev_nbrs)
        prev_node = app
        path.append(app)
        path.append(curr_node)

        prev_stage = 'A'

        for i in range(walk_length - 2):
            stage = path_stages[
                (path_stages.index(prev_stage) + 1) % len(path_stages)
            ]
            print(prev_stage, stage)

            # if curr_node.startswith('api_'):
            #     nbr_apis, nbr_apps = self.all_neighbors_from_api(curr_node)
            #     curr_nbrs = np.concatenate([nbr_apis, nbr_apps])
            # elif curr_node.startswith('app_'):
            #     curr_nbrs = self.get_api_neighbors_A(curr_node)
            # else: raise AssertionError

            if stage.startswith('A'):
                assert curr_node.startswith('app_')
                curr_nbrs = self.get_api_neighbors_A(curr_node)
            elif stage.startswith('B'):
                assert curr_node.startswith('api_')
                nbr_apps = self.get_app_neighbors_A(curr_node)
                nbr_apis = self.get_api_neighbors_B(curr_node)
                curr_nbrs = np.concatenate([nbr_apis, nbr_apps])
            elif stage.startswith('P'):
                assert curr_node.startswith('api_')
                nbr_apps = self.get_app_neighbors_A(curr_node)
                nbr_apis = self.get_api_neighbors_P(curr_node)
                curr_nbrs = np.concatenate([nbr_apis, nbr_apps])
            else: raise AssertionError

            alpha_1 = np.intersect1d(prev_nbrs, curr_nbrs, assume_unique=True)
            alpha_p = prev_node
            alpha_q = np.setdiff1d(
                np.setdiff1d(curr_nbrs, alpha_1, assume_unique=True),
                [alpha_p], assume_unique=True
            )
            alphas = [*alpha_1, *alpha_q, alpha_p]

            # print(len(alpha_1), len(alpha_q), len(curr_nbrs))
            print(prev_node, curr_node)
            # print(np.setdiff1d(alphas, curr_nbrs))
            # print(np.setdiff1d(curr_nbrs, alphas))
            assert len(alphas) == len(curr_nbrs)

            probs_1 = np.full(len(alpha_1), 1/len(alpha_1)) if len(alpha_1) else []
            probs_q = np.full(len(alpha_q), 1/q/len(alpha_q)) if len(alpha_q) else []
            probs = [*probs_1, *probs_q, 1/p]
            probs = np.array(probs) / sum(probs)

            new_node = np.random.choice(alphas, p=probs)
            print(new_node)
            if new_node in alpha_1:
                prev_stage = prev_stage
            elif new_node == alpha_p:
                prev_stage = prev_stage
            elif new_node in alpha_q:
                prev_stage = stage
            else: raise Error('Something went really wrong')

            path.append(new_node)
            prev_node = curr_node
            prev_nbrs = curr_nbrs
            curr_node = new_node

        return path
    
    def perform_walks(self, n, p, q, walk_length):
        # n is how many paths from each app
        n_apps_tr = self.A_tr_csr.shape[0]

        walks = []
        for app_i in tqdm(range(n_apps_tr)):
            app = 'app_' + str(app_i)
            
            for j in range(n):
                path = self.perform_one_walk_full(p, q, walk_length, app=app)
                walks.append(path)

        return walks

    def load_matrix(self):
        indir = self.indir
        A_tr = sparse.load_npz(os.path.join(indir, 'A_reduced_tr.npz'))
        A_tst = sparse.load_npz(os.path.join(indir, 'A_reduced_tst.npz'))
        B_tr = sparse.load_npz(os.path.join(indir, 'B_reduced_tr.npz'))
        P_tr = sparse.load_npz(os.path.join(indir, 'P_reduced_tr.npz'))

        meta_tr = pd.read_csv(os.path.join(indir, 'meta_tr.csv'), index_col=0)
        meta_tst = pd.read_csv(os.path.join(indir, 'meta_tst.csv'), index_col=0)

        assert 'csr_matrix' in str(type(A_tr))
        self.A_tr_csr = A_tr
        self.A_tr_csc = A_tr.tocsc(copy=True)
        self.A_tst = A_tst
        self.B_tr = B_tr
        self.P_tr = P_tr
        self.train_label = pd.read_csv(os.path.join(indir, 'meta_tr.csv'), index_col=None).rename(columns={'Unnamed: 0':'app_id'}).set_index('app_id')['label'].to_dict()
        self.test_label = pd.read_csv(os.path.join(indir, 'meta_tst.csv'), index_col=None).rename(columns={'Unnamed: 0':'app_id'}).set_index('app_id')['label'].to_dict()
        self.meta_tr = meta_tr
        self.meta_tst = meta_tst
        self.num_train = A_tr.shape[0]
    
    def save_corpus(self):

        walks = self.perform_walks(n=self.n, p=self.p, q=self.q, walk_length=self.walk_length)

        # add an offset for every app if in test mode
        if self.offset > 0:
            print('hi')
            for i in range(len(walks)):
                walk = walks[i]
                walks[i] = [
                    f"app_{int(node.split('_')[-1]) + self.offset}"
                    if node.startswith('app') else node
                    for node in walk
                ]

        outfile = open(self.corpus_path, 'w')

        print('saving..')
        for walk in tqdm(walks):
            outfile.write(' '.join(walk) + '\n')
        outfile.close()

    def create_model(self):
        sentences = MyCorpus(self.corpus_path)
        self.model = gensim.models.Word2Vec(
            sentences=sentences, size=64, sg=1, 
            negative=5, window=3, iter=5, min_count=1
        )

    def predict_embeddings(self):
        X = []
        Y = []
        train_labels = []
        for j in range(self.A_tr_csr.shape[0]):

            indexes = np.nonzero((self.A_tr_csr[j]).toarray()[0])[0]
            all_api = self.model.wv.vocab.keys()
            matrix = np.zeros(64)
            for i in indexes:
                element = 'api_' + str(i)
                if element in all_api:
                    matrix += self.model.wv[element]
            matrix /= len(all_api)
            X.append(matrix)
            Y.append(self.model.wv['app_' + str(j)])
            train_labels.append('app_' + str(j))

        regressor = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100).fit(X, Y)

        test_X = []
        test_labels = []
        for j in range(self.A_tst.shape[0]):
            indexes = np.nonzero((self.A_tst[j]).toarray()[0])[0]
            all_api = self.model.wv.vocab.keys()
            matrix = np.zeros(64)
            for i in indexes:
                element = 'api_' + str(i)
                if element in all_api:
                    matrix += self.model.wv[element]
            matrix /= len(all_api)
            test_X.append(matrix)
            test_labels.append('app_' + str(j + self.A_tr_csr.shape[0]))

        embeddings = regressor.predict(test_X)

        for i in range(len(test_labels)):
            self.model.wv[test_labels[i]] = embeddings[i]

        self.train_embeddings = Y
        self.test_embeddings = embeddings
        self.train_labels = self.meta_tr.label == 'class1'
        self.test_labels = self.meta_tst.label == 'class1'

    def plot_embeddings(self):
        x_vals, y_vals, labels = reduce_dimensions(self)
        
        df_dict = {'x_vals': x_vals, 'y_vals': y_vals, 'labels': labels}
        df = pd.DataFrame(df_dict)
        graph_labels = {0: 'train_benign', 1: 'train_malware', 2: 'test_benign', 3: 'test_malware'}
        df = df.replace({"labels": graph_labels})
        graph_title = self.corpus_path.split('/')[-1].split('_')[0] + " two dimensional embeddings"
        plot_with_plotly(df, graph_title)

    def train_nn(self, num_epoch=5000): 
        train_X = torch.tensor(self.train_embeddings).float()
        test_X = torch.tensor(self.test_embeddings).float()
        
        train_Y = torch.tensor(self.train_labels).float()
        test_Y = torch.tensor(self.test_labels).float()

        net = Net(train_X.shape[1])
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adamax(net.parameters(), lr=0.0001)

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

if __name__ == '__main__':
    # indirs = ['data/processed/', '/datasets/home/51/451/yuz530/group_01/pipeline_output/']
    indirs = ['data/processed/']
    
    for indir in indirs:
        outdir = os.path.join(indir, 'walks')
        if not os.path.exists(outdir):
            os.mkdir(outdir)

        A_tr = sparse.load_npz(os.path.join(indir, 'A_reduced_tr.npz'))
        A_tst = sparse.load_npz(os.path.join(indir, 'A_reduced_tst.npz'))
        B_tr = sparse.load_npz(os.path.join(indir, 'B_reduced_tr.npz'))
        P_tr = sparse.load_npz(os.path.join(indir, 'P_reduced_tr.npz'))

        n2v = Node2Vec()

        # pod 1
        # n2v.save_corpus(outdir, n=20, p=2, q=1, walk_length=300)
        n2v.save_corpus(outdir, n=15, p=2, q=1, walk_length=60)
        # n2v.save_corpus(outdir, n=200, p=2, q=1, walk_length=30)

        # pod 2
        # n2v.save_corpus(outdir, n=50, p=2, q=1, walk_length=400)

        # Test corpus using train B and P
        n2v_tst = Node2Vec(test_offset=A_tr.shape[0])
        n2v_tst.save_corpus(outdir, n=15, p=2, q=1, walk_length=60, test=True)


def node2vec_main(**cfg):
    indir = utils.PROC_DIR
    n2v = Node2Vec(indir, n=15, p=2, q=1, walk_length=60)
    n2v.load_matrix()
    n2v.save_corpus()
    n2v.create_model()
    n2v.predict_embeddings()
    n2v.plot_embeddings()
    n2v.train_nn(num_epoch=cfg['node2vec_epoch'])
    n2v.evaluate()

    # meta_tr = pd.read_csv(os.path.join(utils.PROC_DIR, 'meta_tr.csv'), index_col=0)
    # meta_tst = pd.read_csv(os.path.join(utils.PROC_DIR, 'meta_tst.csv'), index_col=0)

    # y_train = meta_tr.label == 'class1'
    # y_test = meta_tst.label == 'class1'
    # app_vec = np.array([n2v.model.wv[f'app_{i}'] for i in range(len(meta_tr))])
    # app_vec_tst = np.array([n2v_tst.model.wv[f'app_{i}'] for i in range(len(meta_tr), len(meta_tr) + len(meta_tst))])

    # from sklearn.svm import SVC
    # svm = SVC(kernel='rbf', C=10, gamma=0.1)
    # svm.fit(app_vec, y_train)
    # print(svm.score(app_vec, y_train))
    # print(svm.score(app_vec_tst, y_test))

class MyCorpus(object):
    """An interator that yields sentences (lists of str)."""
    def __init__(self, corpus_path):
        self.lines = open(corpus_path).readlines()
        # if test_corpus_path is not None:
        #     self.lines += open(test_corpus_path).readlines()  # !!! Test

    def __iter__(self):
        for line in tqdm(self.lines):
            # assume there's one document per line, tokens separated by whitespace
            yield line.strip().split(' ')

class Net(nn.Module):
    def __init__(self, n_features):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512,1)
        )
    def forward(self, x):
        return torch.sigmoid(self.classifier(x))

def calculate_accuracy(y_true, y_pred):
    predicted = y_pred.ge(.5).view(-1)
    return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
    return round(t.item(), decimal_places)