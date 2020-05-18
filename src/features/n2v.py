from tqdm import tqdm
import numpy as np
from scipy import sparse
import os
import gensim.models
import pandas as pd

import src.utils as utils


class Node2Vec():
    def __init__(self, A_tr, B_tr, P_tr, test_offset=0):
        assert 'csr_matrix' in str(type(A_tr))
        self.A_tr_csr = A_tr
        self.A_tr_csc = A_tr.tocsc(copy=True)
        self.B_tr = B_tr
        self.P_tr = P_tr
        self.offset = test_offset
    
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
    
    def save_corpus(self, outdir, n=1, p=2, q=1, walk_length=100, test=False):
        fp = os.path.join(
            outdir, f'node2vec_n={n}_p={p}_q={q}_wl={walk_length}.cor'
        )
        if test:
            fp = os.path.join(
                outdir, f'node2vec_n={n}_p={p}_q={q}_wl={walk_length}_test.cor'
            )
        outfile = open(fp, 'w')

        walks = self.perform_walks(n=n, p=p, q=q, walk_length=walk_length)

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

        print('saving..')
        for walk in tqdm(walks):
            outfile.write(' '.join(walk) + '\n')
        outfile.close()

        return fp

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

        n2v = Node2Vec(A_tr, B_tr, P_tr)

        # pod 1
        # n2v.save_corpus(outdir, n=20, p=2, q=1, walk_length=300)
        n2v.save_corpus(outdir, n=15, p=2, q=1, walk_length=60)
        # n2v.save_corpus(outdir, n=200, p=2, q=1, walk_length=30)

        # pod 2
        # n2v.save_corpus(outdir, n=50, p=2, q=1, walk_length=400)

        # Test corpus using train B and P
        n2v_tst = Node2Vec(A_tst, B_tr, P_tr, test_offset=A_tr.shape[0])
        n2v_tst.save_corpus(outdir, n=15, p=2, q=1, walk_length=60, test=True)


def node2vec_main():
    indir = utils.PROC_DIR
    outdir = os.path.join(indir, 'walks')
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    A_tr = sparse.load_npz(os.path.join(indir, 'A_reduced_tr.npz'))
    A_tst = sparse.load_npz(os.path.join(indir, 'A_reduced_tst.npz'))
    B_tr = sparse.load_npz(os.path.join(indir, 'B_reduced_tr.npz'))
    P_tr = sparse.load_npz(os.path.join(indir, 'P_reduced_tr.npz'))

    n2v = Node2Vec(A_tr, B_tr, P_tr)
    corpus_path = n2v.save_corpus(outdir, n=15, p=2, q=1, walk_length=60)

    # Test corpus using train B and P
    n2v_tst = Node2Vec(A_tst, B_tr, P_tr, test_offset=A_tr.shape[0])
    test_corpus_path = n2v_tst.save_corpus(outdir, n=15, p=2, q=1, walk_length=60, test=True)
    

    class MyCorpus(object, ):
        """An interator that yields sentences (lists of str)."""
        def __init__(self, corpus_path, test_corpus_path):
            self.lines = open(corpus_path).readlines()
            if test_corpus_path is not None:
                self.lines += open(test_corpus_path).readlines()  # !!! Test

        def __iter__(self):
            for line in tqdm(self.lines):
                # assume there's one document per line, tokens separated by whitespace
                yield line.strip().split(' ')

    sentences = MyCorpus(corpus_path, test_corpus_path)
    model = gensim.models.Word2Vec(
        sentences=sentences, size=64, sg=1, 
        negative=5, window=3, iter=5, min_count=1
    )

    meta_tr = pd.read_csv(os.path.join(utils.PROC_DIR, 'meta_tr.csv'), index_col=0)
    meta_tst = pd.read_csv(os.path.join(utils.PROC_DIR, 'meta_tst.csv'), index_col=0)

    y_train = meta_tr.label == 'class1'
    y_test = meta_tst.label == 'class1'
    app_vec = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr))])
    app_vec_tst = np.array([model.wv[f'app_{i}'] for i in range(len(meta_tr), len(meta_tr) + len(meta_tst))])

    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', C=10, gamma=0.1)
    svm.fit(app_vec, y_train)
    print(svm.score(app_vec, y_train))
    print(svm.score(app_vec_tst, y_test))