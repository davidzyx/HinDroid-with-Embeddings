from tqdm import tqdm
import numpy as np
from scipy import sparse


class Node2Vec():
    def __init__(self, A_tr, B_tr, P_tr):
        assert 'csr_matrix' in str(type(A_tr))
        self.A_tr_csr = A_tr
        self.A_tr_csc = A_tr.tocsc(copy=True)
        self.B_tr = B_tr
        self.P_tr = P_tr
    
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
        # weights later?
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
    
    def save_corpus(self):
        n=1
        p=2
        q=1
        walk_length=100
        
        outfile = open(f'node2vec_n={n}_p={p}_q={q}_wl={walk_length}.cor', 'w')
        walks = self.perform_walks(n=n, p=p, q=q, walk_length=walk_length)

        print('saving..')
        for walk in tqdm(walks):
            outfile.write(' '.join(walk) + '\n')
        outfile.close()
