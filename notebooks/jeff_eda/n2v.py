import numpy as np
from scipy import sparse


def test_func():
    print(np.array([]))
    return


def get_api_neighbors_A(app):
    assert app.startswith('app_')
    app_id = int(app.split('_')[1])
    neighbor_ids = np.nonzero(A_csr[app_id])[1]
    return np.array([f'api_{s}' for s in neighbor_ids])


def get_app_neighbors_A(api):
    assert api.startswith('api_')
    api_id = int(api.split('_')[1])
    neighbor_ids = np.nonzero(A_csc[:, api_id])[0]
    return np.array([f'app_{s}' for s in neighbor_ids])


def get_api_neighbors_B(api):
    assert api.startswith('api_')
    api_id = int(api.split('_')[1])
    neighbor_ids = np.nonzero(B[:, api_id])[0]
    ls = [f'api_{s}' for s in neighbor_ids]
    ls.remove(api)
    return np.array(ls)


def get_api_neighbors_P(api):
    assert api.startswith('api_')
    api_id = int(api.split('_')[1])
    neighbor_ids = np.nonzero(P[:, api_id])[0]
    ls = [f'api_{s}' for s in neighbor_ids]
    ls.remove(api)
    return np.array(ls)


def all_neighbors_from_api(api):
    assert api.startswith('api_')
    api_id = int(api.split('_')[1])
    nbr_apis = np.concatenate([
        get_api_neighbors_B(api),
        get_api_neighbors_P(api)
    ])
    nbr_apis = np.unique(nbr_apis)
    nbr_apps = get_app_neighbors_A(api)
    # weights later?
    return nbr_apis, nbr_apps


A = sparse.load_npz('./data/A.npz')
B = sparse.load_npz('./data/B.npz')
P = sparse.load_npz('./data/P.npz')
A_csr = A
A_csc = A.tocsc(copy=True)  # memory is cheap ;D


def node2vec(p, q, walk_length):
    path = []
    app = 'app_' + str(np.random.choice(np.arange(A.shape[0])))
    prev_nbrs = get_api_neighbors_A(app)
    curr_node = np.random.choice(prev_nbrs)
    prev_node = app
    path.append(app)
    path.append(curr_node)

    for i in range(walk_length - 2):
        if curr_node.startswith('api_'):
            nbr_apis, nbr_apps = all_neighbors_from_api(curr_node)
            curr_nbrs = np.concatenate([nbr_apis, nbr_apps])
        elif curr_node.startswith('app_'):
            curr_nbrs = get_api_neighbors_A(curr_node)
        else:
            raise AssertionError

        alpha_1 = np.intersect1d(prev_nbrs, curr_nbrs, assume_unique=True)
        alpha_p = prev_node
        alpha_q = np.setdiff1d(np.setdiff1d(curr_nbrs, alpha_1, assume_unique=True), [
                               alpha_p], assume_unique=True)
        alphas = [*alpha_1, alpha_p, *alpha_q]
        assert len(alpha_1) + len(alpha_q) + 1 == len(curr_nbrs)

        probs_q = np.full(len(alpha_q), 1 / q / len(alpha_q)
                          ) if len(alpha_q) else []
        probs_1 = np.full(len(alpha_1), 1 / len(alpha_1)
                          ) if len(alpha_1) else []
        probs = [*probs_1, 1 / p, *probs_q]
        probs = np.array(probs) / sum(probs)

        new_node = np.random.choice(alphas, p=probs)
        path.append(new_node)
        prev_node = curr_node
        prev_nbrs = curr_nbrs
        curr_node = new_node

    return path
