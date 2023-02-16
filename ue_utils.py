from sklearn import preprocessing
from tqdm import tqdm
import numpy as np
import torch

# latent distance
def standardized_euclidean_distance(support_vectors, query_vectors):
    # support_vectors = preprocessing.scale(support_vectors, axis=1)
    # query_vectors = preprocessing.scale(query_vectors, axis=1)
    eu_dist = []
    for qv in tqdm(query_vectors):
        d = [np.sqrt(np.sum(np.square(qv - sv))) for sv in support_vectors]
        eu_dist.append(d)
    return eu_dist

# mc_dropout
def apply_dropout(m):
    if type(m) == torch.nn.Dropout:
        m.train()

def mc_dropout(preds_probas):
    '''https://www.depends-on-the-definition.com/model-uncertainty-in-deep-learning-with-monte-carlo-dropout/'''
    posterior_vars = np.std(preds_probas, axis=0) ** 2
    posterior_vars_c0 = posterior_vars[:, 0]
    return posterior_vars_c0

def get_nearest_train_list(eu_dist_list, nearest_number)
    dist_copy = copy.deepcopy(eu_dist_list)
    min_distance = []
    min_distance_index = []
    nearest_train_id = []
    for _ in range(nearest_number):
        distance = min(dist_copy)
        index = dist_copy.index(distance)
        dist_copy[index] = 10000
        train_id = train_df['index'][index]
        min_distance.append(distance)
        min_distance_index.append(index)
        nearest_train_id.append(train_id)
    return min_distance, min_distance_index, nearest_train_id