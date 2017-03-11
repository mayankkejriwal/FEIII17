import numpy as np
from sklearn.preprocessing import normalize
import warnings


def generate_doc_embedding(word2vec_model, tokens_list, num_dims=30):
    count = 0
    big_vec = np.zeros(num_dims)
    for token in tokens_list:
        try:
            vec = word2vec_model[token]
            count += 1
            big_vec = np.add(big_vec, vec)
        except:
            continue
    return _normalize_vector(big_vec).tolist()


def _normalize_vector(vector):
    """
    l2-normalizer
    :param vector:
    :return: A normalized vector. Original vector is not modified.
    """
    warnings.filterwarnings("ignore")
    return normalize(vector)[0]

