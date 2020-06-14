from typing import Dict, Tuple, Set
import numpy as np
import scipy
from scipy import spatial

def glove_to_dict(embedding: str, voc: Set[str] = None) -> Dict[str, np.float]:
    result = dict()
    '''
    Reads a text file storing the glove embeddings and creates a dictionary representation of it.
    '''
    for line in open(embedding, 'r'):
        line = line.split()
        if voc == None or line[0] in voc:
            result[line[0]] = np.array(line[1:]).astype(np.float)
    return result



def cos_dist(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Calculates cosine distance of two vectors.
    '''
    return spatial.distance.cosine(x, y)



def equalize_voc(emb1: Dict[str, np.float], emb2: Dict[str, np.float]) -> Tuple[Dict[str, np.float], Dict[str, np.float]]:
    '''
    Creates, from the two given embeddings, two embeddings with common vocabulary and with the same word order.
    '''
    set1 = set(emb1.keys())
    set2 = set(emb2.keys())

    #keep only words that appear in both embeddings
    voc = set1.intersection(set2)

    return {k:emb1[k] for k in voc}, {k:emb2[k] for k in voc}



def rotate_embeddings(emb1: Dict[str, np.float], emb2: Dict[str, np.float]) -> Dict[str, np.float]:
    '''
    Rotates emb1 into emb2 minimizing the distance of each corresponding point. This reduces to solving the orthogonal Procrustes problem.
    The embeddings have to have the same dimension and the same vocabulary.
    '''
    voc = set(emb1.keys())
    A = np.array([emb1[k] for k in voc])
    B = np.array([emb2[k] for k in voc])
    #get rotation matrix
    R = scipy.linalg.orthogonal_procrustes(A, B)[0]

    #if det>0 it's a proper rotation (only rotation, no reflection)
    assert(np.linalg.det(R) > 0)

    return {k:v for k,v in zip(voc, A @ R)}



def normalize_embedding(emb: Dict[str, np.float]) -> Dict[str, np.float]:
    '''
    Normalizes embeddings vectors making them unit vectors
    '''
    return {w:emb[w]/np.linalg.norm(emb[w]) for w in emb}


def word_jaccard_distance(word: str, emb1: Dict[str, np.float], emb1_tree: spatial.KDTree,
                          emb2: Dict[str, np.float], emb2_tree: spatial.KDTree, k: int=100) -> Tuple[float, Dict[str, np.float], Dict[str, np.float]]:
    '''
    Given a word, two embeddings and their KDTree representation, finds the K nearest neighbours of that word on both embeddings and computes the Jaccard distance of the two neightbour sets.
    Returns the Jaccard distance and the neighbours of each embedding with the relative distance.
    The distance used to find the neighbours is the Euclidean distance, therefore the embeddings should be normalized by having only unit vectors so that Cosine distance is proportional to Euclidean distance.
    '''
    dist_emb1, neigh_emb1 = emb1_tree.query(emb1[word], k=k)
    res_emb1 = [list(emb1.keys())[i] for i in neigh_emb1]
    
    dist_emb2, neigh_emb2 = emb2_tree.query(emb2[word], k=k)
    res_emb2 = [list(emb2.keys())[i] for i in neigh_emb2]
    
    s1, s2 = set(res_emb1), set(res_emb2)
    dist = 1 - len(s1.intersection(s2))/len(s1.union(s2))
    
    return dist , dict(zip(res_emb1, dist_emb1)), dict(zip(res_emb2, dist_emb2))
