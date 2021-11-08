import networkx as nx
from networkx.utils.decorators import random_state
import numpy as np
from gensim.models import Word2Vec
from joblib import Parallel, delayed
import itertools
import random
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


SEED = 2020


def create_alias_table(probs):
    """
    :param probs: sum(probs)=1
    :return: accept,alias
    """
    L = len(probs)
    accept, alias = [0] * L,  [0] * L
    small, large = [], []
    for i, prob in enumerate(probs):
        accept[i] = prob * L
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = probs[small_idx]
        alias[small_idx] = large_idx
        probs[large_idx] = probs[large_idx] - (1 - probs[small_idx])
        if probs[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]
    
    
class Node2Vec(object):
    def __init__(self, graph, p, q, walk_length, num_walks, workers=1, verbose=0, seed=2020):
        self.G = graph

        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.workers = workers
        self.alias_nodes = None
        self.alias_edges = None
        self.verbose = verbose

        self.w2v = None
        self.embeddings = None
        self.seed = seed

        self.preprocess_transition_probs()
        self.dataset = self.get_train_data(self.walk_length, self.num_walks, self.workers, self.verbose)

    def fit(self, embed_size=128, window=5, n_jobs=3, epochs=5, **kwargs):
        kwargs["sentences"] = self.dataset
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 0  # node2vec not use Hierarchical Softmax
        kwargs["workers"] = n_jobs
        kwargs["window"] = window
        kwargs["epochs"] = epochs
        kwargs["seed"] = self.seed

        self.w2v = Word2Vec(**kwargs)

    def get_train_data(self, walk_length, num_walks, workers=1, verbose=0):
        if num_walks % workers == 0:
            num_walks = [num_walks//workers]*workers
        else:
            num_walks = [num_walks//workers]*workers + [num_walks % workers]

        nodes = list(self.G.nodes())

        results = Parallel(n_jobs=workers, verbose=verbose)(
            delayed(self.simulate_walks)(nodes, num, walk_length) for num in num_walks
        )

        dataset = list(itertools.chain(*results))
        return dataset

    def simulate_walks(self, nodes, num_walks, walk_length,):
        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                    walks.append(self.node2vec_walk(walk_length=walk_length, start_node=v))
        return walks

    def node2vec_walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges
        walk_length = self.walk_length

        walk = [start_node]
        while len(walk) < walk_length:
            current_node = walk[-1]
            current_nerghbors = list(G.neighbors(current_node))
            if len(current_nerghbors) > 0:
                if len(walk) == 1:
                    walk.append(
                        current_nerghbors[alias_sample(alias_nodes[current_node][0], alias_nodes[current_node][1])]
                    )
                else:
                    previous_node = walk[-2]
                    edge = (previous_node, current_node)
                    next_node = current_nerghbors[
                                                    alias_sample(alias_edges[edge][0],   alias_edges[edge][1])
                    ]
                    walk.append(next_node)
            else:
                break
        return walk

    def get_alias_edge(self, t, v):
        """
        2阶随机游走，顶点间的转移概率
        :param t: 上一顶点
        :param v: 当前顶点
        """
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for x in G.neighbors(v):
            weight = G[v][x].get('weight', 1.0)  # w_vx，无权图权重设为1
            if x == t:  # d_tx == 0
                unnormalized_probs.append(weight/p)
            elif G.has_edge(x, t):  # d_tx == 1
                unnormalized_probs.append(weight)
            else:  # d_tx > 1
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [
                            float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return create_alias_table(normalized_probs)

    def preprocess_transition_probs(self):
        """
        在随机游走之前进行初始化
        """
        G = self.G
        alias_nodes = {}
        for node in G.nodes():
            unnormalized_probs = [
                                  G[node][nbr].get('weight', 1.0) for nbr in G.neighbors(node)]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                                float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = create_alias_table(normalized_probs)

        alias_edges = {}
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        
        self.alias_edges = alias_edges
        self.alias_nodes = alias_nodes

    def get_embeddings(self):
        if self.w2v:
            self.embeddings = {}
            for node in self.G.nodes():
                self.embeddings[node] = self.w2v.wv[node]
            return self.embeddings
        else:
            print("Please train the model first")
            return None
 


class Classifier(object):
    def __init__(self, embeddings, clf):
        self.embeddings = embeddings
        self.clf = clf
        self.label_encoder = MultiLabelBinarizer()

    def evaluate(self, y_true, y_pred):
        average_list = ["micro", "macro", "samples", "weighted"]
        results = {}
        y_true = self.label_encoder.transform(y_true)
        y_pred = self.label_encoder.transform(y_pred)
        for average in average_list:
            results[average] = f1_score(y_true, y_pred, average=average)
        return results
        
    def split_train_evaluate(self, X, Y, test_size=0.2, **kwargs):
        X = [self.embeddings[x] for x in X]
        self.label_encoder.fit(Y)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, **kwargs)
        self.clf.fit(X_train, Y_train)
        y_pred = self.clf.predict(X_test)
        return self.evaluate(Y_test, y_pred)       
 

def evaluate_embeddings(embeddings, clf, X, Y, **kwargs):
    clf = Classifier(embeddings, clf)
    res = clf.split_train_evaluate(X, Y, **kwargs)
    return res
       
def plot_embeddings(X, Y, embeddings,):
    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i], [])
        color_idx[Y[i]].append(i)

    plt.figure(figsize=(15, 10))
    for c, idx in color_idx.items():
        plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)
    plt.legend()
    plt.show()
        
def main():
    X, Y = [], []
    with open("../wiki/Wiki_labels.txt", "r") as f:
        for line in f:
            x, y = line.split()
            X.append(x)
            Y.append(y)
            
    G = nx.read_edgelist('../wiki/Wiki_edgelist.txt', create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    model = Node2Vec(G, p=0.25, q=4, walk_length=10, num_walks=80, workers=3, verbose=0, seed=SEED)
    model.fit(embed_size=128, window=5, n_jobs=3, epochs=3)
    embeddings = model.get_embeddings()
    clf = LogisticRegression(solver="liblinear", random_state=SEED)
    evaluate_embeddings(embeddings, clf, X, Y, test_size=0.2, random_state=SEED)
    plot_embeddings(X, Y, embeddings)

if __name__ == "__main__":
    main()