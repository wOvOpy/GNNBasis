import networkx as nx
from gensim.models import Word2Vec

from joblib import Parallel, delayed
import itertools
import random

class DeepWalk(object):
    def __init__(self, graph, walk_length, num_walks, walkers=1, verbose=0, random_state=None):
        self.G = graph
        # 单个节点游走的长度
        self.walk_length = walk_length
        # 单个节点游走的序列个数
        self.num_walks = num_walks
        # 采用多少个线程生成embbeding(Word2Vec参数)
        self.walkers = walkers
        # Parallel可选参数
        self.verbose = verbose
        # 生成的Word2Vec模型
        self.w2v = None
        # 所有的节点embedding列表
        self.embeddings = None
        # 随机种子
        self.random_state = 2020
        # 训练数据集
        self.dataset = self.get_train_data(self.walk_length, self.num_walks, self.walkers, self.verbose)

    def fit(self, embed_size=128, window=5, n_jobs=3, epochs=5, **kwargs):
        kwargs["sentences"] = self.dataset
        kwargs["min_count"] = kwargs.get("min_count", 0)
        kwargs["vector_size"] = embed_size
        kwargs["sg"] = 1  # skip gram
        kwargs["hs"] = 1  # deepwalk use Hierarchical Softmax
        kwargs["workers"] = n_jobs
        kwargs["window"] = window
        kwargs["epochs"] = epochs
        kwargs["seed"] = self.random_state

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
                    walks.append(self.deep_walk(walk_length=walk_length, start_node=v))
        return walks

    def deep_walk(self, walk_length, start_node):

        G = self.G

        walk = [start_node]
        while len(walk) < self.walk_length:
            current_node = walk[-1]
            current_nerghbors = list(G.neighbors(current_node))
            if len(current_nerghbors) > 0:
                walk.append(random.choice(current_nerghbors))
            else:
                break
        return walk

    def get_embeddings(self):
        if self.w2v:
            self.embeddings = {}
            for node in self.G.nodes():
                self.embeddings[node] = self.w2v.wv[node]
            return self.embeddings
        else:
            print("Please train the model first")
            return None