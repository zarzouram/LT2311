# %%
from collections import namedtuple
import networkx as nx
import dgl
import torch as th
from dgl.data.tree import SSTDataset
import matplotlib.pyplot as plt

# %%
SSTBatch = namedtuple('SSTBatch', ['graph', 'mask', 'wordid', 'label'])

# Each sample in the dataset is a constituency tree. The leaf nodes
# represent words. The word is an int value stored in the "x" field.
# The non-leaf nodes have a special word PAD_WORD. The sentiment
# label is stored in the "y" feature field.
trainset = SSTDataset()  # the "tiny" set has only five trees
tiny_sst = trainset.trees
num_vocabs = trainset.num_vocabs
num_classes = trainset.num_classes

vocab = trainset.vocab  # vocabulary dict: key -> id
inv_vocab = {v: k for k, v in vocab.items()}  # inverted vocabulary dict: id -> word


# %%
def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=300,
            node_color=[[.5, .5, .5]], arrowsize=10)
    plt.show()


# %%
def message_func(edges):
    return {'h': edges.src['h'], 'c': edges.src['c']}


def reduce_func(nodes):
    h_cat = nodes.mailbox['h']
    c_cat = nodes.mailbox['c']
    return {'iou': c_cat - h_cat, 'c': c}


def apply_node_func(nodes):
    c = nodes.data["c"] * -1
    h = nodes.data["h"] * -1
    return {"h": h, "c": c}


# %%
g = tiny_sst[3]
plot_tree(g.to_networkx())

# %%
g = dgl.graph(g.edges())
# g.ndata["type"] = th.tensor([0, 0, 1, 0, 1, 1, 0, 1], dtype=th.float32)
n = g.number_of_nodes()
h = []; c = []
for i in range(n):
    h.append([i] * 3); c.append([n - i] * 3)
h = th.tensor(h, dtype=th.float32)
c = th.tensor(c, dtype=th.float32)
g.ndata["h"] = h
g.ndata["c"] = c
g.ndata["iou"] = th.zeros((n, 3))

# %%
dgl.prop_nodes_topo(g,
                    message_func=message_func,
                    reduce_func=reduce_func,
                    apply_node_func=apply_node_func)
