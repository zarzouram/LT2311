# %%
import networkx as nx
import dgl
from preprocess import preprocess as prep
import matplotlib.pyplot as plt
import torch as th


# %%
def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=300,
            node_color=[[.5, .5, .5]], arrowsize=10)
    plt.show()


# %%
text = """First of all, through cooperation, children can learn about interpersonal skills which are significant in the future life of all students."""
s1 = list(range(10, 20))
s2 = list(range(26, 46))
s1.extend(s2)
dep = prep.DEPENDENCYDGL(text, s1)

# %%
plot_tree(dep.g.to_networkx())

# %%
g = dgl.graph(([1, 2, 3, 4, 6, 5, 7], [0, 0, 2, 2, 3, 4, 4]))
plot_tree(g.to_networkx())


# %%
def message_func(edges):
    return {'h': edges.src['h'], 'c': edges.src['c']}


def reduce_func(nodes):
    h_cat = nodes.mailbox['h']
    c_cat = nodes.mailbox['c']
    return {'iou': c_cat - h_cat, 'c': c}


def apply_node_func(nodes):
    c = nodes.data["c"]
    h = nodes.data["h"]
    return {h: "h", c: "c"}


# %%
g = dgl.graph(g.edges())
g.ndata["type"] = th.tensor([0, 0, 1, 0, 1, 1, 0, 1], dtype=th.float32)
n = g.number_of_nodes()
h = []; c = []
for i in range(n):
    h.append([i] * 3); c.append([n - i] * 3)
h = th.tensor(h, dtype=th.float32)
c = th.tensor(c, dtype=th.float32)
g.ndata["h"] = h
g.ndata["c"] = c
g.ndata["iou"] = th.zeros((n, 3))

dgl.prop_nodes_topo(g,
                    message_func=message_func,
                    reduce_func=reduce_func,
                    apply_node_func=apply_node_func)
