#%%
import spacy
from spacy import displacy
import networkx as nx
import dgl
import matplotlib.pyplot as plt

nlp = spacy.load("en")

text = '''It is always said that competition can effectively promote the development of economy.'''

#%%
def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=False, node_size=10,
            node_color=[[.5, .5, .5]], arrowsize=4)
    plt.show()

#%%
doc = nlp(text)
edges = []
for token in doc:
    for child in token.children:
        edges.append(('{0}'.format(token.lower_),
                      '{0}'.format(child.lower_)))

graph = nx.Graph(edges)
g = dgl.from_networkx(graph)

#%%
plot_tree(g.to_networkx())
# %%
print(g)
# %%
