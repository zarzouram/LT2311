#%%
import spacy
from spacy import displacy
import networkx as nx
import dgl
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

text = '''From this point of view, I firmly believe that we should attach more importance to cooperation during primary education.
First of all, through cooperation, children can learn about interpersonal skills which are significant in the future life of all students.'''

#%%
def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=300,
            node_color=[[.5, .5, .5]], arrowsize=10)
    plt.show()

#%%
doc = nlp(text)
u1 = []; v1 = []
u2 = []; v2 = []

for token in doc:
    if token.dep_ == "ROOT":
        continue
    else:
       u1.append(token.head.i)
       v1.append(token.i)

       u2.append(token.i)
       v2.append(token.head.i)

g1 = dgl.graph((u1, v1))
g2 = dgl.graph((u2, v2))

#%%
plot_tree(g1.to_networkx())

# %%
plot_tree(g2.to_networkx())
# %%
