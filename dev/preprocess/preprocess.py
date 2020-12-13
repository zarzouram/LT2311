#%%
import spacy
from spacy import displacy
import networkx as nx
import dgl
import matplotlib.pyplot as plt

class DEPGRAPH():
    def __init__(self, text):
        self.__nlp = spacy.load("en_core_web_sm")
        self.dep_tree = self.__dep_parse(text)
        self.g_forward, self.g_back =  self.__dep_graph()
    
    def __dep_parse(self, text):
        doc = self.__nlp(text)
        return doc

    def __dep_graph(self):
        u1 = []; v1 = []
        u2 = []; v2 = []

        for token in self.dep_tree:
            if token.dep_ == "ROOT":
                continue
            else:
                u1.append(token.head.i)
                v1.append(token.i)

                u2.append(token.i)
                v2.append(token.head.i)

        g_top_bottom = dgl.graph((u1, v1))
        g_bottom_top = dgl.graph((u2, v2))

        return g_top_bottom, g_bottom_top

# %%
