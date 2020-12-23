import spacy
import torch as th
import networkx as nx
import dgl
# import matplotlib.pyplot as plt


class DependencyDGL():
    def __init__(self, text, word2id: dict, graph_type=0, device="cpu"):
        if device == "cpu":
            self.device = th.device('cpu')
        else:
            self.device = device

        # NOTE:
        # ## EITHER PARSE OR RECONSTRUCT ## #
        self.text_parsed = self.__dep_parse(text)
        # convert dependency output to graph
        self.g = self.__dep_graph(graph_type, word2id)

    def __dep_parse(self, text):
        # parse using Spacy.
        # TODO: give the ability to parse using external function.
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return doc

    def __dep_graph(self, graph_type: int, word2id: dict):
        # source node, token: u1
        # destination node, token head: v1
        u1 = []; v1 = []; roots = []; token_id = []

        # construct graph u_i -> v_i
        for token in self.text_parsed:
            token_id.append(word2id[token.text])
            if token.dep_ == "ROOT":
                roots.append(token.i)
            else:
                # Forward path:
                # bottom-up = leaves-root (word->head)
                u1.append(token.i)
                v1.append(token.head.i)

        # connect multiple graphs to make one graph
        # TODO:
        # add multiple ways to construct the document tree from sentences tree
        if graph_type == 0 and len(roots) > 1:
            u1.extend(roots[:-1])
            v1.extend(roots[1:])

        # graph from leaves to root direction
        g_lr = dgl.graph((u1, v1), device=self.device)
        # add vocab word_id as an attribute to each node
        g_lr.ndata["word"] = th.tensor( token_id,
                                        dtype=th.long,
                                        device=self.device)
        n = g_lr.number_of_nodes()
        g_lr.ndata["type_n"] = th.zeros((n))
        return g_lr

    def get_shortest(self, start_n, end_n):
        # get the shortest path between two nodes
        # convert dgl to networks graph
        g = self.g.to_networkx().to_undirected()
        thepath = nx.shortest_path(g, source=start_n, target=end_n)
        return thepath

    def __change_nodes_type(self, nodes):
        n = nodes.data["type_n"].size()[0]
        type_nodes = th.ones((n))
        return {"type_n": type_nodes}

    def graph_process(self, idx: list, pross_type=0):
        if pross_type == 0:  # test
            nodes_shortest = self.get_shortest(idx[0], idx[1])
            self.g.apply_nodes( func=self.__change_nodes_type,
                                v=nodes_shortest)


if __name__ == "__main__":
    # dummy example
    text = """From this point of view, I firmly believe that we should attach more importance to cooperation during primary education.
    First of all, through cooperation, children can learn about interpersonal skills which are significant in the future life of all students."""
    s1 = list(range(10, 20))
    s2 = list(range(26, 46))
    s1.extend(s2)
    dep = DependencyDGL(text, s1)
