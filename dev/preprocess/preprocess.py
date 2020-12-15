import spacy
import torch as th
import dgl
# import matplotlib.pyplot as plt


class DEPENDENCYDGL():
    def __init__(self, text, comp_idxs, device="cpu"):
        # parse text
        self.device = device
        self.text_parsed = self.__dep_parse(text)
        self.g = self.__dep_graph(comp_idxs)

    def __dep_parse(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return doc

    def __dep_graph(self, comp_idxs):
        # source node, destination node, node type (u1, v1, n_type)
        u1 = []; v1 = []; n_type = []

        for token in self.text_parsed:
            # node type = 1 if token in argument component else 0
            n_type.append(int(token.i in comp_idxs))

            if token.dep_ == "ROOT":
                continue
            else:
                # Forward path: bottom-up = leaves-root
                u1.append(token.head.i)
                v1.append(token.i)

        g_lr = dgl.graph((u1, v1), device=self.device)  # leaves to root
        g_lr.ndata["type"] = th.tensor(n_type, dtype=th.uint8)
        return g_lr


if __name__ == "__main__":
    # dummy example
    text = """From this point of view, I firmly believe that we should attach more importance to cooperation during primary education.
    First of all, through cooperation, children can learn about interpersonal skills which are significant in the future life of all students."""
    s1 = list(range(10, 20))
    s2 = list(range(26, 46))
    s1.extend(s2)
    dep = DEPENDENCYDGL(text, s1)
