# %%
import spacy

# import numpy as np
import networkx as nx
import dgl
import torch as th
# import torch.nn as nn

import matplotlib.pyplot as plt

from models import TreeLSTM
from preprocess import preprocess as prep


# %%
def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, with_labels=True, node_size=300,
            node_color=[[.5, .5, .5]], arrowsize=10)
    plt.show()


# %%
class Vocabulary:
    # map word to ints
    def __init__(self, start=0):
        self.vocab = {}
        self.reverse_vocab = {}
        self.__next = start

    def add(self, word):
        # Add word to the vocabulary dict
        if word not in self.vocab:
            self.vocab[word] = self.__next
            self.reverse_vocab[self.__next] = word
            self.__next += 1
        return self.vocab.get(word)

    def __getitem__(self, item):
        # Get the word id of given word or vise versa
        if isinstance(item, int):
            return self.reverse_vocab.get(item, None)
        else:
            return self.vocab.get(item, None)

    def __len__(self):
        # Get vocabulary size
        return len(self.vocab)


# %%
# class TreeLSTMCell(nn.Module):
#     def __init__(self, xemb_size, h_size, N=2):
#         super(TreeLSTMCell, self).__init__()
#         self.W_iou = nn.Linear(xemb_size, 3 * h_size, bias=False)
#         self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
#         self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))

#         self.W_f = nn.Linear(xemb_size, h_size, bias=False)
#         self.U_f = nn.Linear(N * h_size, N * h_size)
#         self.b_f = nn.Parameter(th.zeros(1, h_size))

#     def message_func(self, edges):
#         return {"h": edges.src["h"],
#                 "c": edges.src["c"],
#                 "type_n": edges.src["type_n"]}

#     def reduce_func(self, nodes):

#         # A. Sizes abbreviation:
#             # Nt     = B: Batch size or total number of nodes (Nt)
#             # Nchn   = Number of children nodes in the batch
#             # H      = LSTM's Hidden size
#         #
#         # B. General
#         # -----------
#             # Here we get the children's information that the `message_func` has
#             # sent: h, c, type_n. The data is retained in `nodes.mailbox`. The
#             # return of this function is sent to the next function,
#             # `apply_node_func`.
#             #
#             # We receive h and c in a tensor of size (Nt, Nchn, H). Because the
#             # number of children in the batch may vary, the `reduce_function`
#             # collects/groups the information according to the `Nchn`. Then it
#             # calls itself iteratively to process each group separately.  The
#             # function then stacks the results vetically and sends them. Thus,
#             # the dimensions other than Dimension(0) must be equal to each
#             # other. Also, the number of rows, Dimension(0), must be equal to
#             # the number of nodes (batch size).
#         #
#         # C. The forget gate eqn:
#         # -----------------------
#             # assuming the following:
#             #   1. For nodes in a graph [Ng], the number of nodes = n
#             #   2. For node-t ∈ Ng (1<=t<=N):
#             #      a. Child nodes of node-t is [Nct]
#             #
#             #      b. number of children of node-t: Nchn(t) = ℓ, where For a
#             #      node-r: Nchn(r) may not be equal to ℓ and r ≠ t and r ∈ [Ng]
#             #
#             #      c. the hidden states for the child nodes htl = [hi] where
#             #         1 <= i <= ℓ d.
#             #         Each child node is either of type_n0 or type_n1;
#             #         the hidden state for typn_0 is [h_t0] and for type_n1 is
#             #         [h_t1], where
#             #         [h_t0] = Sum(hj), where 1 <= j <= ℓ & m(j)=type_n0
#             #         [h_t1] = Sum(hj), where 1 <= j <= ℓ & m(j)=type_n1
#             #
#             #      e. Node-t have ℓ forget gates; a gate for each child
#             #
#             # In [1] eqn 4, the second part of the forget gate (Sum(U*h)) could
#             # be written as follows:
#             #   - For each node in the child nodes (Nct): The forget gate is
#             #     either a type_0 (f0) or (f1).  where:
#             #     f0 = U00 h_t0 + U01 h_t1,  eq(a)
#             #     f1 = U10 h_t0 + U11 h_t1   eq(b)
#         #
#         # D. i,o,u eqn:
#         # --------------
#             # For node_t:
#             # i_t = U_i0 h_t0 + U_i1 h_t1   eq(c)
#             # o_t = U_o0 h_t0 + U_o1 h_t1   eq(d)
#             # u_t = U_u0 h_t0 + U_u1 h_t1   eq(e)
#         #
#         # **Example:**:
#         # -------------
#             # - Assuming a node-t in a graph: node-1:
#             # - node-1 have 5 child nodes: Nct=[n1, n2, n3, n4, n5].
#             # - The types of child nodes are as follows [0, 1, 1, 0]
#             # - Ignoring the fixed parts in the forget gates' equation: Wx & b:
#             #     * the forget gate for each child node will be as follows:
#             #       For node-k that is child of node-t:
#             #         ftk = Um(tk)m(tl) * htl,
#             #         where: tl ∈ Nct, 1 <= tl <= ℓ=5 & m(lt)=is either 0 or 1
#             # - For each child node the equation are:
#             #    child-node-1: f11 = U00 h11 + U01 h12 + U01 h13 + U00 h14
#             #    child-node-2: f12 = U10 h11 + U11 h12 + U11 h13 + U10 h14
#             #    child-node-3: f13 = U10 h11 + U11 h12 + U11 h13 + U10 h14
#             #    child-node-4: f14 = U00 h11 + U01 h12 + U01 h13 + U00 h14
#             #    child-node-5: f15 = U10 h11 + U11 h12 + U11 h13 + U10 h14
#             #
#             # - The equation of child-node 1,4 (type_n0) are equal to each
#             #   other, the same are for child nodes 2,3, (type_n1).
#             #
#             # - The equations can then be reduced to be as follows:
#             #   forget type_0: f0 = U00 h11 + U01 h12 + U01 h13 + U00 h14
#             #   forget type_1: f1 = U10 h11 + U11 h12 + U11 h13 + U10 h14
#             #
#             # - Further reduction can be done as follows:
#             #   forget type_0: f0 = U00 (h11 + h14) + U01 (h12 + h13)
#             #   forget type_1: f1 = U10 (h11 + h14) + U11 (h12 + h13)
#             #   h_t0 = (h11 + h14)
#             #   h_t1 = (h12 + h13), see section A.c above.
#             #
#             #   f0 = U00 h_t0 + U01 h_t1
#             #   f1 = U10 h_t0 + U11 h_t1
#             #   where ht_0 is hidden states for type_n0 child nodes and ht_1 is
#             #   hidden states for type_n1 child nodes.
#         #
#         # - E. Impelemntation:
#         # --------------------
#             # Step:1 Get ht_0 anf ht_1:
#             # ******************
#                 # 1. Get hidden states for each node type: ht_0, ht_1
#                 #    a. Get nodes' id that are belong to each node type
#                 #       (type: 0 & 1)
#                 #    b. Using indices get h and c for each nod type "ht_0, ht_1"
#                 #    c. If there is no specific node type,
#                 #       the respective ht_0 or ht_1 are zeros
#                 #
#             # Step:2 i,o,t gates: based on eqs(c,d,e) Under section D:
#             # **************************************************
#                 #  a. [ht_0, ht_1] [   Uiot   ] = [i, o, t]
#                 #        (Nt , 2H)   (2H , 3H)   = (Nt , 3H)
#                 #
#                 #  b. return [i, o, t]
#                 #
#             # Step:3 Forget gate: based on eqs(a,b) Under section C:
#             # ************************************************
#                 #   a. [ht_0, ht_1] [    Uf    ] =  [f0, f1]
#                 #         (Nt , 2H)   (2H , 2H)  =  (Nt , 2H)
#                 #
#                 #   b. Then, construct a tensor f_cell (Nt, Nchn, H) ,
#                 #      where each tensor at (Nt, Nchn) is either
#                 #      f_0 or f_1 according to the type of the respective
#                 #      child node. for the example in section C the matrix
#                 #      f_cell (1, 4, H) = [f0; f1; f1; f0]
#                 #
#                 #   c. f_tk = sigma( W X_emb + f_cell + b)
#                 #      The size of f_tk, [W X_emb] and f_cell = (Nt, Nchn, H)
#                 #      The size of b is (1, H)
#                 #
#                 #   d. c_cell = SUM(mailbox(c) . f_tk) over Dimension(Nchn)
#                 #      The size of c mailbox(c) = size of f_tk
#                 #      c_cell size = (Nt, H)
#                 #
#                 #   e. return c_cell

#         c_child = nodes.mailbox["c"]              # (Nt, Nchn, H)
#         h_child = nodes.mailbox["h"]              # (Nt, Nchn, H)
#         childrn_num = c_child.size(1)
#         hidden_size = c_child.size(2)

#         # Step 1
#         type_n = nodes.mailbox["type_n"]   # (Nt)
#         type_n0_id = (type_n == 0)  # 1.a
#         type_n1_id = (type_n == 1)  # 1.a

#         # 1.b: creat mask matrix with the same size of h and c with zeros at
#         # either type_0 node ids or type_1 node ids
#         mask = th.zeros((*h_child.size()))
#         mask[type_n0_id] = 1                # mask one at type_0 nodes
#         ht_0 = (mask * h_child)             # (Nt, Nchn, H)
#         ht_0 = th.sum(ht_0, dim=1)          # sum over child nodes => (Nt, H)

#         mask = th.zeros((*h_child.size()))  # do the same for type_1
#         mask[type_n1_id] = 1
#         ht_1 = (mask * h_child)             # (Nt, Nchn, H)
#         ht_1 = th.sum(ht_1, dim=1)          # sum over child nodes => (Nt, H)

#         # # Step 2
#         h_iou = th.cat((ht_0, ht_1), dim=1)     # (Nt, 2H)

#         # Step 3
#         # (Nt, 2H) => (Nt, 2, H)
#         f = self.U_f(th.cat((ht_0, ht_1), dim=1)).view(-1, 2, hidden_size)
#         # 3.b select from f either f_0 or f_1 using type_n as index
#         # generate array repeating elements of nodes_id by their number of
#         # children. e.g. if we have 3 nodes that have 2 children.
#         # select_id = [0, 0, 1, 1, 2, 2]
#         select_id = np.repeat(range(c_child.size(0)), c_child.size(1))
#         f_cell = f[select_id, type_n.view(-1), :].view(*c_child.size())

#         # Steps 3.c,d
#         X = self.W_f(nodes.data["emb"])                     # (Nt, H)
#         X = X.repeat(childrn_num, 1).view(*c_child.size())  # (Nt, Nchn, H)
#         f_tk = th.sigmoid(X + f_cell + self.b_f)            # (Nt, Nchn, H)
#         c_cell = th.sum(f_tk * c_child, dim=1)              # (Nt, H)

#         return {"h": h_iou, "c": c_cell}

#     def apply_node_func(self, nodes):
#         # The leaf nodes have no child the h_child and c_child are zeros!
#         h_cell = nodes.data["h"]
#         c_cell = nodes.data["c"]

#         # Initialize hand c states for leaves nodes
#         if nodes._graph.srcnodes().nelement() == 0:
#             # initialize h states, for node type-0 and noe type-1
#             h_cell = th.cat((h_cell, h_cell), dim=1)           # (Nt, Nchn*H)

#         iou = self.W_iou(nodes.data["emb"]) + self.U_iou(h_cell) + self.b_iou
#         i, o, u = th.chunk(iou, 3, 1)       # (Nt x H) for each of i,o,u
#         i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)

#         c = i * u + c_cell
#         h = o * th.tanh(c)

#         return {"h": h, "c": c}


# class Embedings(nn.Module):
#     def __init__(   self,
#                     num_vocabs,
#                     emb_size,
#                     emb_type=None,
#                     pretrained_emb=None):
#         super(Embedings, self).__init__()

#         if emb_type is not None and pretrained_emb is None:
#             raise ValueError("missing pretrained embedding \"pretrained_emb\"")

#         self.embed = nn.Embedding(num_vocabs, emb_size)
#         if emb_type == "Glove":
#             self.embed.weight.data.copy_(pretrained_emb)
#             self.embed.weight.requires_grad = True
#         else:
#             pass


# class TreeLSTM(nn.Module):
#     def __init__(   self,
#                     num_vocabs,
#                     x_size,
#                     h_size,
#                     dropout=0,
#                     bidirection=False
#                     ):
#         super(TreeLSTM, self).__init__()
#         self.x_size = x_size
#         self.embeddings = Embedings(num_vocabs, x_size)
#         self.dropout = nn.Dropout(dropout)
#         self.cell = TreeLSTMCell(x_size, h_size)

#     def forward(self, g, h, c):
#         """Compute tree-lstm prediction given a batch.
#         Parameters
#         ----------
#         batch : dgl.data.SSTBatch
#             The data batch.
#         g : dgl.DGLGraph
#             Tree for computation.
#         h : Tensor
#             Initial hidden state.
#         c : Tensor
#             Initial cell state.
#         Returns
#         -------
#         logits : Tensor
#             The prediction of each node.
#         """
#         embeddings = self.embeddings.embed(g.ndata["word"])
#         g.ndata['emb'] = self.dropout(embeddings)
#         g.ndata['h'] = h
#         g.ndata['c'] = c
#         # propagate
#         dgl.prop_nodes_topo(g,
#                             message_func=self.cell.message_func,
#                             reduce_func=self.cell.reduce_func,
#                             apply_node_func=self.cell.apply_node_func)
#         h = self.dropout(g.ndata.pop('h'))
#         return h


# %%
def dummy_word2id(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    vocab = Vocabulary()
    for token in doc:
        vocab.add(token.text)
    return vocab.vocab


# %%
text = """From this point of view, I firmly believe that we should attach more importance to cooperation during primary education.
First of all, through cooperation, children can learn about interpersonal skills which are significant in the future life of all students."""

vocab = dummy_word2id(text)
# NOTE: What is include in the batch? Will the text be included in the batch? if not, I need to change the preprocess.DependencyDGL to compile the text from word_id (in the vocabulary) to text, then parse the compiled text. Other option is to parse the whole documents during preprocessing and incluse it in the batch.
dep = prep.DependencyDGL(text, vocab)

# %%
G = dep.g
dep.graph_process([19, 44], 0)
# G = dgl.batch([g])
print(G.ndata["type_n"])
plot_tree(dgl.to_homogeneous(G).to_networkx())


# %%
n = G.number_of_nodes()
h_size = 5
emb_size = 6

h = th.zeros((n, h_size))
c = th.zeros((n, h_size))

model = TreeLSTM(   len(vocab),
                    emb_size,
                    h_size,
                    0.5,
                    False
                    )


# %%
model(G, h, c)
# %%
