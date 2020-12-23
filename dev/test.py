# %%
import networkx as nx
import spacy
import dgl
from preprocess import preprocess as prep
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn


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
class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, N=2):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.Um0_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.Um1_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(N * h_size, N * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox['h'].size())
        c = th.sum(f * nodes.mailbox['c'], 1)
        return {'iou': self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        type_n = nodes.data["type_n"]
        nodes_lst = nodes.nodes()
        type_n0_id = (type_n == 0).nonzero().view(-1)
        type_n1_id = (type_n == 1).nonzero().view(-1)
        if type_n0_id.nelement():
            h_0 = nodes.data["h"][type_n0_id, :]
            h_0 = th.sum(h_0, dim=0)
        else:
            h_iou_0 = th.zeros(type_n0_id.size(0), 3 * h_size)

        if type_n1_id.nelement():
            h_1 = nodes.data["h"][type_n1_id, :]
        else:
            h_iou_1 = th.zeros(type_n1_id.size(0), 3 * h_size)

        h_iou = h_iou_0 + h_iou_1

        iou = nodes.data['iou'] + h_iou + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class Embedings(nn.Module):
    def __init__(   self,
                    num_vocabs,
                    emb_size,
                    emb_type=None,
                    pretrained_emb=None):
        super(Embedings, self).__init__()

        if emb_type is not None and pretrained_emb is None:
            raise ValueError("missing pretrained embedding \"pretrained_emb\"")

        self.embed = nn.Embedding(num_vocabs, emb_size)
        if emb_type == "Glove":
            self.embed.weight.data.copy_(pretrained_emb)
            self.embed.weight.requires_grad = True
        else:
            pass


class TreeLSTM(nn.Module):
    def __init__(   self,
                    num_vocabs,
                    x_size,
                    h_size,
                    dropout
                    ):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embeddings = Embedings(num_vocabs, x_size)
        self.dropout = nn.Dropout(dropout)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, g, h, c):
        """Compute tree-lstm prediction given a batch.
        Parameters
        ----------
        batch : dgl.data.SSTBatch
            The data batch.
        g : dgl.DGLGraph
            Tree for computation.
        h : Tensor
            Initial hidden state.
        c : Tensor
            Initial cell state.
        Returns
        -------
        logits : Tensor
            The prediction of each node.
        """
        embeddings = self.embeddings.embed(g.ndata["word"])
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeddings))
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g,
                            message_func=self.cell.message_func,
                            reduce_func=self.cell.reduce_func,
                            apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        return h


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
g = dep.g
dep.graph_process([19, 44], 0)
G = dgl.batch([g, g])
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
                    0.5
                    )


# %%
model(G, h, c)
# %%
