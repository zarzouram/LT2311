# %% [markdown]
# # Packages Import

# %%
import spacy

import networkx as nx
import dgl
import torch as th

import matplotlib.pyplot as plt

from models import TreeLSTM, Embedings, NerModule
from preprocess import preprocess as prep

# %% [markdown]
# # Custom functions and classes


# %%
def plot_tree(g):
    pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
    nx.draw(
        g,
        pos,
        with_labels=True,
        node_size=300,
        node_color=[[0.5, 0.5, 0.5]],
        arrowsize=10,
    )
    plt.show()


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


def dummy_word2id(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    vocab = Vocabulary()
    for token in doc:
        vocab.add(token.text)
    return vocab


# %% [markdown]
# # TESTING TreeLSTM

# %%

text = """From this point of view, I firmly believe that we should attach more importance to cooperation during primary education.
First of all, through cooperation, children can learn about interpersonal
skills which are significant in the future life of all students."""  # noqa: E501

vocab = dummy_word2id(text).vocab
# NOTE: What is include in the batch? Will the text be included in the batch?
# if not, I need to change the preprocess. DependencyDGL to compile the text
# from word_id (in the vocabulary) to text, then parse the compiled text. Other
# option is to parse the whole documents during preprocessing and incluse it in
# the batch.
dep = prep.DependencyDGL(text, vocab)

g = dep.g
dep.graph_process([19, 44], 0)
G = dgl.batch([g, g])
print(G.ndata["type_n"])
plot_tree(dgl.to_homogeneous(G).to_networkx())

n = G.number_of_nodes()
# Model parameters
h_size = 5
emb_size = 6

h = th.zeros((n, h_size))
c = th.zeros((n, h_size))

# Embedding words
embeddings = Embedings(len(vocab), emb_size)
G.ndata["emb"] = embeddings.embed(G.ndata["word"])

# Model construct, test model
model = TreeLSTM(emb_size, h_size)
output = model(G, h, c)
output.size()

# %% [markdown]
# # TESTING NER

#  %%
text = "From this point of view"
label = "O O T O O"
word_vocab = dummy_word2id(text).vocab
label_vocab = dummy_word2id(label)
word_seq = [word_vocab[token] for token in text.split()]
word_seq = th.tensor(word_seq, dtype=th.long).view(1, -1)
word_seq = th.cat((word_seq, word_seq), dim=0)  # batch

# Model parameters
h_size = 10
emb_size = 6
label_emb_size = 5

# Embedding words, labels embedding layer
word_embeddings = Embedings(len(word_vocab), emb_size)
word_embedding = word_embeddings.embed(word_seq)
label_embeddings = Embedings(len(label_vocab.vocab), emb_size)

# Model construct, test model
model_ner = NerModule(emb_size, emb_size, h_size, label_emb_size,
                      len(label_vocab), label_embeddings)
ner_output = model_ner(word_embedding, label_embeddings)
print(ner_output[0].size())
print(ner_output[1].size())

# %%
