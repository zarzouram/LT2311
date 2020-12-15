"""
Papers list:

    1. Neural End-to-End Learning for Computational Argumentation Mining.
    https://arxiv.org/abs/1704.06104v2

    2. End-to-End Relation Extraction using LSTMs on Sequences and Tree Structures
    https://arxiv.org/abs/1601.00770

    3. Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.
    https://arxiv.org/abs/1503.00075

    4. A Shortest Path Dependency Kernel for Relation Extraction
    https://dl.acm.org/doi/10.3115/1220575.1220666

------

Equations List:

    1. equations (2) to (8) in [3] are (3.2) to (3.8)
    2. equations (9) to (14) in [3] are (3.9) to (3.14)
    3. equations (1) are in [1] is (1.1)

-----

This code is based on DGL's tree-LSTM implementation found in the paper [3]
DGL Implementation can be found at https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py

------

This code implements the LSTM-ER model in [1], which is based on [2]. The treeLSTM or LSTM-ER (as named in [3]) are derived from the N-ary architecture found in [3]. The N-ary treeLSTM needs that the number of children to be fixed. In [2], the RE classification module utilizes treeLSTM to process a sentence over its dependency tree. The dependency tree in nature has a varying number of children. In [3], the N-ary tree is used with constituency trees, which can be transformed into binary-tree where each node has a left and a right child-node.

To overcome such an issue, the dependency tree nodes are categorized into two classes; nodes that belong to the shortest-path and the others are not. Nodes that belong to the same class share the same weights.

As indicated in [4], the shortest path between two entities in the same sentence contains all the information required to identify the relationship between those entities. However, as stated in [1], 92% of argument components' relationships are across different sentences. We cannot identify the path that encodes all the required information to identify the relationships between two arguments components across different sentences. Our solution is to differentiate between nodes that belong to the argument components and nodes that do not.

"""

import torch as th
import torch.nn as nn
import dgl


class TreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size, N=2):
        super(TreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U1_iou = nn.Linear(h_size, 3 * h_size, bias=False)  #
        self.U2_iou = nn.Linear(h_size, 3 * h_size, bias=False)  #
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
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data['c']
        h = o * th.tanh(c)
        return {'h': h, 'c': c}


class TreeLSTM(nn.Module):
    def __init__(self,
                 num_vocabs,
                 x_size,
                 h_size,
                 num_classes,
                 dropout,
                 #  cell_type='nary',
                 pretrained_emb=None):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.embedding = nn.Embedding(num_vocabs, x_size)
        if pretrained_emb is not None:
            print('Using glove')
            self.embedding.weight.data.copy_(pretrained_emb)
            self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, num_classes)
        self.cell = TreeLSTMCell(x_size, h_size)

    def forward(self, batch, g, h, c):
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
        # feed embedding
        embeds = self.embedding(batch.wordid * batch.mask)
        g.ndata['iou'] = self.cell.W_iou(self.dropout(embeds)) * batch.mask.float().unsqueeze(-1)
        g.ndata['h'] = h
        g.ndata['c'] = c
        # propagate
        dgl.prop_nodes_topo(g, self.cell.message_func, self.cell.reduce_func, apply_node_func=self.cell.apply_node_func)
        # compute logits
        h = self.dropout(g.ndata.pop('h'))
        logits = self.linear(h)
        return logits
