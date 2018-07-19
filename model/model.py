#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Preprocessor(nn.Module):
    """"""

    def __init__(self, vocab_size, embed_size, hidden_size, pretrained_emb=None):
        """"""
        super(Preprocessor, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # initialize with pretrained
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.W_i = nn.Parameter(torch.FloatTensor(self.embed_size, self.hidden_size))
        self.b_i = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.W_u = nn.Parameter(torch.FloatTensor(self.embed_size, self.hidden_size))
        self.b_u = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W_i.data.uniform_(-stdv, stdv)
        self.W_u.data.uniform_(-stdv, stdv)

    def forward(self, inputs):
        """
        """
        inputs = self.embedding(inputs)  # b x l x d

        W = torch.bmm(inputs, self.W_i.unsqueeze(0).expand(inputs.size(0), *self.W_i.size()))  # b x l x h
        W = W + self.b_i

        U = torch.bmm(inputs, self.W_u.unsqueeze(0).expand(inputs.size(0), *self.W_u.size()))
        U = U + self.b_u

        outputs = F.sigmoid(W) * F.tanh(U)

        return outputs


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, bidirectional=False, pretrained_emb=None):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.birnn = bidirectional
        if bidirectional:
            self.n_cells = 2
        else:
            self.n_cells = 1

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # initialize with pretrained
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.encoder = nn.LSTM(embed_size, hidden_size, bidirectional=self.birnn, num_layers=1, dropout=.3)

    def forward(self, inputs, input_length, pooling='max'):
        """"""
        embedded = self.embedding(inputs)  # batch x seq x dim
        embedded = embedded.permute(1, 0, 2)  # seq x batch x dim

        batch_size = embedded.size()[1]
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = Variable(embedded.data.new(*state_shape).zero_()).cuda()

        packed_input = pack_padded_sequence(embedded, input_length.cpu().numpy())
        packed_output, (ht, ct) = self.encoder(packed_input, (h0, c0))
        outputs, _ = pad_packed_sequence(packed_output)

        if pooling == 'raw':
            return outputs  # len x batch x 2*h

        if pooling == 'last':
            return ht[-1] if not self.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

        outputs = outputs.permute(1, 0, 2)
        if pooling == 'max':
            return torch.max(outputs, 1)[0]  # return values and index --> first values
        else:
            return torch.mean(outputs, 1)


class SelfAttentionLayer(nn.Module):
    """"""

    def __init__(self, hidden_size, hops=5):
        """"""
        super(SelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.hops = hops

        self.W_w = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        self.b_w = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.U_w = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hops))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = .1
        self.W_w.data.uniform_(-stdv, stdv)
        self.U_w.data.uniform_(-stdv, stdv)

    def forward(self, inputs, pooling='sum'):
        """
        """
        # print('inputs', inputs)
        M = torch.bmm(inputs, self.W_w.unsqueeze(0).expand(inputs.size(0), *self.W_w.size()))  # batch x len x h
        M += self.b_w

        M = F.tanh(M)

        U = torch.bmm(M, self.U_w.unsqueeze(0).expand(M.size(0), *self.U_w.size()))  # batch x len x hops
        alpha = F.softmax(U, dim=1)  # b x l x hops

        # b x hops x h
        return torch.bmm(alpha.permute(0, 2, 1), inputs).permute(1, 0, 2)  # hops x b x h


class CNN(nn.Module):
    def __init__(self, vocab_size, embed_size, out_channels, pretrained_emb=None):
        super(CNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.out_channels = out_channels

        self.kernel_size = 3

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        # initialize with pretrained
        if pretrained_emb is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.convs1 = nn.Conv2d(1, self.out_channels, (self.kernel_size, self.embed_size), stride=1,
                                padding=(self.kernel_size // 2, 0))

    def forward(self, inputs, pooling='max'):
        """
        """
        inputs = self.embedding(inputs)  # batch x seq x dim

        inputs = inputs.unsqueeze(1)  # b x 1 x l x h

        outputs = F.relu(self.convs1(inputs))  # (batch, Co, seq, 1)
        outputs = outputs.squeeze(3)  # batch x Co x seq

        if pooling == 'max':
            outputs = F.max_pool1d(outputs, outputs.size(2))  # b x Co x 1
            return outputs.squeeze(2)

        outputs = outputs.permute(0, 2, 1)  # b x l x Co
        return outputs


class SequentialAttention(nn.Module):
    """"""

    def __init__(self, hidden_size):
        """"""
        super(SequentialAttention, self).__init__()

        self.hidden_size = hidden_size // 2

        self.encoder = nn.LSTM(hidden_size, self.hidden_size, bidirectional=True, num_layers=1, dropout=.2)

    def forward(self, inputs, pooling='sum'):
        """
        """
        y = inputs[0] * inputs[1].unsqueeze(1).expand_as(inputs[0])  # b x l x h

        y = y.permute(1, 0, 2)  # l x b x h

        batch_size = y.size()[1]
        state_shape = 2, batch_size, self.hidden_size
        h0 = c0 = Variable(y.data.new(*state_shape).zero_()).cuda()

        outputs, _ = self.encoder(y, (h0, c0))

        outputs = outputs.permute(1, 0, 2)  # len x batch x h --> batch x len x h

        outputs = torch.sum(outputs, 2).unsqueeze(2)  # b x l x 1

        alpha = F.softmax(outputs, dim=1)

        return torch.sum(inputs[0] * alpha, 1)


class MLPttentionLayer(nn.Module):
    """
    """

    def __init__(self, hidden_size, activation='tanh'):
        super(MLPttentionLayer, self).__init__()

        self.hidden_size = hidden_size
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.W_0 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_b = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W_0.data.uniform_(-stdv, stdv)
        self.W_1.data.uniform_(-stdv, stdv)
        self.W_b.data.uniform_(-stdv, stdv)

    def forward(self, inputs, pooling='sum'):
        """
        """
        M = torch.bmm(inputs[0], self.W_0.unsqueeze(0).expand(inputs[0].size(0), *self.W_0.size()))  # batch x len x h
        M += torch.mm(inputs[1], self.W_1).unsqueeze(1).expand_as(M)  # batch x h --> batch x len x h

        M = self.activation(M)  # batch x len x h

        U = torch.bmm(M, self.W_b.unsqueeze(0).expand(M.size(0), *self.W_b.size()))
        alpha = F.softmax(U, dim=1)  # batch x len x 1

        if pooling == 'max':
            return torch.max(inputs[0] * alpha, 1)[0]  # batch x h
        elif pooling == 'mean':
            return torch.mean(inputs[0] * alpha, 1)
        elif pooling == 'raw':
            return inputs[0] * alpha
        else:
            return torch.sum(inputs[0] * alpha, 1)


class BilinearAttentionLayer(nn.Module):
    """"""

    def __init__(self, hidden_size):
        super(BilinearAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W.data.uniform_(-stdv, stdv)

    def forward(self, inputs, pooling='sum'):
        """
        """
        M = torch.mm(inputs[1], self.W).unsqueeze(1).expand_as(inputs[0])  # batch x len x h
        alpha = F.softmax(torch.sum(M * inputs[0], dim=2), dim=1)  # batch x len
        alpha = alpha.unsqueeze(2).expand_as(inputs[0])

        if pooling == 'max':
            return torch.max(inputs[0] * alpha, 1)[0]  # batch x h
        elif pooling == 'mean':
            return torch.mean(inputs[0] * alpha, 1)
        elif pooling == 'raw':
            return inputs[0] * alpha
        else:
            return torch.sum(inputs[0] * alpha, 1)


class MultiHopAttention(nn.Module):
    """"""

    def __init__(self, hidden_size, num_steps=2):
        """"""
        super(MultiHopAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_steps = num_steps  # <=3 in this implementation

        self.W_u_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_m_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_h_1 = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

        self.W_u_2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_m_2 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_h_2 = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

        self.W_u_3 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_m_3 = nn.Parameter(torch.FloatTensor(self.hidden_size, self.hidden_size))
        self.W_u_h_3 = nn.Parameter(torch.FloatTensor(self.hidden_size, 1))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 0.1
        self.W_u_1.data.uniform_(-stdv, stdv)
        self.W_u_m_1.data.uniform_(-stdv, stdv)
        self.W_u_h_1.data.uniform_(-stdv, stdv)
        self.W_u_2.data.uniform_(-stdv, stdv)
        self.W_u_m_2.data.uniform_(-stdv, stdv)
        self.W_u_h_2.data.uniform_(-stdv, stdv)
        self.W_u_3.data.uniform_(-stdv, stdv)
        self.W_u_m_3.data.uniform_(-stdv, stdv)
        self.W_u_h_3.data.uniform_(-stdv, stdv)

    def forward(self, inputs, qvector=None, pooling='sum'):
        """
        """
        m_u = [None] * self.num_steps
        m_u[0] = torch.mean(inputs, 1) * qvector if qvector is not None else torch.mean(inputs, 1)
        u_att = [None] * (self.num_steps + 1)
        u_att[0] = torch.mean(inputs, 1)

        M = torch.bmm(inputs, self.W_u_1.unsqueeze(0).expand(inputs.size(0), *self.W_u_1.size()))  # b x l x h
        M = F.tanh(M)
        M = M * F.tanh(torch.mm(m_u[0], self.W_u_m_1)).unsqueeze(1).expand_as(M)
        U = torch.bmm(M, self.W_u_h_1.unsqueeze(0).expand(M.size(0), *self.W_u_h_1.size()))
        alpha = F.softmax(U, dim=1)  # batch x len x 1

        u_att[1] = torch.sum(inputs * alpha, 1)

        if self.num_steps > 1:
            m_u[1] = m_u[0] + u_att[0] * qvector if qvector is not None else m_u[0] + u_att[0]
            M = torch.bmm(inputs, self.W_u_2.unsqueeze(0).expand(inputs.size(0), *self.W_u_2.size()))  # b x l x h
            M = F.tanh(M)
            M = M * F.tanh(torch.mm(m_u[1], self.W_u_m_2)).unsqueeze(1).expand_as(M)
            U = torch.bmm(M, self.W_u_h_2.unsqueeze(0).expand(M.size(0), *self.W_u_h_2.size()))
            alpha = F.softmax(U, dim=1)  # batch x len x 1

            u_att[2] = torch.sum(inputs * alpha, 1)

        if self.num_steps > 2:
            m_u[2] = m_u[1] + u_att[1] * qvector if qvector is not None else m_u[1] + u_att[1]
            M = torch.bmm(inputs, self.W_u_3.unsqueeze(0).expand(inputs.size(0), *self.W_u_3.size()))  # b x l x h
            M = F.tanh(M)
            M = M * F.tanh(torch.mm(m_u[2], self.W_u_m_3)).unsqueeze(1).expand_as(M)
            U = torch.bmm(M, self.W_u_h_3.unsqueeze(0).expand(M.size(0), *self.W_u_h_3.size()))
            alpha = F.softmax(U, dim=1)  # batch x len x 1
            u_att[3] = torch.sum(inputs * alpha, 1)

        return u_att


class QAMatching(nn.Module):
    """"""

    def __init__(self, vocab_size, embed_size, hidden_size, bidirectional=False, pretrained_emb=None,
                 pooling='max', num_steps=2, att_method='sequential'):
        super(QAMatching, self).__init__()

        self.encoder = Encoder(vocab_size=vocab_size,
                               embed_size=embed_size,
                               hidden_size=hidden_size,
                               bidirectional=bidirectional,
                               pretrained_emb=pretrained_emb)

        self.num_steps = num_steps
        self.pooling = pooling  # [raw, max, last, mean]

        self.att_size = 2 * hidden_size
        if att_method == 'sequential':
            self.att_layer = SequentialAttention(hidden_size=self.att_size)
        elif att_method == 'mlp':
            self.att_layer = MLPttentionLayer(hidden_size=self.att_size)
        else:
            self.att_layer = BilinearAttentionLayer(hidden_size=self.att_size)

        self.qatt_layer = MultiHopAttention(hidden_size=self.att_size, num_steps=num_steps)

        self.sim_layer = nn.CosineSimilarity(dim=1, eps=1e-8)

    def single_forward(self, q_batch, q_batch_length, pooling='max'):
        q_batch_length = torch.cuda.LongTensor(q_batch_length)
        q_batch_length, q_perm_idx = q_batch_length.sort(0, descending=True)
        q_batch = q_batch[q_perm_idx]

        q_out = self.encoder(q_batch, q_batch_length, pooling=pooling)
        if pooling == 'raw':
            q_out = q_out.permute(1, 0, 2)  # len x batch x h --> batch x len x h

        q_inverse_idx = torch.zeros(q_perm_idx.size()[0]).long().cuda()
        for i in range(q_perm_idx.size()[0]):
            q_inverse_idx[q_perm_idx[i]] = i

        q_out = q_out[q_inverse_idx]

        return q_out

    def forward(self, q_batch, q_batch_length, d_batch, d_batch_length, training=True):
        q_out_raw = self.single_forward(q_batch, q_batch_length, pooling='raw')  # b x l x h

        # q_out = [None] * 3
        # q_out[0] = torch.max(q_out_raw, 1)[0]
        # q_out[1] = self.single_forward2(q_batch, q_batch_length, pooling='last')
        # q_out[2] = torch.mean(q_out_raw, 1)

        q_out = self.qatt_layer(q_out_raw)

        d_out = self.single_forward(d_batch, d_batch_length, pooling='raw')  # b x l x h
        d_batch_len = (d_batch.size()[0]) // 2

        self.num_steps = 2  # q_out.size(0)

        if training:
            sim = None
            nsim = None
            for idx in range(self.num_steps + 1):
                pd_out = self.att_layer([d_out[:d_batch_len], q_out[idx]])
                nd_out = self.att_layer([d_out[d_batch_len:], q_out[idx]])
                # s = torch.sum(pd_out[idx] * q_out[idx], dim=1)
                s = self.sim_layer(pd_out, q_out[idx])
                ns = self.sim_layer(nd_out, q_out[idx])
                sim = s if idx == 0 else sim + s
                nsim = ns if idx == 0 else nsim + ns

            return sim, nsim
        else:
            sim = None
            for idx in range(self.num_steps + 1):
                pd_out = self.att_layer([d_out, q_out[idx]])
                # s = torch.sum(d_out[idx] * q_out[idx], dim=1)
                s = self.sim_layer(pd_out, q_out[idx])
                sim = s if idx == 0 else sim + s

            return sim, None
