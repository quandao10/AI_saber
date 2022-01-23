from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Norm(nn.Module):
    def __init__(self, fn, dim):
        super(Norm, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-5)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class Residual(nn.Module):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super(Attention, self).__init__()
        self.heads = heads
        self.head_dim = int(dim / heads)

        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        # x: [batch_size, seq_len, dim]
        batch_size, seq_len, dim = x.size()
        qkv = self.to_qkv(x)
        qkv = qkv.view(batch_size, seq_len, self.heads, self.head_dim * 3)

        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, _ = self.scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, dim)
        values = self.to_out(values)
        return values

    def scaled_dot_product(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention


# model = Attention(800, 10).to("cuda")
# input = torch.randn(32, 960, 800).to("cuda")
# output = model(input)
# print("output_size", output.size())


class TransformerEncoder(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_hidden_layers,
                 num_attention_heads,
                 intermediate_size):
        super(TransformerEncoder, self).__init__()
        blocks = []
        for i in range(num_hidden_layers):
            blocks.extend([
                ('block_{}_attn'.format(i), Residual(Norm(Attention(hidden_size, heads=num_attention_heads), hidden_size))),
                ('block_{}_mlp'.format(i), Residual(Norm(MLP(hidden_size, hidden_size, intermediate_size), hidden_size))),
            ])
        self.net = nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        return self.net(x)


# model = TransformerEncoder(num_hidden_layers = 2, num_attention_heads=10, hidden_size=800, intermediate_size=3072).to("cuda")
# input = torch.randn(32, 960, 800).to("cuda")
# output = model(input)
# print(output.size())


class LinearEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearEmbedding, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.fc(x)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, seq_length):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(seq_length, dim)
        position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class CrossModalLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, out_dim, num_hidden_layers):
        super(CrossModalLayer, self).__init__()
        self.transformer_layer = TransformerEncoder(hidden_size=hidden_size,
                                                    num_hidden_layers=num_hidden_layers,
                                                    num_attention_heads=num_attention_heads,
                                                    intermediate_size=intermediate_size)
        self.cross_output_layer = nn.Linear(hidden_size, out_dim)
        self.softmax2d = nn.Softmax2d()

    def forward(self, x, y):
        _, _, x_len = x.size()
        _, _, y_len = y.size()
        if x_len != y_len:
            raise ValueError('x and y should have the same length')
        merge_seq = torch.cat([x, y], dim=1)
        merge_seq = self.transformer_layer(merge_seq)
        logits = self.cross_output_layer(merge_seq)
        batch_size, seq_len, dim = logits.size()
        logits = logits.view(batch_size, seq_len, -1, 20)
        logits = self.softmax2d(logits)
        return logits
