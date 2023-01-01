import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()

        self.embed_size= embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (embed_size % heads == 0), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, key_len, self.heads, self.head_dim)


        energy = torch.einsum('nqhd,nkhd->nhqk')

        return True


class EncoderBlock(nn.Module):
    def __init__(self):
        super(EncoderBlock, self).__init__()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()


class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()


    def forward(x, self):
        out = Encoder(x)
        out = Decoder(out)