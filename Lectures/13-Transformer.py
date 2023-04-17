"""
From: [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
By:
- Austin Huang
- Suraj Subramanian
- Jonathan Sum
- Khalid Almubarak
- Stella Biderman

I've edited their code for better consistency and readability.
"""

from math import log, sqrt

import torch
from torch import nn
from torch.nn.functional import log_softmax

# ----------------------------------------------------------------
#   ____                           _
#  / ___| ___ _ __   ___ _ __ __ _| |
# | |  _ / _ \ '_ \ / _ \ '__/ _` | |
# | |_| |  __/ | | |  __/ | | (_| | |
#  \____|\___|_| |_|\___|_|  \__,_|_|
# ----------------------------------------------------------------


class MultiHeadedAttention(nn.Module):
    "Multi-Head Attention module."

    def __init__(self, d_model: int, num_head: int, dropout_prob: float):
        super().__init__()
        assert d_model % num_head == 0

        self.d_per_head = d_model // num_head
        self.num_head = num_head

        self.linear_layers = nn.ModuleList()
        for _ in range(4):
            self.linear_layers.append(nn.Linear(d_model, d_model))

        self.dropout = nn.Dropout(dropout_prob)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if mask is not None:
            mask = mask.unsqueeze(1)
        num_batches = query.size(0)

        # (1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(num_batches, -1, self.num_head, self.d_per_head).transpose(1, 2)
            for lin, x in zip(self.linear_layers, (query, key, value))
        ]

        # (2) Apply attention on all the projected vectors in batch.
        # "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = scores.softmax(dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = torch.matmul(attention, value)

        # (3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(num_batches, -1, self.num_head * self.d_per_head)
        )
        del query
        del key
        del value
        return self.linear_layers[-1](x)


class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model: int, ff_dim: int, dropout_prob: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, ff_dim)
        self.w2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.w1(x).relu()))


class SublayerConnection(nn.Module):
    "A residual connection followed by a LayerNorm."

    def __init__(self, d_model: int, dropout_prob: float):
        super().__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor, sublayer: nn.Module) -> torch.Tensor:
        return x + self.dropout(sublayer(self.layernorm(x)))


# ----------------------------------------------------------------
#  _____                     _
# | ____|_ __   ___ ___   __| | ___ _ __
# |  _| | '_ \ / __/ _ \ / _` |/ _ \ '__|
# | |___| | | | (_| (_) | (_| |  __/ |
# |_____|_| |_|\___\___/ \__,_|\___|_|
# ----------------------------------------------------------------


class EncoderLayer(nn.Module):
    "An encoder layer is composed of self-attention and feed forward components."

    def __init__(self, d_model: int, num_head: int, ff_dim: int, dropout_prob: float):
        super().__init__()
        self.self_attention = MultiHeadedAttention(d_model, num_head, dropout_prob)
        self.feed_forward = PositionWiseFeedForward(d_model, ff_dim, dropout_prob)
        self.sublayer1 = SublayerConnection(d_model, dropout_prob)
        self.sublayer2 = SublayerConnection(d_model, dropout_prob)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, mask))
        return self.sublayer2(x, self.feed_forward)


class Encoder(nn.Module):
    "An encoder block comprising N encoder layers."

    def __init__(
        self,
        d_model: int,
        num_head: int,
        num_layers: int,
        ff_dim: int,
        dropout_prob: float,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(EncoderLayer(d_model, num_head, ff_dim, dropout_prob))

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ----------------------------------------------------------------
#  ____                     _
# |  _ \  ___  ___ ___   __| | ___ _ __
# | | | |/ _ \/ __/ _ \ / _` |/ _ \ '__|
# | |_| |  __/ (_| (_) | (_| |  __/ |
# |____/ \___|\___\___/ \__,_|\___|_|
# ----------------------------------------------------------------


class DecoderLayer(nn.Module):
    "A decode layer comprises self and source attention and feed forward components."

    def __init__(self, d_model: int, num_head: int, ff_dim: int, dropout_prob: float):
        super().__init__()
        self.self_attention = MultiHeadedAttention(d_model, num_head, dropout_prob)
        self.src_attention = MultiHeadedAttention(d_model, num_head, dropout_prob)
        self.feed_forward = PositionWiseFeedForward(d_model, ff_dim, dropout_prob)
        self.sublayer1 = SublayerConnection(d_model, dropout_prob)
        self.sublayer2 = SublayerConnection(d_model, dropout_prob)
        self.sublayer3 = SublayerConnection(d_model, dropout_prob)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        m = memory
        x = self.sublayer1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayer2(x, lambda x: self.src_attention(x, m, m, src_mask))
        return self.sublayer3(x, self.feed_forward)


class Decoder(nn.Module):
    "A decoder block comprising N layers with masking."

    def __init__(
        self,
        d_model: int,
        num_head: int,
        num_layers: int,
        ff_dim: int,
        dropout_prob: float,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DecoderLayer(d_model, num_head, ff_dim, dropout_prob))

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ----------------------------------------------------------------
#  _____                     __
# |_   _| __ __ _ _ __  ___ / _| ___  _ __ _ __ ___   ___ _ __
#   | || '__/ _` | '_ \/ __| |_ / _ \| '__| '_ ` _ \ / _ \ '__|
#   | || | | (_| | | | \__ \  _| (_) | |  | | | | | |  __/ |
#   |_||_|  \__,_|_| |_|___/_|  \___/|_|  |_| |_| |_|\___|_|
# ----------------------------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout_prob: float, max_len: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)  # type: ignore
        return self.dropout(x)


class ScaledEmbedding(nn.Module):
    "Embedding layer scaled by sqrt(d_model)."

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.sqrt_d_model = sqrt(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * self.sqrt_d_model


class PositionalEmbedding(nn.Module):
    "Scaled embedding followed by a positional encoding."

    def __init__(
        self, vocab_size: int, d_model: int, dropout_prob: float, max_len: int
    ):
        super().__init__()
        self.embed = ScaledEmbedding(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, dropout_prob, max_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.embed(x))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax(self.linear(x), dim=-1)


class Transformer(nn.Module):
    "An encoder-decoder transformer architecture."

    def __init__(
        self,
        src_embed: nn.Module,
        encoder: Encoder,
        tgt_embed: nn.Module,
        decoder: Decoder,
        generator: Generator,
    ):
        super().__init__()
        self.src_embed = src_embed
        self.encoder = encoder
        self.tgt_embed = tgt_embed
        self.decoder = decoder
        self.generator = generator

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        "Take in and process masked src and tgt sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_embed(src), src_mask)

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    num_head: int = 8,
    num_layers: int = 6,
    d_feedforward: int = 2048,
    dropout_prob: float = 0.1,
    max_len: int = 5000,
):
    model = Transformer(
        PositionalEmbedding(src_vocab_size, d_model, dropout_prob, max_len),
        Encoder(d_model, num_head, num_layers, d_feedforward, dropout_prob),
        PositionalEmbedding(tgt_vocab_size, d_model, dropout_prob, max_len),
        Decoder(d_model, num_head, num_layers, d_feedforward, dropout_prob),
        Generator(d_model, tgt_vocab_size),
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# ----------------------------------------------------------------
#  ____
# |  _ \  ___ _ __ ___   ___
# | | | |/ _ \ '_ ` _ \ / _ \
# | |_| |  __/ | | | | | (_) |
# |____/ \___|_| |_| |_|\___/
# ----------------------------------------------------------------


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


if __name__ == "__main__":
    test_model = make_transformer(src_vocab_size=11, tgt_vocab_size=11)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)
