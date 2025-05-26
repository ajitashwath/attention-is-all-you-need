import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim = -1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(attention_output)
        return output
    
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tget_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self_attn_output = self.self_attention(x, x, x, tget_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))

        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
    
    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tget_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tget_mask)
        return x
    
class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size: int,
            tget_vocab_size: int,
            d_model: int = 512,
            n_heads: int = 8,
            n_layers: int = 6,
            d_ff: int = 2048,
            max_seq_length: int = 5000,
            dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tget_embedding = nn.Embedding(tget_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.encoder = TransformerEncoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.decoder = TransformerDecoder(d_model, n_heads, d_ff, n_layers, dropout)
        self.output_projection = nn.Linear(d_model, tget_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.__init_parameters()

    def __init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def create_padding_mask(self, seq: torch.Tensor, pad_idx: int = 0) -> torch.Tensor:
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None, memory_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout(src_emb)

        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)

        encoder_output = self.encoder(src_emb, src_mask)
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask=memory_mask, tget_mask=tgt_mask)
        output = self.output_projection(decoder_output)
        return output
    
    def encode(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        return self.encoder(src_emb, src_mask)
    
    def decode(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: Optional[torch.Tensor] = None, tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        decoder_output = self.decoder(tgt_emb, encoder_output, src_mask, tgt_mask)
        return self.decoder(decoder_output)
    
def create_transformer_model(src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    return Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model = 512,
        n_heads = 8,
        n_layers = 6,
        d_ff = 2048,
        max_seq_length = 5000,
        dropout = 0.1
    )

def create_big_transformer_model(src_vocab_size: int, tgt_vocab_size: int) -> Transformer:
    return Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model = 1024,
        n_heads = 16,
        n_layers = 6,
        d_ff = 4096,
        max_seq_length = 5000,
        dropout = 0.3
    )

def train_step(model: Transformer, src: torch.Tensor, tgt: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: nn.Module, pad_idx: int = 0):
    model.train()
    src_mask = model.create_padding_mask(src, pad_idx)
    tgt_input = tgt[:, :-1]
    tgt_output = tgt[:, 1:]
    
    tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1))
    tgt_padding_mask = model.create_padding_mask(tgt_input, pad_idx)
    tgt_mask = tgt_mask & tgt_padding_mask

    output = model(src, tgt_input, src_mask, tgt_mask, src_mask)
    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    src_vocab_size = 10000
    tgt_vocab_size = 10000

    model = create_transformer_model(src_vocab_size, tgt_vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print(f"Model architecture: {model}")
    
    batch_size = 32
    src_seq_len = 50
    tgt_seq_len = 50
    
    src = torch.randint(1, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(1, tgt_vocab_size, (batch_size, tgt_seq_len))
    with torch.no_grad():
        output = model(src, tgt[:, :-1])
        print(f"Output shape: {output.shape}")
        print("Model successfully created and tested!")
                