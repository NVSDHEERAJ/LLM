import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from tqdm import tqdm
import numpy as np


# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 64 # what is the maximum context length for predictions?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 128
n_head = 4
n_layer = 10
dropout = 0.2
vocab_size = 50257
# ------------

class Head(nn.Module):
    """One head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)
        v = self.value(x) # (B, T, C)

        # Compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1) # (B, T, T)
        wei = self.dropout(wei)
        # Perform the weighted aggregation of values
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """a simple linear layer followed by non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""
    # The multihead attention is the communication of data between the tokens and the feed forward is computation i.e. the tokemns thinking on the data

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection (Splitting the gradient equally in the backward direction)
        x = x + self.ffwd(self.ln2(x))
        return x
    

# LSTM block implementation from scratch (slow)
"""class RecurrentBlock(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.sig1 = nn.Sequential(nn.Linear(n_embd*2, n_embd), nn.Sigmoid())
        self.sig2 = nn.Sequential(nn.Linear(n_embd*2, n_embd), nn.Sigmoid())
        self.sig3 = nn.Sequential(nn.Linear(n_embd*2, n_embd), nn.Sigmoid())
        self.tanh1 = nn.Sequential(nn.Linear(n_embd*2, n_embd), nn.Tanh())
        self.prev_cell_state = torch.zeros((n_embd,), device = device)
        self.prev_output = torch.zeros((n_embd,), device = device)
        
    def forward(self, x):
        self.out = torch.empty_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                xt = torch.cat((self.prev_output, x[i,j]), dim = -1)
                x_s1 = self.sig1(xt)
                self.prev_cell_state = self.prev_cell_state * x_s1
                x_s2 = self.sig2(xt)
                x_tanh1 = self.tanh1(xt)
                x_s2t1 = x_s2 * x_tanh1
                self.prev_cell_state = self.prev_cell_state + x_s2t1
                x_s3 = self.sig3(xt)
                prev_cell_tanh = torch.tanh(self.prev_cell_state)
                self.prev_output = x_s3 * prev_cell_tanh
                self.out[i,j] = self.prev_output
        
        return self.out
"""        

class RecurrentTransformer(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks_1 = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer//2 - 1)])
        self.ln_p = nn.LayerNorm(n_embd) # Penultimate Layer Norm
        self.lstm_block1 = nn.LSTM(n_embd, n_embd, 1, batch_first = True)
        self.ln_lstm = nn.LayerNorm(n_embd)
        self.blocks_2 = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer//2 - 1)])
        self.ln_b2 = nn.LayerNorm(n_embd)
        self.lstm_block2 = nn.LSTM(n_embd, n_embd, 1, batch_first = True)
        self.ln_lstm2 = nn.LayerNorm(n_embd)
        self.blocks_3 = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(2)])
        #self.rec_block = RecurrentBlock()
        self.ln_f = nn.LayerNorm(n_embd) # Final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):

        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # T, C
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks_1(x) # (B, T, C)
        x = self.ln_p(x) # (B, T, C)
        #x = self.rec_block(x) #(B, T, C)
        hidden_state_1 = torch.randn((1, B, n_embd), device = device)
        cell_state_1 = torch.randn((1, B, n_embd), device = device)
        hidden_1 = (hidden_state_1, cell_state_1)
        x, hidden_1 = self.lstm_block1(x, hidden_1)
        x = self.ln_lstm(x)
        x = self.blocks_2(x)
        x = self.ln_b2(x)
        hidden_state_2 = torch.randn((1, B, n_embd), device = device)
        cell_state_2 = torch.randn((1, B, n_embd), device = device)
        hidden_2 = (hidden_state_2, cell_state_2)
        x, hidden_2 = self.lstm_block2(x, hidden_2)
        x = self.ln_lstm2(x)
        x = self.blocks_3(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            """if(idx_next[0].item() == 13):
                return idx"""
            
        return idx
