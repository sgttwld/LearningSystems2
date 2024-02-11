from os import environ
environ['CUDA_VISIBLE_DEVICES'] = '0'
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from uniplot import plot
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# if torch.backends.mps.is_available():
#     device = 'mps'

torch.manual_seed(11235)

st.title('LS2: Learning with Attention - large transformer model')

st.write("""
    Here, we have added 
    * a linear projection layer to **merge the SA heads**
    * **layer norm**
    * **dropout**
     
    and scaled up the model.
    """)


text = open('feynman.txt', 'r').read() 
vocab = sorted(list(set(text)))
vocab_size = len(vocab)
str2ind = {s:i for i,s in enumerate(vocab)}
encode = lambda s: [str2ind[c] for c in s]
decode = lambda l: ''.join([vocab[i] for i in l])
full_data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]


def get_batch(split, batch_size, block_size):
    data = train_data if split == 'train' else val_data
    start_idx = torch.randint(len(data) - block_size, (batch_size,))    # sample random positions to grab chunks from the data
    x = torch.stack([data[i:i+block_size] for i in start_idx])          # get one chunk of block_size length for each batch
    y = torch.stack([data[i+1:i+1+block_size] for i in start_idx])  
    return x.to(device), y.to(device)


def train(model, batch_size, lr, max_iters, block_size, name, **kwargs):
    progressbar = st.progress(0, text=name)
    placeholder = st.empty()
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
    losses = []
    for step in tqdm(range(max_iters)):
        # sample a batch
        xb, yb = get_batch('train', batch_size=batch_size, block_size=block_size)
        # evaluate the loss
        logits, loss = model(xb,yb)
        # set gradients to 0 
        optimizer.zero_grad(set_to_none=True)
        # calculate gradients
        loss.backward()
        # 'gradient descent' step
        optimizer.step()
        progressbar.progress((step+1)/max_iters, text=name)
        placeholder.text('step: {}'.format(step) + ' loss: {:.3f}'.format(loss.item()))
        losses.append(loss.item())
    plot(losses)


def get_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params    



@torch.no_grad()
def estimate_loss(model, eval_iters, batch_size, block_size, **kwargs):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        progressbar_loss = st.progress(0, text="estimating loss ({})".format(split))
        for k in tqdm(range(eval_iters)):
            X,Y = get_batch(split, batch_size, block_size)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
            progressbar_loss.progress((k+1)/eval_iters, text="estimating loss ({})".format(split))
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """ one head of masked self-attention """

    def __init__(self, block_size, n_embed, head_size, dropout):
        super().__init__()
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        att = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,16) @ (B, 16, T) ---> (B,T,T)
        att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        probs = F.softmax(att, dim=-1)
        probs = self.dropout(probs)
        out = probs @ v
        return out



class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, block_size, n_embed, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(block_size, n_embed, head_size, dropout) for _ in range(num_heads)])
        self.merge = nn.Linear(n_embed,n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.merge(out))
        return out


class FeedForward(nn.Module):
    """ position-wise feed forward """

    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout)
            )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """ Transformer block: attention followed by position-wise ff """

    def __init__(self, block_size, n_embed, num_heads, dropout):
        super().__init__()
        head_size = n_embed // num_heads
        self.sa = MultiHeadAttention(num_heads, block_size, n_embed, head_size, dropout)
        self.ffwd = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



class TransformerLanguageModel(torch.nn.Module):
        
        def __init__(self, vocab_size, n_embed, block_size, num_heads, num_blocks, dropout, **kwargs):
            super().__init__()
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
            self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
            self.lm_head = nn.Linear(n_embed, vocab_size)
            sa_layers = [Block(block_size, n_embed, num_heads, dropout) for _ in range(num_blocks)]
            sa_layers += [nn.LayerNorm(n_embed)]
            self.blocks = nn.Sequential(*sa_layers)
            
        def forward(self, idx, targets=None):
            B, T = idx.shape
            token_emb = self.token_embedding_table(idx) # (B,T,C)
            pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
            x = token_emb + pos_emb # (B,T,C)
            x = self.blocks(x)
            logits = self.lm_head(x) # (B,T,vocab_size)
            # calculate loss
            if targets is None:
                loss = None
            else:
                # we have to merge the batch and time dimension to be able to
                # calculate the cross entropy loss over the batch
                B,T,C = logits.shape                         
                logits = logits.view(B*T, C)                
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
            return logits, loss

        def generate(self, idx, max_new_tokens):
            progressbar_gen = st.progress(0, text='Generating...')
            for i in tqdm(range(max_new_tokens)):
                # crop idx to the last block_size tokens
                idx_cond = idx[:, -self.block_size:]
                logits, loss = self(idx_cond) 
                # focus only on the last timestep
                logits = logits[:, -1, :]           # (B,C)
                probs = F.softmax(logits, dim=-1)   # (B, C)
                idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                progressbar_gen.progress((i+1)/max_new_tokens, text='Generating...')
            progressbar_gen.progress((i+1)/max_new_tokens, text='Stopped generating.')
            return idx.to(device)



##############################################################################
##############################################################################

n_embed = st.selectbox('Embedding dimension', (96,192,384,768))


config = {
    'name': 'large transformer model',
    'vocab_size': vocab_size,
    'batch_size': 64,
    'block_size': 256,
    'n_embed': n_embed,
    'num_blocks': 6,
    'num_heads': 6,
    'dropout': 0.35,
    'max_iters': 10000,
    'eval_iters': 500,
    'lr': 3e-4,
}
st.write(config)
m = TransformerLanguageModel(**config).to(device)
st.write("number of params:", get_params(m))


checkbox_train = st.checkbox("Train model")
if checkbox_train:
    train(m, **config)
    torch.save(m.state_dict(),'saved/largetransformer_{}.zip'.format(n_embed))
    st.write('Model saved.')

checkbox_load = st.checkbox("Load model")
if checkbox_load:
    m.load_state_dict(torch.load('saved/largetransformer_{}.zip'.format(n_embed)))
    st.write('Model loaded.')
    st.write('**Losses:**')
    losses = estimate_loss(m, **config)
    st.write(losses)
    print(losses)
    st.write('**Generation:**')
    idx = m.generate(idx=torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=5000)[0].tolist()
    text = decode(idx)
    st.write(text)
    print(text)




