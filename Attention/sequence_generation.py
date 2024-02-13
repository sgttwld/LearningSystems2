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

# plot settings
sns.reset_defaults()
sns.set(
    rc={
        'figure.figsize': (12,12),
        'axes.titlesize': 10,
        'axes.labelsize': 10,
        'legend.fontsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        # 'figure.autolayout': True,
    }, 
    style="white" 
)
device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# if torch.backends.mps.is_available():
#     device = 'mps'

torch.manual_seed(2147483647)
st.title('LS2: Learning with Attention - Sequence generation')
"""
## 
"""
st.write("""
        *Note: what follows is greatly inspired by Andrej Karpathys makemore 
        project:* https://github.com/karpathy/makemore
        """)


st.write("## Bigram models for word generation ")

show_wordgen = st.checkbox("Show word generation")

if show_wordgen:

    words = open('names.txt', 'r').read().splitlines()
    vocab = ['.'] + sorted(list(set(''.join(words))))
    vocab_size = len(vocab)
    str2ind = {s:i for i,s in enumerate(vocab)}

    st.write(f"""
        ### Dataset
        * **data**: list of {len(words)} words
        * **examples:** {', '.join(words[1030:1040])}
        * **vocabulary**: {''.join(vocab)}
        * **str2ind**: {str2ind}
    """)


######################################################################################################
################################ Bigram counting model ###############################################
######################################################################################################
    st.write("### Bigram counting model")

    # create count matrix
    counts = torch.zeros((vocab_size,vocab_size), dtype=torch.int32)
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ind1 = str2ind[ch1]
            ind2 = str2ind[ch2]
            counts[ind1,ind2] += 1

    show_bcm = st.checkbox("Show bigram counting model")
    if show_bcm:

        st.write("""
            * Here, we are iterating over all words in the dataset, extracting the bigrams and populating a
                matrix recording the count of each bigram. 
            * we use a '.' character to indicate the start and end of a word
            * Example: 'ava' results in the four bigrams '.a', 'av', 'va', 'a.' 
            * We can then use the rows of the count matrix to calculate the empirical probabilities $p_\\textrm{emp}(X_{i+1}|X_i=x_i)$, 
                where each row corresponds to a different value of $x_i$. 
            * Sampling a word then corresponds to starting at the first row (the probability distribution over characters given '.' 
                is the first character), sampling a character, moving to the row of the sampled character and sampling another
                character from that probability distribution, etc.
            """)

         
        # visualize the matrix
        show_count_matrix = st.expander("Count matrix")
        with show_count_matrix:
            fig, ax = plt.subplots(1,1)
            ax.imshow(counts, cmap="Blues")
            for i in range(vocab_size):
                for j in range(vocab_size):
                    bigram = vocab[i] + vocab[j]
                    ax.text(j,i, bigram, ha="center", va="bottom", color="gray", fontsize=9)
                    ax.text(j,i, counts[i,j].item(), ha="center", va="top", color="gray", fontsize=9)
            ax.set_axis_off()
            st.pyplot(fig)

        # create p_emp(X_{i+1}|X_i) out of the bigram counts
        p_emp = (counts+1).float()      # add 1 to all counts for smoothing (to have no 0 probs for log-likelihood)
        p_emp = p_emp / p_emp.sum(1, keepdims=True)

        # calculate log-likelihood:
        ll = 0
        n = 0
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ind1 = str2ind[ch1]
                ind2 = str2ind[ch2]
                ll += torch.log(p_emp[ind1,ind2])
                n+=1
        st.write("* log-likelihood:", -ll/n)


        # turn p_emp into simple generative model:
        def generate_sample(probabilities, g):
            out = []
            ind = 0
            while True:
                p = probabilities[ind]
                ind = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(vocab[ind])
                if ind == 0:
                    break
            return out[:-1]
        

        g = torch.Generator()

        show_samples_counts = st.expander("Generation: Counting model")
        with show_samples_counts:
            for i in range(5):
                st.write(''.join(generate_sample(p_emp, g)))
            

        show_samples_unif = st.expander("Generation: Uniform probabilities")
        with show_samples_unif:
            unif = torch.ones((vocab_size, vocab_size))/vocab_size
            for i in range(5):
                st.write(''.join(generate_sample(unif, g)))
        



######################################################################################################
##################################   Bigram nn model   ###############################################
######################################################################################################

    st.write("### Bigram neural network model")
    # single character input -> produce distribution over next characters as output
    # i.e. P(Xn+1|Xn=x) = f(x)

    show_bnn = st.checkbox("Show bigram neural network model")
    if show_bnn:


        # compile training dataset of bigrams 
        inputs, labels = [], []
        for w in words:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                ind1 = str2ind[ch1]
                ind2 = str2ind[ch2]
                inputs.append(ind1)
                labels.append(ind2)
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        inputs_enc = F.one_hot(inputs, num_classes=vocab_size).float()
        labels_enc = F.one_hot(labels, num_classes=vocab_size).float()

        st.write(f"""
            **Training dataset**
            * We use each bigram as a tuple (input, label)
            * before we had {len(words)} words, resulting in {counts.sum()} bigrams
            * now we have {len(inputs)} inputs and {len(labels)} corresponding labels
            * one-hot encoded, this results in input and label datasets each of shape {inputs_enc.shape}
            """) 


        # small example
        i_ava = 0
        for w in words[:2]:
            chs = ['.'] + list(w) + ['.']
            for ch1, ch2 in zip(chs, chs[1:]):
                i_ava += 1

        show_example_dataset = st.expander("Example (data for 'ava')")
        with show_example_dataset:
            st.write(f"""
                * the word '{words[2]}' results in 4 bigrams '.a', 'av', 'va', 'a.'
                * this translates to input tokens {inputs[i_ava:i_ava+4]} and output tokens {labels[i_ava:i_ava+4]}
                * the corresponding one-hot encoded input vectors are:
                """)
            st.write(inputs_enc[i_ava:i_ava+4].numpy())
            st.write("""
                * and the corresponding one-hot encoded output vectors:

                """)
            st.write(labels_enc[i_ava:i_ava+4].numpy())


        # neural network
        st.write(""" 
            **Neural Network** 
            * the input is the one-hot encoded index of the current token
            * it is linearly mapped to 27 outputs using a 27x27-dimensional weight matrix $W$, 
            * these 27 outputs are used as logits for a softmax, representing the probs over the next token
            * we train the network using max log-likelihood
            
            in other words: a one-layer fully connected network of 27 neurons with 27 inputs each, softmax activation, 
            trained using cross entropy loss (equivalent to max log-likelihood).
            """)


        ## init weights
        W = torch.randn((vocab_size,vocab_size), requires_grad=True)

        show_example_loss = st.expander("Example (loss for 'ava')")
        with show_example_loss:
            # Example: batch consisting of the bigrams of 'ava'
            batchsize = 4
            curr_ind = i_ava
            xs_1h = inputs_enc[curr_ind:curr_ind+batchsize]
            ys_1h = labels_enc[curr_ind:curr_ind+batchsize]
            xs = inputs[curr_ind:curr_ind+batchsize]
            ys = labels[curr_ind:curr_ind+batchsize]

            # forward pass of this input
            logits = xs_1h @ W    # (num_data, 27) @ (27, 27) -> (num_data, 27)
            counts_nn = logits.exp()      
            probs = counts_nn / counts_nn.sum(1, keepdims = True)     # softmax of logits
            loss_ll = -probs[torch.arange(batchsize),ys].log().mean()

            st.write(f"""
                * Calculating the loss (neg. log-likelihood) for the batch consisting of the bigrams of 'ava':
                    - inputs: {xs}
                    - labels: {ys}
                    - p = softmax( inputs @ W ): shape = {probs.shape} (distribution over the next token for each current token)
                    - probabilities that should to be high given these inputs and labels:
                    p[0,1], p[1,22], p[2,1], p[3,0] = {probs[0,1].data, probs[1,22].data, probs[2,1].data, probs[3,0].data}
                    - using pytorch indexing: p[tensor([0,1,2,3]),labels] = {probs[torch.arange(4),ys].data}
                    - neg-log-likelihood = `-probs[torch.arange(4),ys].log().mean()` = {loss_ll}
                """)
            st.write("""
                * alternatively, we can use the cross entropy loss, either, the one provided by pytorch, or calculated manually:
                    $\\frac{1}{N}\\sum_{i=1}^N\\sum_c \\delta_{c,y_i} \\ p_W(c|x_i)$ = `-(ys_1h*probs.log()).sum(1).mean()`
                """)
            st.write(f"""
                * for 'ava':
                    - crossentropy(labels_1hot,logP) = {-(ys_1h*probs.log()).sum(1).mean()}
                    - torch.nn.CrossEntropyLoss(logits,labels) = {torch.nn.CrossEntropyLoss()(logits,ys)}
                """)

        epochs = 1000 
        lmbda = 70  # learning rate

        progressbar0 = st.progress(0)
        placeholder0 = st.empty()
        # for this simple example: batchsize = full dataset
        for i in range(epochs):
            # forward pass
            logits = inputs_enc @ W    # (num_data, 27) @ (27, 27) -> (num_data, 27)
            its = logits.exp()
            probs = its / its.sum(1, keepdims = True)     # softmax of logits
            loss = -probs[torch.arange(len(labels)),labels].log().mean()
            # backward pass
            W.grad = None
            loss.backward()
            # gradient descent
            W.data += -lmbda * W.grad
            progressbar0.progress((i+1)/epochs)
            placeholder0.text('step: {}'.format(i) + ' loss: {:.3f}'.format(loss.item()))


        def generate_sample_from_nn(W):
            out = []
            ind = 0
            while True:
                xenc = F.one_hot(torch.tensor([ind]), num_classes=vocab_size).float()   # one-hot rep of the current character
                logits = xenc @ W                                                      # calculate logits
                p = F.softmax(logits, dim=1)                                           # calculate probabilities of the next character
                ind = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
                out.append(vocab[ind])
                if ind == 0:
                    break
            return out[:-1]

        show_sampling_from_nn = st.expander('Generation (trained to loss {})'.format(loss.item()))
        with show_sampling_from_nn:
            g = torch.Generator()
            for i in range(5):
                st.write(''.join(generate_sample_from_nn(W)))
            


######################################################################################################
############################# Continuous data streams #############################
######################################################################################################

st.write("## From bigram models to transformers")


show_textgen = st.checkbox("Show text generation", key='showtextgen')

if show_textgen:

    st.write("""
        We have one big chunk of text for training and we want to train models that can generate similar chunks,
        instead of a given dataset of sequences (e.g. words) as above.
        """)

    ## preparing the data as one big chuck of text (as in GPT, etc):

    st.write("### Data preparation")


    text = open('feynman.txt', 'r').read() 

    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)

    st.write("**example dataset**:", "Feynman lectures v1 ch1-ch30")
    st.write("**dataset size**:", len(text))
    st.write("**vocab size**:", len(vocab))
    st.write("**vocab**:", ''.join(vocab))

    show_example_text = st.expander("Show example from dataset")
    with show_example_text:
        start = 400000
        st.write(text[start:start+10000])

    str2ind = {s:i for i,s in enumerate(vocab)}
    encode = lambda s: [str2ind[c] for c in s]
    decode = lambda l: ''.join([vocab[i] for i in l])
    full_data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(full_data))
    train_data = full_data[:n]
    val_data = full_data[n:]

    show_example_encodedecode = st.expander("Tokenize and detokenize example")
    with show_example_encodedecode:
        st.write('**text**:', text[:120])
        st.write('**vocab**:')
        st.text(vocab)
        st.write('**encoded**:', full_data[:120])
        st.write('**decoded**:', decode(full_data[:120]))

    st.write("""
        * **block size/maximum context length**: maximum length of a chunk of input data (a token sequence) used to 
            predict the next token
        * **time dimension**: for each chunk of data (of block size length) we obtain many training examples (block size many), 
            depending on how far we are into the sequence (from left to right)
        """)


    show_example_blocksize = st.expander("Block size and time dimension example")
    with show_example_blocksize:
        block_size = 8
        x = train_data[:block_size]
        y = train_data[1:block_size+1]

        st.write(f"e.g. for block size 8, the chunk of data {x.numpy()}, translates into the training examples:")
        for t in range(block_size):
            context = x[:t+1]
            target = y[t]
            st.write(f'when input is {context.numpy()} the target is {target}')


    def get_batch(split, batch_size, block_size):
        data = train_data if split == 'train' else val_data
        start_idx = torch.randint(len(data) - block_size, (batch_size,))    # sample random positions to grab chunks from the data
        x = torch.stack([data[i:i+block_size] for i in start_idx])          # get one chunk of block_size length for each batch
        y = torch.stack([data[i+1:i+1+block_size] for i in start_idx])  
        return x.to(device), y.to(device)

    st.write("""
        * **batches**: independent training examples, consisting of sampled chunks of input data of block size length, together 
            with the right-shifted outputs corresponding to the targets when moving through the input from left to right '
            (time dimension).
        """)

    show_example_batch = st.expander(f"Example batch (block_size=8, batch_size=4)")
    with show_example_batch:
        xb, yb = get_batch('train', batch_size=4, block_size=8)
        st.write("inputs:", xb.cpu().numpy())
        st.write("outputs:", yb.cpu().numpy())




    st.write("### Bigram text generation model")


    def get_params(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params    


    def train(model, batch_size, lr, max_iters, block_size, name, **kwargs):
        progressbar = st.progress(0)
        placeholder = st.empty()
        optimizer = torch.optim.AdamW(m.parameters(), lr=lr)
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
            progressbar.progress((step+1)/max_iters)
            placeholder.text('step: {}'.format(step) + ' loss: {:.3f}'.format(loss.item()))
        return model

    @torch.no_grad()
    def estimate_loss(model, eval_iters, batch_size, block_size, **kwargs):
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X,Y = get_batch(split, batch_size, block_size)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out




    st.write("Here, we define the bigram model in a more general way than necessary, allowing for an easy adaptation to more complex models.")


    class BigramLanguageModel(torch.nn.Module):
        
        def __init__(self, vocab_size):
            super().__init__()
            # embedding(num,dim) creates a learnable matrix containing `num` embedding vectors of size `dim`. input to the model is a tensor of indices that retrieves the corresponding embeddings
            self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)


        def forward(self, idx, targets=None):
            # idx is shape (B,T) 
            logits = self.token_embedding_table(idx)        # (B,T,C) = batch, time, classes/vocab_size
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

        def generate(self, idx, max_new_tokens, log=False):
            for i in range(max_new_tokens):
                if log and i < 10: st.write('step', i)
                logits, loss = self(idx)       # logits = (1,T,C)
                if log and i < 10: 
                    st.write('input:', idx, '`-->` model `-->` sample')
                    st.write('model(input).shape:', logits.shape)
                # focus only on the last timestep (bigram model!)
                logits = logits[:, -1, :]      # becomes (1,C)
                probs = F.softmax(logits, dim=-1)      # (1,C)
                idx_next = torch.multinomial(probs, num_samples=1)  
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            return idx.to(device)



    st.write("""
        * **embedding layer:** The weight matrix $W$ is created by using `torch.nn.Embedding`
        * **forward method:** inputs are the previously generated indices that are fed 
            into the embedding layer
        * **training:** cross entropy loss optimized using AdamW
        """)


    m = BigramLanguageModel(vocab_size).to(device)

    show_example_generation = st.expander("Generation (untrained)")
    with show_example_generation:
        idx = m.generate(
            idx=torch.zeros((1,1), dtype=torch.long).to(device),
            max_new_tokens=1000, log=True)[0].tolist()
        st.write(decode(idx))


    # hyperparameters
    config = {
        'name': 'bigram model',
        'vocab_size': vocab_size,
        'block_size': 8,
        'batch_size': 32,
        'max_iters': 10000,
        'lr': 1e-3,
    }

    show_bigram_lang_model = st.expander("**Bigram text generation model**" + ", {} params".format(get_params(m))) 
    with show_bigram_lang_model:
        st.write(config)
        train(m, **config)
        st.write('*Losses:*')    
        losses = estimate_loss(m, eval_iters=500, **config)
        st.write(losses)
        print(losses)
        st.write('*Generation:*')
        idx = m.generate(
            idx=torch.zeros((1,1), dtype=torch.long).to(device),
            max_new_tokens=1000, log=False)[0].tolist()
        st.write(decode(idx))


    st.write("### Transformer text generation models")

    show_example_head = st.expander(f"Example self-attention head (head_size = 16)")
    with show_example_head:
        B,T,C = 4,8,32 # batch, time, n_embed
        head_size = 16
        x = torch.randn(B,T,C)
        key = torch.nn.Linear(C, head_size, bias=False)
        query = torch.nn.Linear(C, head_size, bias=False)
        value = torch.nn.Linear(C, head_size, bias=False)
        q, k, v = query(x), key(x), value(x)  # (B, T, 16)
        st.write('x ({})'.format(x.shape), '----QUERIES--->', 'q ({})'.format(q.shape))  
        st.write('x ({})'.format(x.shape), '-----KEYS----->', 'k ({})'.format(k.shape))     
        st.write('x ({})'.format(x.shape), '-----VALUES--->', 'v ({})'.format(v.shape))  
        att = q @ k.transpose(-2,-1) # (B,T,16) @ (B, 16, T) ---> (B,T,T)
        st.write('attention scores = q @ k.T', att.shape)
        tril = torch.tril(torch.ones(T,T))
        st.write('mask:')
        st.code(tril)
        att = att.masked_fill(tril == 0, float('-inf'))
        st.write('masked attention scores (first sequence in the batch): att[0] =')
        st.code(att[0])
        probs = F.softmax(att,dim=-1)
        st.write('softmax of attention scores (full batch): probs ({}) ='.format(probs.shape))
        st.code(probs)
        out = probs @ v
        st.write('output = probs @ v ({})'.format(out.shape))


    
    class Head(nn.Module):
        """ one head of masked self-attention """

        def __init__(self, block_size, n_embed, head_size):
            super().__init__()
            self.query = nn.Linear(n_embed, head_size, bias=False)
            self.key = nn.Linear(n_embed, head_size, bias=False)
            self.value = nn.Linear(n_embed, head_size, bias=False)
            self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        def forward(self, x):
            B, T, C = x.shape
            k = self.key(x)
            q = self.query(x)
            v = self.value(x)
            att = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,16) @ (B, 16, T) ---> (B,T,T)
            att = att.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
            probs = F.softmax(att, dim=-1)
            out = probs @ v
            return out


    class MultiHeadAttention(nn.Module):
        """ multiple heads of self-attention in parallel """

        def __init__(self, num_heads, block_size, n_embed, head_size):
            super().__init__()
            self.heads = nn.ModuleList([Head(block_size, n_embed, head_size) for _ in range(num_heads)])

        def forward(self, x):
            out = torch.cat([h(x) for h in self.heads], dim=-1)
            return out


    class FeedForward(nn.Module):
        """ position-wise feed forward """

        def __init__(self, n_embed):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_embed, 4*n_embed),
                nn.ReLU(),
                nn.Linear(4*n_embed, n_embed),
                )

        def forward(self, x):
            return self.net(x)


    class Block(nn.Module):
        """ Transformer block: attention followed by position-wise ff """

        def __init__(self, block_size, n_embed, num_heads, use_ff, skip):
            super().__init__()
            self.use_ff = use_ff
            self.skip = skip
            head_size = n_embed // num_heads
            self.sa = MultiHeadAttention(num_heads, block_size, n_embed, head_size)
            if self.use_ff:
                self.ffwd = FeedForward(n_embed)

        def forward(self, x):
            x = x + self.sa(x) if self.skip else self.sa(x)
            if self.use_ff:
                x = x + self.ffwd(x) if self.skip else self.ffwd(x)
            return x


    class TransformerLanguageModel(torch.nn.Module):
            
            def __init__(self, vocab_size, n_embed, block_size, num_heads, num_blocks, 
                use_ff=False, skip=False, **kwargs):
                super().__init__()
                self.block_size = block_size
                self.num_blocks = num_blocks
                self.token_embedding_table = torch.nn.Embedding(vocab_size, n_embed)
                self.position_embedding_table = torch.nn.Embedding(block_size, n_embed)
                self.lm_head = nn.Linear(n_embed, vocab_size)
                sa_layers = [Block(block_size, n_embed, num_heads, use_ff, skip) for _ in range(num_blocks)]
                self.blocks = nn.Sequential(*sa_layers)
                
            def forward(self, idx, targets=None):
                B, T = idx.shape
                token_emb = self.token_embedding_table(idx) # (B,T,C)
                pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
                x = token_emb + pos_emb # (B,T,C)
                if self.num_blocks > 0:
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
                for i in range(max_new_tokens):
                    # crop idx to the last block_size tokens
                    idx_cond = idx[:, -self.block_size:]
                    logits, loss = self(idx_cond) 
                    # focus only on the last timestep
                    logits = logits[:, -1, :]           # (B,C)
                    probs = F.softmax(logits, dim=-1)   # (B, C)
                    idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
                    idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
                return idx.to(device)


    def run(m, config):
        show = st.expander("**{}**".format(config['name']+", {} params".format(get_params(m))))
        with show:
            st.write(config)
            train(m, **config)
            st.write('*Losses:*')
            losses = estimate_loss(m, eval_iters=500, **config)
            st.write(losses)
            print(losses)
            st.write('*Generation:*')
            idx = m.generate(
                idx=torch.zeros((1,1), dtype=torch.long).to(device),
                max_new_tokens=1000)[0].tolist()
            st.write(decode(idx))
            

    ##############################################################################
    ##############################################################################
    
    config = {
        'name': 'non-transformer model (no attention, no FF)',
        'vocab_size': vocab_size,
        'batch_size': 32,
        'block_size': 8,
        'n_embed': 32,
        'num_blocks': 0,
        'num_heads': 0,
        'max_iters': 10000,
        'lr': 1e-3,
    }
    m = TransformerLanguageModel(**config).to(device)
    run(m, config)

    ##############################################################################
    ##############################################################################

    config = {
        'name': 'transformer model (1 block, 1 head, no FF, no skip)',
        'vocab_size': vocab_size,
        'batch_size': 32,
        'block_size': 8,
        'n_embed': 32,
        'num_blocks': 1,
        'num_heads': 1,
        'max_iters': 10000,
        'lr': 1e-3,
    }
    m = TransformerLanguageModel(**config).to(device)
    run(m, config)


    ##############################################################################
    ##############################################################################

    config = {
        'name': 'transformer model (1 block, 4 heads, no FF, no skip)',
        'vocab_size': vocab_size,
        'batch_size': 32,
        'block_size': 8,
        'n_embed': 32,
        'num_blocks': 1,
        'num_heads': 4,
        'max_iters': 10000,
        'lr': 1e-3,
    }
    m = TransformerLanguageModel(**config).to(device)
    run(m, config)


    ##############################################################################
    ##############################################################################
    
    config = {
        'name': 'transformer model (1 block, 4 heads, with FF, no skip)',
        'vocab_size': vocab_size,
        'batch_size': 32,
        'block_size': 8,
        'n_embed': 32,
        'num_blocks': 1,
        'num_heads': 4,
        'use_ff': True,
        'max_iters': 10000,
        'lr': 1e-3,
    }
    m = TransformerLanguageModel(**config).to(device)
    run(m, config)



    ##############################################################################
    ##############################################################################
    
    config = {
        'name': 'transformer model (3 blocks, 4 heads, with FF, no skip)',
        'vocab_size': vocab_size,
        'batch_size': 32,
        'block_size': 8,
        'n_embed': 32,
        'num_blocks': 3,
        'num_heads': 4,
        'use_ff': True,
        'max_iters': 10000,
        'lr': 1e-3,
    }
    m = TransformerLanguageModel(**config).to(device)
    run(m, config)


    ##############################################################################
    ##############################################################################
    
    config = {
        'name': 'transformer model (3 blocks, 4 heads, with FF, with skip)',
        'vocab_size': vocab_size,
        'batch_size': 32,
        'block_size': 8,
        'n_embed': 32,
        'num_blocks': 3,
        'num_heads': 4,
        'use_ff': True,
        'skip': True,
        'max_iters': 10000,
        'lr': 1e-3,
    }
    m = TransformerLanguageModel(**config).to(device)
    run(m, config)





