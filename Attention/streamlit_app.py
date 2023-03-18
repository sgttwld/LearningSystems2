import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F

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

SEED = 2147483647

st.title('LS2: Learning with Attention - Sequence generation')
"""
## 
"""



st.write("## Bigram models for word generation")
st.write("""
        *Note: this section is greatly inspired by Andrej Karpathys makemore 
        project:* https://github.com/karpathy/makemore
        """)


words = open('names.txt', 'r').read().splitlines()
vocab = ['.'] + sorted(list(set(''.join(words))))
vocab_size = len(vocab)
str2ind = {s:i for i,s in enumerate(vocab)}

st.write(f"""
    ### Dataset
    * **data**: list of {len(words)} words
    * **examples:** {', '.join(words[1030:1040])}
    * **vocabulary**: {''.join(vocab)}
""")


######################################################################################################
################################ Bigram counting model ###############################################
######################################################################################################


st.write("### Bigram counting model")



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

# populate count matrix
counts = torch.zeros((vocab_size,vocab_size), dtype=torch.int32)
for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ind1 = str2ind[ch1]
        ind2 = str2ind[ch2]
        counts[ind1,ind2] += 1

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


show_samples = st.expander("Counting model samples")

def generate_sample(probabilities):
    out = []
    ind = 0
    while True:
        p = probabilities[ind]
        ind = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(vocab[ind])
        if ind == 0:
            break
    return out

with show_samples:
    g = torch.Generator().manual_seed(SEED)
    for i in range(5):
        st.write(''.join(generate_sample(p_emp)))
    st.write("**comparison to uniform probabilites**:")
    unif = torch.ones((vocab_size, vocab_size))/vocab_size
    for i in range(5):
        st.write(''.join(generate_sample(unif)))
    button = st.button("Generate")
    




######################################################################################################
##################################   Bigram nn model   ###############################################
######################################################################################################



st.write("### Bigram neural network model")
# single character input -> produce distribution over next characters as output
# i.e. P(Xn+1|Xn=x) = f(x)



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
    * one-hot encoded, this results in an input dataset of shape {inputs_enc.shape}
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
        * and the corresponding one-hot encoded input vectors are:
        """)
    st.write(inputs_enc[i_ava:i_ava+4].numpy())


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

# neural net

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
            - probabilities that should to be high given these inputs and labels:
            p[0,1], p[1,22], P[2,1], P[3,0] =, {probs[0,1].data, probs[1,22].data, probs[2,1].data, probs[3,0].data}
            - using pytorch indexing: P[tensor([0,1,2,3]),labels] = {probs[torch.arange(4),ys].data}
            - neg-log-likelihood = `-probs[torch.arange(4),ys].log().mean()` = {loss_ll}
        """)
    st.write("""
        * alternatively, we can use the cross entropy loss, either, the one provided by pytorch, or calculated manually:
            $\\frac{1}{N}\\sum_{i=1}^N\\sum_c \\delta_{c,y_i} \\ p_W(c|x_i)$ = `-(ys_1h*probs.log()).sum(1).mean()`
        """)
    st.write(f"""
        * 
            - crossentropy(labels_1hot,logP) = {-(ys_1h*probs.log()).sum(1).mean()}
            - torch.nn.CrossEntropyLoss(logits,labels) = {torch.nn.CrossEntropyLoss()(logits,ys)}
        """)

show_training = st.expander('Training')
with show_training:
    epochs = 100 
    lmbda = 70  # learning rate
    for i in range(epochs):
        # forward pass
        logits = inputs_enc @ W    # (num_data, 27) @ (27, 27) -> (num_data, 27)
        its = logits.exp()
        probs = its / its.sum(1, keepdims = True)     # softmax of logits
        loss = -probs[torch.arange(len(labels)),labels].log().mean() # + 0.01*(W**2).mean()
        st.write('loss ({} epochs)'.format(i), loss.item())
        # backward pass
        W.grad = None
        loss.backward()
        # gradient descent
        W.data += -lmbda * W.grad

show_sampling_from_nn = st.expander('Neural network samples')

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
    return out

with show_sampling_from_nn:
    g = torch.Generator().manual_seed(SEED)
    for i in range(5):
        st.write(''.join(generate_sample_from_nn(W)))
    


######################################################################################################
############################# Continuous data streams + PyTorchification #############################
######################################################################################################

st.write("## Text generation")

st.write("""
    Imagine we have one big chuck of text for training and we want to train models that can generate similar chunks,
    instead of a given dataset of sequences (e.g. words) as above.
    """)

## preparing the data as one big chuck of text (as in GPT, etc):

st.write("### Data preparation")


text = open('feynman.txt', 'r').read() 

vocab = sorted(list(set(text)))
vocab_size = len(vocab)

st.write("**example dataset**:", "Feynman lectures v1 ch2 Basic Physics")
st.write("**dataset size**:", len(text))
st.write("**vocab size**:", len(vocab))
st.write("**vocab**:", ''.join(vocab))

str2ind = {s:i for i,s in enumerate(vocab)}
encode = lambda s: [str2ind[c] for c in s]
decode = lambda l: ''.join([vocab[i] for i in l])
full_data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(full_data))
train_data = full_data[:n]
val_data = full_data[n:]

show_example_encodedecode = st.expander("Tokenize and detokenize example")
with show_example_encodedecode:
    st.write('text:', text[:120])
    st.write('encoded:', full_data[:120])
    st.write('decoded:', decode(full_data[:120]))

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
        

batch_size = 4
block_size = 8


def get_batch(split):
    data = train_data if split == 'train' else val_data
    start_idx = torch.randint(len(data) - block_size, (batch_size,))    # sample random positions to grab chunks from the data
    x = torch.stack([data[i:i+block_size] for i in start_idx])          # get one chunk of block_size length for each batch
    y = torch.stack([data[i+1:i+1+block_size] for i in start_idx])  
    return x,y

st.write("""
    * **batches**: independent training examples, consisting of sampled chunks of input data of block size length, together 
        with the right-shifted outputs corresponding to the targets when moving through the input from left to right '
        (time dimension).
    """)

show_example_batch = st.expander(f"Example batch ({block_size=}, {batch_size=})")
with show_example_batch:
    xb, yb = get_batch('train')
    st.write("inputs:", xb.numpy())
    st.write("outputs:", yb.numpy())



st.write("### Bigram text generation model")

st.write("Here, we define the bigram model in a more general way, allowing for an easy adaptation to more complex models.")


class BigramLanguageModel(torch.nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = torch.nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx is shape (B,T) 
        logits = self.token_embedding_table(idx)        # (B,T,C) = batch, time, classes/vocab_size
        # calculate loss
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape                        # we have to merge the batch and time dimension to be able to 
            logits = logits.view(B*T, C)                # calculate the cross entropy loss over the batch
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, log=False):
        for i in range(max_new_tokens):
            if log and i < 10: st.write('step', i)
            logits, loss = self(idx)                            # logits = (1,T,C)
            if log and i < 10: 
                st.write('input:', idx, '`-->` model `-->` sample')
                st.write('model(input).shape:', logits.shape)
            # focus only on the last timestep (bigram model!)
            logits = logits[:, -1, :]                           # becomes (1,C)
            probs = F.softmax(logits, dim=-1)                   # (1,C)
            idx_next = torch.multinomial(probs, num_samples=1)  
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)

st.write("""
    * **embedding layer:** The weight matrix $W$ is created by using `torch.nn.Embedding`
    * **forward method:** inputs are the previously generated indices that are fed 
        into the embedding layer
    * **training:** cross entropy loss optimized using AdamW
    """)

show_example_generation = st.expander("Generation example (untrained)")
with show_example_generation:
    idx = m.generate(
        idx=torch.zeros((1,1), dtype=torch.long),
        max_new_tokens=1000, log=True)[0].tolist()
    st.write(decode(idx))


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for step in range(10000):
    # sample a batch
    xb, yb = get_batch('train')
    # evaluate the loss
    logits, loss = m(xb,yb)
    # set gradients to 0 
    optimizer.zero_grad(set_to_none=True)
    # calculate gradients
    loss.backward()
    # 'gradient descent' step
    optimizer.step()



show_example_generation = st.expander(f"Generation example (trained to loss {loss.item()})")
with show_example_generation:
    idx = m.generate(
        idx=torch.zeros((1,1), dtype=torch.long),
        max_new_tokens=1000, log=False)[0].tolist()
    st.write(decode(idx))







