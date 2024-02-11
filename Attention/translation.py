import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text

# plot settings
sns.reset_defaults()
sns.set(
    rc={
        'figure.figsize': (10,10),
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

def show_vec(x):
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.matshow(x)
    plot = st.pyplot(fig)





st.write("# Transformer: Text Translation")
st.write('Main source: https://www.tensorflow.org/text/tutorials/transformer')

#########################################################################################################
#########################################################################################################
st.write('## Preprocessing')


########################
####### DATASET ########
########################

st.write('### Dataset (pt to en)')
st.write('We are loading the dataset `ted_hrlr_translate/pt_to_en` provided by TensorFlow.')

@st.cache_resource
def load_data():
    # examples = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True)
    examples = tfds.load('ted_hrlr_translate/pt_to_en', as_supervised=True, split='train[:5%]')
    # return examples['train'], examples['validation']
    return examples

train_examples = load_data()


# show examples:
show_ex_sentences = st.checkbox("Example sentences")

if show_ex_sentences:
    for pt_examples, en_examples in train_examples.batch(1).take(4):
        for pt in pt_examples.numpy():
            st.write('**Portuguese**: ',pt.decode('utf-8'))
        for en in en_examples.numpy():
            st.write('**English**:', en.decode('utf-8'))


#######################
####### TOKENS #######
#######################

st.write('')
st.write('### Tokenizers (sentence => token sequence)')

st.write('''
    Tokenizers transform language sentences into token sequences. Which elements of a sentence are used as tokens
    depends on the model. There are word tokenizers (problems with out of vocabulary words), character tokenizers 
    (very long sequences, hard to learn the meaning of single words), and subword tokenizers, the happy medium between 
    word and character tokenizers, which are used in most current NLP models.

    Here, we load the model `ted_hrlr_translate_pt_en_converter` which contains a portuguese and an 
    english subword tokenizer.''')


@st.cache_resource
def load_tokenizers(model_name):
    tf.keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True
    )
    return tf.saved_model.load(model_name)

tokenizers = load_tokenizers(model_name = 'ted_hrlr_translate_pt_en_converter')
st.write('vocabulary size (en):', tokenizers.en.get_vocab_size())
st.write('vocabulary size (pt):', tokenizers.pt.get_vocab_size())

show_ex_tokens = st.checkbox("Example tokenized sentences")
if show_ex_tokens:

    for pt_examples, en_examples in train_examples.batch(1).take(4):
        break

    with tf.device('/cpu:0'):
      encoded_en = tokenizers.en.tokenize(en_examples)
      encoded_pt = tokenizers.pt.tokenize(pt_examples)

    for (en,row) in zip(en_examples.numpy(),encoded_en):
      st.write('**English sentence**: ', en.decode('utf-8'))
      st.write('**tokenized**:', row)

    st.write('''The method `detokenize` attempts to convert a sequence of tokens back to a sentence, even though 
        words can be tokenized into multiple tokens. We can observe this using the `lookup` method, converting 
        the token-IDs back to token text one-by-one:''')

    ### lookup (lower level mapping: token-ID -> token text)
    detokenized = tokenizers.en.detokenize(encoded_en)
    st.write('**detokenize(token sequence):**', detokenized[0])
    tokens = tokenizers.en.lookup(encoded_en)
    st.write('**lookup(token sequence)**:', tokens[0])



##############################
####### INPUT PIPELINE #######
##############################

st.write('')
st.write('### Data preparation')

st.write('''The (pt,en) sentence tuples in the dataset are transformed to token sequence tuples. 
    The English token sequences are once taken as is (for the actual targets to calculate the loss) and once 
    shifted right (as inputs to the decoder using masked self attention to enable _teacher forcing_).
    ''')


MAX_TOKENS = 128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels

BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))

# Create training and validation set batches.
train_batches = make_batches(train_examples)
# val_batches = make_batches(val_examples)

show_ex_trainingbatch = st.checkbox("Example")
if show_ex_trainingbatch:
    for (pt, en), en_labels in train_batches.take(1):
        st.write('**target** sequence:', en_labels[0][:5])
        st.write('lookup(target sequence):', tokenizers.en.lookup(en_labels)[0][:5])
        st.write('**right-shifted** decoder input:', en[0][:5])
        st.write('lookup(right-shifted):', tokenizers.en.lookup(en)[0][:5])
        break




#########################################################################################################
#########################################################################################################
st.write('')
st.write('## Building blocks')



###################################
############ EMBEDDINGS ###########
###################################

st.write('### Embeddings (token sequences => vector sequences)')
st.write("""
    The embedding layer maps tokens (integers) to vectors of size `d_model`. 
    Each token ID is one-hot encoded and passed through a linear map using a (vocab_size, d_model)-sized weight matrix.
    Then the positional encoding is added.
    """)

def positional_encoding(length, depth):
  depth = depth/2
  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)
  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)
  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 
  return tf.cast(pos_encoding, dtype=tf.float32)


class Embedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


embed_pt = Embedding(vocab_size=tokenizers.pt.get_vocab_size(), d_model=512)
embed_en = Embedding(vocab_size=tokenizers.en.get_vocab_size(), d_model=512)

@st.cache_data
def get_batch_embeds():
    for (pt, en), en_labels in train_batches.take(1):
        break
    pt_emb = embed_pt(pt)
    en_emb = embed_en(en)
    W_embed_pt = embed_pt.trainable_variables[0]
    W_embed_en = embed_en.trainable_variables[0]
    return pt, en, pt_emb, en_emb, W_embed_pt, W_embed_en 


pt, en, pt_emb, en_emb, W_embed_pt, W_embed_en = get_batch_embeds()

st.write('**trainable parameters** (weight matrices):')
st.write('* pt embedding shape: ', W_embed_pt.shape)
st.write('* en embedding shape: ', W_embed_en.shape)

show_ex_batchembed = st.checkbox("Embedding example")
if show_ex_batchembed:

    st.write("* en: (batch_size, num_tokens, d_model) =", en_emb.shape)
    st.write("* pt: (batch_size, num_tokens, d_model) =", pt_emb.shape)

    ex1 = en[0,1]
    ex2 = en[0,2]
    ex1vec = en_emb[0,1]
    ex2vec = en_emb[0,2]

    st.write("**Example token embedding 1**:")
    st.write('English token', ex1, ' = ', tokenizers.en.lookup(en)[0,1])
    st.write('embedding vector [0:10] = ', ex1vec[:10], '...')
    st.write('First 50 entries:')
    show_vec(ex1vec[:50].numpy().reshape(1,50))
    st.write("**Example token embedding 2**:")
    st.write('English token', ex1, ' = ', tokenizers.en.lookup(en)[0,2])
    st.write('embedding vector [0:10] = ', ex2vec[:10], '...')
    st.write("First 50 entries:")
    show_vec(ex2vec[:50].numpy().reshape(1,50))

    st.write('''
        There is no built-in inverse map (as for the tokens) because the embedding layer is learnable (in contrast to the 
        tokenizers, which are frozen). However we can try to get back to the token from the embedding vector by 
        pseudo-inverting ('Moore-Penrose inverse') the weight matrix and multiplying it to the embedding vector. The final
        vector can be passed through a softmax to obtain a distribution over tokens:
        ''')
    st.write()
    st.write(tf.math.argmax(tf.nn.softmax(tf.linalg.matmul(tf.expand_dims(ex1vec,0),tf.linalg.pinv(W_embed_en))),1))



#########################################
############ ATTENTION BLOCKS ###########
########################################

st.write('')
st.write('### Attention Blocks')


st.write("""
    We use the multihead query-key-value attention layer provided by TensorFlow to create 
    * (global) **self-attention** (q,k,v = encoder input embedding)
    * (causal) **masked self-attention** (q,k,v = masked decoder input embedding)
    * **cross attention** (q = decoder output, k,v = encoder output)
    """)


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            key=x,
            value=x)
        # Cache the attention scores for plotting later.
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query=x,
            key=context,
            value=context,
            return_attention_scores=True)

        # Cache the attention scores for plotting later.
        self.last_attn_scores = attn_scores
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class MultiHeadAttentionCausal(tf.keras.layers.MultiHeadAttention):
    """
    For TF 2.9 support (which is required for ARM macs right now, to 
    be able to use tf-text). 
    The new 2.10.0 release supports `use_causal_mask = True` in the
    call function of tf.keras.layers.MultiHeadAttention.
    """

    def __init__(self, num_heads, key_dim, **kwargs):
        super().__init__(num_heads, key_dim, **kwargs)
        self.supports_masking = True

    def _compute_attention_mask(self, query, value, key=None, attention_mask=None):
        use_causal_mask = True
        query_mask = getattr(query, "_keras_mask", None)
        value_mask = getattr(value, "_keras_mask", None)
        key_mask = getattr(key, "_keras_mask", None)
        auto_mask = None
        if query_mask is not None:
            # B = batch size, T = max query length
            auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
        if value_mask is not None:
            # B = batch size, S == max value length
            mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            # B == batch size, S == max key length == max value length
            mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if use_causal_mask:
            # the shape of the causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if auto_mask is not None:
            # merge attention_mask & automatic mask, to shape [B, T, S]
            attention_mask = (
                auto_mask
                if attention_mask is None
                else attention_mask & auto_mask
            )
        return attention_mask

    def _compute_causal_mask(self, query, value=None):
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )


    def call(
        self,
        query,
        value,
        key=None,
        attention_mask=None,
        return_attention_scores=False,
        training=None,
    ):
        attention_mask = self._compute_attention_mask(
            query,
            value,
            key=key,
            attention_mask=attention_mask,
        )

        if not self._built_from_signature:
            self._build_from_signature(query=query, value=value, key=key)
        if key is None:
            key = value

        query_is_ragged = isinstance(query, tf.RaggedTensor)
        if query_is_ragged:
            query_lengths = query.nested_row_lengths()
            query = query.to_tensor()

        key_is_ragged = isinstance(key, tf.RaggedTensor)
        value_is_ragged = isinstance(value, tf.RaggedTensor)
        if key_is_ragged and value_is_ragged:
            # Ensure they have the same shape.
            bounding_shape = tf.math.maximum(
                key.bounding_shape(), value.bounding_shape()
            )
            key = key.to_tensor(shape=bounding_shape)
            value = value.to_tensor(shape=bounding_shape)
        elif key_is_ragged:
            key = key.to_tensor(shape=tf.shape(value))
        elif value_is_ragged:
            value = value.to_tensor(shape=tf.shape(key))

        #   N = `num_attention_heads`
        #   H = `size_per_head`
        # `query` = [B, T, N ,H]
        query = self._query_dense(query)

        # `key` = [B, S, N, H]
        key = self._key_dense(key)

        # `value` = [B, S, N, H]
        value = self._value_dense(value)

        attention_output, attention_scores = self._compute_attention(
            query, key, value, attention_mask, training
        )
        attention_output = self._output_dense(attention_output)

        if query_is_ragged:
            attention_output = tf.RaggedTensor.from_tensor(
                attention_output, lengths=query_lengths
            )

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output



class CausalSelfAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = MultiHeadAttentionCausal(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
    
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x



show_ex_causalattention = st.checkbox("Causal attention test")
if show_ex_causalattention:

    sample_csa = CausalSelfAttention(num_heads=1, key_dim=32)
    st.write('''
        We can test the masking by comparing the result of feeding a sequence into the layer and cutting off the result 
        at a certain index and feeding the same sequence but cutting it off before feeding it into the layer. Since the 
        causal masking only allows backward connections, the results should be the same.
        
    ''')
    out1 = sample_csa(embed_en(en[:, :3])) 
    out2 = sample_csa(embed_en(en))[:, :3]
    st.write('$|f(seq[0:3])-f(seq)[0:3]| =$', tf.reduce_max(abs(out1 - out2)).numpy())



#########
###### Example: 
#########

@st.cache_data
def get_dims_crossattention(heads, key_dims, value_dims):
    dims_dict = {}
    for num_heads in heads:
        dims_dict[num_heads] = {}
        for key_dim in key_dims:
            dims_dict[num_heads][key_dim] = {}
            for value_dim in value_dims:                
                
                crossattention = CrossAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim)
                crossattention_output = crossattention(en_emb, pt_emb)

                dims_dict[num_heads][key_dim][value_dim] = {
                    'attentionscore': crossattention.last_attn_scores.shape,
                    'output': crossattention_output.shape,
                    'trainablevariables': [{'name': var.name, 'shape': var.shape} for var in crossattention.mha.trainable_variables]
                }
    return dims_dict

show_ex_crossattention = st.checkbox("Cross attention example")
if show_ex_crossattention:

    heads = [1,2,3]
    key_dims = [16,32,48]
    value_dims = [128,256,512]

    dims_dict = get_dims_crossattention(heads, key_dims, value_dims)

    st.write('**Inputs:**')
    st.write('* Query input dimensions (en): (batch_size, num_tokens, d_model) =',en_emb.shape)
    st.write('* Key/Value input dimensions (pt): (batch_size, num_tokens, d_model) =', pt_emb.shape)

    st.write('**Parameters:**')

    num_heads = st.select_slider("Number of heads", options = heads, value = heads[1])
    key_dim = st.select_slider("Key/Query dimension", options = key_dims, value = key_dims[0])
    value_dim = st.select_slider("Value dimension", options = value_dims, value = value_dims[1])

    dims = dims_dict[num_heads][key_dim][value_dim]
    st.write('* attention score dimensions:', dims['attentionscore'])
    st.write('* output dimensions:', dims['output'])


    st.write("""
        **Trainable variables:** There are 3 weight matrices (and biases) corresponding to the 3 linear transformations 
        that map the input vectors to key, query, and value vectors. In order to maintain a fixed embedding dimension 
        (here: 512), a final linear transformation is required to map the (num_heads $\\times$ value_dim)-sized
        attention output to a single vector with the embedding dimension. 
        """)
    
    for var in dims['trainablevariables']:
        if not(var['name'][-6:-2] == 'bias'):
            st.write('* ',var['name'], var['shape'])




#########################################
############# FEED FORWARD #############
########################################


st.write('')
st.write('### Feed forward network')

st.write("""
    At the end of every attention layer, there is a single feed forward network with one `dff`-dimensional hidden layer 
    taking each `d_model`-dimensional vector in the output sequences of the multihead attention blocks as input and producing 
    a new `d_model`-dimensional vector as output (mixing along the hidden dimension). Here, `d_model = 512`, `dff = 2048`.  
    """)

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x



st.write('')
st.write('## Architecture')

#########################################################################################################
#########################################################################################################
st.write('### Encoder')


st.write("""
    One encoder layer contains one self attention and one feed forward layer. The encoder consists of a stack of `N` such layers,
    where the first layer takes the output of the embedding as input.
    """)


class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x


class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
               dff, vocab_size, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = Embedding(
        vocab_size=vocab_size, d_model=d_model)

    self.enc_layers = [
        EncoderLayer(d_model=d_model,
                     num_heads=num_heads,
                     dff=dff,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)

    for i in range(self.num_layers):
        x = self.enc_layers[i](x)

    return x  # Shape `(batch_size, seq_len, d_model)`.


@st.cache_data
def encoder_example(): 
    sample_encoder = Encoder(num_layers=2,
                             d_model=512,
                             num_heads=3,
                             dff=2048,
                             vocab_size=tokenizers.pt.get_vocab_size())

    sample_encoder_output = sample_encoder(pt, training=False)
    emb = sample_encoder.embedding(pt)
    return emb, sample_encoder_output


show_ex_encoder = st.checkbox("Encoder example")
if show_ex_encoder:

    emb, sample_encoder_output = encoder_example()

    # Print the shape.
    st.write('* input batch shape:', pt.shape)
    st.write('* example encoder output shape:', sample_encoder_output.shape)  
    st.write('* example sequence:')

    L = 100
    st.write('input sentence:', tokenizers.pt.lookup(pt)[0])
    st.write('input token sequence:',pt[0])
    st.write('input vector sequence (embedding): shape =',emb[0].shape)
    show_vec(emb[0][0][:L].numpy().reshape(1,L))
    st.write('$\\vdots$')
    show_vec(emb[0][-1][:L].numpy().reshape(1,L))
    st.write('output vector sequence: shape =', sample_encoder_output[0].shape)
    show_vec(sample_encoder_output[0][0][:L].numpy().reshape(1,L))
    st.write('$\\vdots$')
    show_vec(sample_encoder_output[0][-1][:L].numpy().reshape(1,L))




#########################################################################################################
#########################################################################################################
st.write('')
st.write('### Decoder')


st.write("""
    One decoder layer contains one masked self attention, one cross attention, and one feed forward layer. The 
    decoder consists of a stack of `N` such layers.
    """)


class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               *,
               d_model,
               num_heads,
               dff,
               dropout_rate=0.1):
    super(DecoderLayer, self).__init__()

    self.causal_self_attention = CausalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.cross_attention = CrossAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x, context):
    x = self.causal_self_attention(x=x)
    x = self.cross_attention(x=x, context=context)

    # Cache the last attention scores for plotting later
    self.last_attn_scores = self.cross_attention.last_attn_scores

    x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
    return x


class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
               dropout_rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.pos_embedding = Embedding(vocab_size=vocab_size,
                                             d_model=d_model)
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.dec_layers = [
        DecoderLayer(d_model=d_model, num_heads=num_heads,
                     dff=dff, dropout_rate=dropout_rate)
        for _ in range(num_layers)]

    self.last_attn_scores = None

  def call(self, x, context):
    # `x` is token-IDs shape (batch, target_seq_len)
    x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

    x = self.dropout(x)

    for i in range(self.num_layers):
      x  = self.dec_layers[i](x, context)

    self.last_attn_scores = self.dec_layers[-1].last_attn_scores

    # The shape of x is (batch_size, target_seq_len, d_model).
    return x



#########################################################################################################
#########################################################################################################
st.write('')
st.write('### Transformer')

st.write('''For the transformer architecture we simply combine the encoder with the decoder and add a final 
    dense layer to transform each vector of the last decoder output sequence into a softmax over tokens.
    ''')

class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
               input_vocab_size, target_vocab_size, dropout_rate=0.1):
    super().__init__()
    self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=input_vocab_size,
                           dropout_rate=dropout_rate)

    self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                           num_heads=num_heads, dff=dff,
                           vocab_size=target_vocab_size,
                           dropout_rate=dropout_rate)

    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs):
    # To use a Keras model with `.fit` you must pass all your inputs in the
    # first argument.
    context, x  = inputs

    context = self.encoder(context)  # (batch_size, context_len, d_model)

    x = self.decoder(x, context)  # (batch_size, target_len, d_model)

    # Final linear layer output.
    logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

    try:
      # Drop the keras mask, so it doesn't scale the losses/metrics.
      # b/250038731
      del logits._keras_mask
    except AttributeError:
      pass

    # Return the final output and the attention weights.
    return logits



st.write('## "Tiny Transformer"')

num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1


st.write("""
    For the full transformer model (from Vaswani 2017) the large amount of parameters results in large model sizes (~ 3GB). Hence, for some quick testing, we use a much smaller
    transformer model with only 5\% of the parameters, for which we compare the performance under different number of episodes.
    Additionally we trained a medium-sized model for 200 epochs to compare the results with.
    """)    
    
st.table({"": ["tiny","medium","full"],"num_layers": [4,6,6],"d_model": [128,384,512], "dff": [512,1024,2048], "num_heads": [8,8,8], "#parameters": ["10 million", "103 million", "188 million"]})



transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=dropout_rate)




################################
########### Training ###########
################################


# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super().__init__()

#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps

#   def __call__(self, step):
#     step = tf.cast(step, dtype=tf.float32)
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)

#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


# learning_rate = CustomSchedule(d_model)
# optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
#                                      epsilon=1e-9)


## one has to apply a mask to the loss and accuracy to deal with the padded target sequences 
## as we do not want the model to predict arbitrary padding.

# def masked_loss(label, pred):
#     mask = label != 0
#     loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
#     loss = loss_object(label, pred)
#     mask = tf.cast(mask, dtype=loss.dtype)
#     loss *= mask
#     loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
#     return loss


# def masked_accuracy(label, pred):
#     pred = tf.argmax(pred, axis=2)
#     label = tf.cast(label, pred.dtype)
#     match = label == pred
#     mask = label != 0
#     match = match & mask
#     match = tf.cast(match, dtype=tf.float32)
#     mask = tf.cast(mask, dtype=tf.float32)
#     return tf.reduce_sum(match)/tf.reduce_sum(mask)


# @st.cache_resource
# def train(epochs=20):
#     transformer.compile(
#         loss=masked_loss,
#         optimizer=optimizer,
#         metrics=[masked_accuracy])
#     transformer.fit(train_batches,
#                 epochs=EPOCHS,
#                 validation_data=val_batches)


# EPOCHS = 2
# show_ex_training = st.checkbox("Train ({} epochs)".format(EPOCHS))
# if show_ex_training:
#     train(epochs = EPOCHS)




class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    # The input sentence is Portuguese, hence adding the `[START]` and `[END]` tokens.
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
        sentence = sentence[tf.newaxis]
    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()
    encoder_input = sentence
    # As the output language is English, initialize the output with the
    # English `[START]` token.
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=tf.int64,size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
        output = tf.transpose(output_array.stack())
        predictions = self.transformer([encoder_input, output], training=False)

        # Select the last token from the `seq_len` dimension.
        predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

        predicted_id = tf.argmax(predictions, axis=-1)

        # Concatenate the `predicted_id` to the output which is given to the
        # decoder as its input.
        output_array = output_array.write(i+1, predicted_id[0])

        if predicted_id == end:
            break

    output = tf.transpose(output_array.stack())
    # The output shape is `(1, tokens)`.
    text = tokenizers.en.detokenize(output)[0]  
    tokens = tokenizers.en.lookup(output)[0]
    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return text, tokens, attention_weights



def print_translation(tokens):
    st.write(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')

@st.cache_resource
def load_saved_transformer(name):
    return tf.saved_model.load(name)

def show_translations(sentence, ground_truth, transformers):
    st.write(f'{"**Input**":15s}: {sentence}')
    st.write(f'{"**Ground truth**":15s}: {ground_truth}')   
    with tf.device('/CPU:0'):
        for trans in transformers:
            st.write('* trained for {} epochs:'.format(trans["epochs"]), trans["model"](sentence)[0])


def plot_attention_head(in_tokens, translated_tokens, attention):
    # The model didn't generate `<START>` in the output. Skip it.
    translated_tokens = translated_tokens[1:]
    fig, ax = plt.subplots()

    # ax = plt.gca()
    ax.matshow(attention)
    ax.set_xticks(range(len(in_tokens)))
    ax.set_yticks(range(len(translated_tokens)))

    labels = [label.decode('utf-8') for label in in_tokens.numpy()]
    ax.set_xticklabels(labels, rotation=90)

    labels = [label.decode('utf-8') for label in translated_tokens.numpy()]
    ax.set_yticklabels(labels)
    st.pyplot(fig)

@st.cache_data
def get_attention_scores(sentence):
    scores = {}
    for transformer in transformers:
        with tf.device('/CPU:0'):
            in_tokens = tf.convert_to_tensor([sentence])
            in_tokens = tokenizers.pt.tokenize(in_tokens).to_tensor()
            in_tokens = tokenizers.pt.lookup(in_tokens)[0]
            # res1, tok1, att1 = transf1(sentence)
            res, tokens, att = transformer["model"](sentence)
        # Shape: `(batch=1, num_heads, seq_len_q, seq_len_k)`.
        attention_scores = tf.squeeze(att, 0)
        scores[transformer["epochs"]] = (in_tokens, tokens, attention_scores)
    return scores


# epochs_list = [0]
epochs_list = [0,1,2,5,9]

transformers = []
for epochs in epochs_list:
    transformers.append({
        "model": load_saved_transformer('saved/translator_{}ep'.format(epochs)),
        "epochs": epochs,
        })



show_ex_transformer = st.checkbox("Show examples")
if show_ex_transformer:
    st.write('**Example 1**')
    sentence = 'este é um problema que temos que resolver.'
    ground_truth = 'this is a problem we have to solve.'
    show_translations(sentence, ground_truth, transformers)
    st.write("* medium model:", "`this is a problem that we have to solve .`")
        
    st.write('**Example 2**')
    sentence = 'o animal não pode atravessar a rua porque é muito lento.'
    ground_truth = 'the animal cannot cross the street because it is too slow'
    show_translations(sentence, ground_truth, transformers)
    st.write("* medium model:", "`the animal can ' t crossing the street because it ' s too slow .`")

    st.write('**Example 3**')
    sentence = 'vou então muito rapidamente partilhar convosco algumas histórias de algumas coisas mágicas que aconteceram.'
    ground_truth = "so I'm going to very quickly share with you some stories of some magical things that happened."
    show_translations(sentence, ground_truth, transformers)
    st.write("* medium model:", "`so i ' m going to share with you some stories from some magic things that had happened`")


show_ex_attentionscores = st.checkbox("Show attention scores")
if show_ex_attentionscores:
    sentence = 'o animal não pode atravessar a rua porque é muito lento.'
    scores = get_attention_scores(sentence)

    epochs = st.select_slider("Select number of trained epochs", options = epochs_list, value = 0)
    head = st.select_slider("Select head", options = range(num_heads), value = 0)

    in_tokens, out_tokens, attention_scores = scores[epochs]
    plot_attention_head(in_tokens, out_tokens, attention_scores[head])




# transformer_med = load_saved_transformer('saved/translator_l6_dm384_df1024')
# sentence = 'este é um problema que temos que resolver.'

# with tf.device('/CPU:0'):
#     st.write('result:', transformer_med(sentence)[0])

