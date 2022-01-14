"""
Exemplary implementation of the most simple attention mechanism possible solving the XOR problem.
Author: Sebastian Gottwald
Date: 2022-01-14
"""


# %%
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# %%
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


class Neuron(object):
    def __init__(self,dimH):
        self.w = tf.Variable(np.random.uniform(0,1,size=(dimH)), dtype=tf.float64)
        self.b = tf.Variable(np.random.uniform(0,1), dtype=tf.float64)
        self.trainable_variables = [self.w,self.b]

    def predict(self,x):
        return tf.math.sigmoid(tf.reduce_sum(self.w*x, axis=-1)+self.b)


class ThreeNeurons(object):
    def __init__(self):
        self.p1 = Neuron(2)
        self.p2 = Neuron(2)
        self.p3 = Neuron(2)
        self.trainable_variables = self.p1.trainable_variables + self.p2.trainable_variables + self.p3.trainable_variables

    def predict(self,x):
        h = tf.concat([tf.expand_dims(self.p1.predict(x),axis=1),tf.expand_dims(self.p2.predict(x),axis=1)],axis=1)
        return self.p3.predict(h)


class SelfAttention(object):
    def __init__(self,dimS,dimH):
        self.dimS = dimS
        self.dimH = dimH
        self.Wq = tf.Variable(np.random.uniform(0,1,size=(dimH,dimH)), dtype=tf.float64)
        self.Wk = tf.Variable(np.random.uniform(0,1,size=(dimH,dimH)), dtype=tf.float64)
        self.Wv = tf.Variable(np.random.uniform(0,1,size=(dimH,dimH)), dtype=tf.float64)
        # self.trainable_variables = [self.Wq,self.Wk,self.Wv]
        self.trainable_variables = [self.Wk]

    def Q(self,inputs):
        return tf.matmul(inputs,self.Wq,transpose_b=True)

    def K(self,inputs):
        return tf.matmul(inputs,self.Wk,transpose_b=True)

    def V(self,inputs):
        return tf.matmul(inputs,self.Wv,transpose_b=True)

    def weights(self, inputs):
        # return tf.matmul(self.K(inputs),self.K(inputs),transpose_b=True)
        return tf.matmul(self.K(inputs),self.K(inputs),transpose_b=True)

    def prob(self,inputs):
        w = self.weights(inputs)
        return tf.exp(w)/tf.reduce_sum(tf.exp(w), axis=-1,keepdims=True)
        
    def output(self,inputs):
        # return tf.matmul(self.prob(inputs),self.V(inputs)) ## for this task it seems to be enough to have V = K
        return tf.matmul(self.prob(inputs),self.K(inputs))


class Attendron(object):
    def __init__(self,dimS,dimH):
        self.att = SelfAttention(dimS,dimH)
        self.Neuron = Neuron(dimH*dimS)
        self.trainable_variables = self.att.trainable_variables + self.Neuron.trainable_variables

    def predict(self,embeddedInputs):
        attentionOutput = self.att.output(embeddedInputs)
        res = embeddedInputs + attentionOutput
        NeuronInput = tf.cast(tf.keras.layers.Flatten()(res),dtype=tf.float64)
        return self.Neuron.predict(NeuronInput)

loss = tf.keras.losses.MSE
optimizer = tf.keras.optimizers.Adam(learning_rate=.01)

@tf.function 
def train(model,inputs,outputs):
    with tf.GradientTape() as g:
        obj = loss(outputs,model.predict(inputs))
        gradients = g.gradient(obj, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def f_xor(v):
    return 1.0 if not((v[0] + v[1] -1)%2) else 0.0

def f_or(v):
    return 1.0 if v[0] == 1 or v[1] == 1 else 0.0

f = f_xor

## possible input and output values
inputs = np.array([[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]])
outputs = tf.constant([f(d) for d in inputs], dtype=tf.float64)

def binaryVectorEmbedding(y):
    return (tf.expand_dims(y,axis=1)*tf.expand_dims(tf.constant(np.array([1.0,0.0]),dtype=tf.float64),axis=0) 
            + tf.expand_dims(1-y,axis=1)*tf.expand_dims(tf.constant(np.array([0.0,1.0]),dtype=tf.float64),axis=0))

def input_embedding():
    # the full input_embedding() array is like different sentences, two vectors for each sentence
    return tf.constant(np.array([binaryVectorEmbedding(inp) for inp in inputs]))


embeddedInputs = input_embedding()
model = Attendron(dimS=2,dimH=2)

for i in range(5000):
    train(model,embeddedInputs,outputs)
    print(loss(outputs,model.predict(embeddedInputs)))

# print("K:",model.att.K(embeddedInputs))
# print("weights:",model.att.weights(embeddedInputs))
# print("prob:",model.att.prob(embeddedInputs))
print("output:",model.att.output(embeddedInputs))
print("residual output:", embeddedInputs + model.att.output(embeddedInputs))


#####################################################
## MLP
## remove the comments to train an MLP with 3 neurons: 
## (you might have to change the learning rate to 0.001 if it gets stuck)
#####################################################

# model = ThreeNeurons()
# for i in range(10000):
#     train(model,inputs,outputs)
#     print(loss(outputs,model.predict(inputs)))



