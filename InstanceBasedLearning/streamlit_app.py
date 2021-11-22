import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from qpsolvers import solve_qp,available_solvers
import time 

# plot settings
sns.reset_defaults()
sns.set(
    rc={
        'figure.figsize': (8,5),
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

@st.cache
def gen_data(num, f):
    points = np.random.sample(size=(num,2))
    labels = []
    for x,y in points:
        labels.append(1 if f(x)<y else -1)
    return points, labels

def show_data(X,Y):
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.scatter(X[:,0], X[:,1], c = Y, cmap=sns.cubehelix_palette(dark=.7,light=.4, as_cmap=True))
    plot = st.pyplot(fig)
    return fig, plot 

def empirical_risk(f,inputs, outputs):
    return tf.reduce_sum(1/2*tf.abs(outputs-f(inputs)))/len(inputs)


def fit_gd(model,inputs,outputs):
    with tf.GradientTape() as g:
        obj = model.loss(outputs,model.fsmooth(inputs))
        gradients = g.gradient(obj, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_gd(model,inputs,outputs,plot,fig):
    model.plot_hyperplane(plot,fig, update=False)
    placeholder, placeholder_bar = st.empty(), st.empty()
    my_bar = placeholder_bar.progress(0)
    T = 0
    for i in range(model.maxEpochs):
        t0 = time.time()
        fit_gd(model, inputs,outputs)
        T += time.time()-t0
        L, R = model.loss(outputs,model.fsmooth(inputs)).numpy(), empirical_risk(model.f, inputs,outputs).numpy()
        my_bar.progress(i/model.maxEpochs)
        model.summaryMessage(placeholder,T, L)
        if i % 100 == 0:
            model.plot_hyperplane(plot, fig, update=True)
        if L < 0.01 and R < 1e-5:
            break            
    placeholder_bar.empty()
    model.plot_hyperplane(plot, fig, update=True)



def fit_svm(model,inputs,outputs):
    K = model.calc_gramMatrix(inputs).numpy()
    y = np.array(outputs).astype(float)
    q = -1*np.ones(len(inputs))
    P = np.einsum('i,ij,j->ij',y,K,y)
    lb = np.zeros(len(inputs))
    A = y.reshape([1,len(y)])
    b = np.array([0.])
    lmbda = solve_qp(P, q, A=A, b=b, lb=lb, solver='cvxopt')
    model.get_SVs(inputs, outputs, lmbda)

def train_svm(model,inputs,outputs,plot,fig):
    model.addToPlot(fig.axes[0], label= model.prefix + " SVM", color= model.get_color())
    placeholder = st.empty()
    t0 = time.time()
    fit_svm(model,inputs,outputs)
    t1 = time.time()        
    model.b = model.get_bias()
    model.w = tf.reduce_sum(model.lambdaSV*model.ySV*tf.transpose(model.xSV),axis=1)
    model.summaryMessage(placeholder, t1-t0)
    model.plot_hyperplane(plot, fig)






class HyperplaneClassifier(object): 
    def __init__(self):
        self.line = None
        self.w, self.b = np.zeros(2), 0

    def graph(self):
        # graph of hyperplane (for plotting)
        xx = np.linspace(0,1,100)
        yy = (-self.w[0]*xx - self.b)/self.w[1]   ## <w,x> + b = 0 solved for x2
        return xx,yy

    def addToPlot(self,ax,label,color="black",alpha=1.0):
        grph = self.graph()
        self.line = ax.plot(grph[0],grph[1], color=color, alpha=alpha, label=label)

    def f(self,x):
        # decision function f_{w,b}
        return tf.sign(tf.reduce_sum(self.w*x, axis=-1)+self.b)


class GradientDescentHC(HyperplaneClassifier):
    def __init__(self, maxEpochs=10000):
        HyperplaneClassifier.__init__(self)
        self.w = tf.Variable(np.random.uniform(-1,1,size=(2)), dtype=tf.float64)
        self.b = tf.Variable(np.random.uniform(-1,1), dtype=tf.float64)
        self.trainable_variables = [self.w,self.b]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.1)
        self.loss = tf.keras.losses.MSE
        self.maxEpochs = maxEpochs

    def fsmooth(self,x):
        # smooth version of the decision function (for gradient descent)
        return tf.math.tanh(tf.reduce_sum(self.w*x, axis=-1)+self.b)

    def plot_hyperplane(self, plot, fig, update=False):
        if not(update):
            self.addToPlot(fig.axes[0], label="Gradient descent", color="black")
            fig.legend()
        else:
            self.line[0].set_ydata(self.graph()[1])
            plot.pyplot(fig)

    def summaryMessage(self, placeholder, T, L):
        with placeholder.beta_container(): 
            st.write('* _Storage (floats):_', 3, "(1 $d$-dimensional parameter plus bias)")
            st.write('* _Training time (seconds):_', round(T,4), "(loss: {:.4f})".format(L) )



class MLP(object):
    def __init__(self, maxEpochs):
        self.neurons = [GradientDescentHC("Neuron") for i in range(3)]
        self.trainable_variables = [var for neuron in self.neurons for var in neuron.trainable_variables]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=.01)
        self.loss = tf.keras.losses.MSE
        self.maxEpochs = maxEpochs
        self.ctr = None # for plotting

    def fsmooth(self,x):
        h = tf.concat([
            tf.expand_dims(self.neurons[0].fsmooth(x),axis=1),
            tf.expand_dims(self.neurons[1].fsmooth(x),axis=1)
            ],axis=1)
        return self.neurons[2].fsmooth(h)

    def f(self,x):
        return tf.sign(self.fsmooth(x))

    def plot_hyperplane(self, plot, fig, update):
        xx = np.linspace(0, 1, 100)
        yy = np.linspace(0, 1, 100)
        YY, XX = np.meshgrid(yy, xx)                # XX = [[0,...,1],...,[0,...,1]] = YY.T
        xy = np.vstack([XX.ravel(), YY.ravel()]).T  # .ravel() = cheap .flatten(), xy = coordinate list of grid
        Z = self.fsmooth(xy).numpy().reshape(XX.shape) 
        if update:
            self.ctr.collections[0].remove()
            self.ctr = fig.axes[0].contour(XX, YY, Z, label="MLP", colors='black', levels=[0], alpha=.7,linestyles=['-'])
            self.ctr.collections[0].set_label("MLP")
            fig.legend()
            plot.pyplot(fig)
        else:
            self.ctr = fig.axes[0].contour(XX, YY, Z, label="MLP", colors='black', levels=[0], alpha=.7,linestyles=['-'])

    def summaryMessage(self, placeholder, T, L):
        with placeholder.beta_container():   
            st.write('* _Storage (floats):_', int(np.sum([np.prod(np.shape(par)) for par in self.trainable_variables])), "(1 $d$-dimensional parameter, 2 hidden-dimensional parameters, 3 biases)")
            st.write('* _Training time (seconds):_', round(T, 4), "(loss: {:.4f})".format(L))





class LinearSVM(HyperplaneClassifier): 
    def __init__(self, prefix, inputs):
        HyperplaneClassifier.__init__(self)
        self.prefix = prefix
        self.xSV = None
        self.ySV = None
        self.lambdaSV = None
        self.line = None

    def summaryMessage(self, placeholder, T):
        with placeholder.beta_container():   
            st.write('* _Storage (floats):_', np.prod(np.shape(self.xSV)) + np.shape(self.ySV)[0] + 
                np.shape(self.lambdaSV)[0], "({} $d$-dim support vectors, plus their labels and coefficients)".format(len(self.ySV)))
            st.write('* _Training time (seconds):_', round(T, 4))

    def calc_gramMatrix(self, inputs):
        return tf.matmul(inputs,inputs, transpose_b = True)

    def get_SVs(self, inputs, outputs, lmbda):
        tfwhere = tf.where(lmbda>0.001)
        ySV = tf.gather(outputs,tfwhere)
        self.ySV = tf.cast(tf.reshape(ySV, [tf.shape(ySV)[0]]),dtype=tf.float64)    # safer than tf.squeeze
        lmbdaSV = tf.gather(lmbda,tfwhere)
        self.lambdaSV = tf.reshape(lmbdaSV, [tf.shape(lmbdaSV)[0]])
        xSV = tf.gather(inputs,tfwhere)
        self.xSV = tf.reshape(xSV, [tf.shape(xSV)[0],tf.shape(xSV)[-1]])
    
    def get_bias(self):
        KSV = self.calc_gramMatrix(self.xSV)
        return tf.reduce_mean(self.ySV - tf.reduce_sum(self.lambdaSV*self.ySV*KSV, axis=1)) 
        
    def margin_graph(self, a):
        # graph of parallel line to hyperplane with distance a, i.e. a=+-1 for margin
        xx = np.linspace(0,1,100)
        yy = (a-self.w[0]*xx - self.b)/self.w[1]   ## <w,x> + b = 0 solved for x2
        return xx,yy 

    def get_color(self):
        return "red" if (self.prefix == "Linear" or self.prefix == "RBF") else "blue" 

    def plot_hyperplane(self, plot, fig):
        self.line[0].set_ydata(self.graph()[1])
        marginPos = self.margin_graph(1)
        marginNeg = self.margin_graph(-1)
        fig.axes[0].plot(marginPos[0],marginPos[1], color=self.get_color(), ls='--', lw=1, alpha=.4)
        fig.axes[0].plot(marginNeg[0],marginNeg[1], color=self.get_color(), ls='--', lw=1, alpha=.4)
        fig.axes[0].fill_between(marginPos[0],marginNeg[1],marginPos[1], color = self.get_color(), alpha=0.2)
        plot.pyplot(fig)

    



class NonlinearSVM(LinearSVM):
    def __init__(self, prefix, inputs, kernel):
        LinearSVM.__init__(self, prefix, inputs)
        self.kernel = kernel

    def summaryMessage(self, placeholder, T):
        with placeholder.beta_container():   
            st.write('* _Kernel:_ ${}$'.format(self.kernel.equation))
            st.write('* _Storage (floats):_', np.prod(np.shape(self.xSV)) + np.shape(self.ySV)[0] + 
                np.shape(self.lambdaSV)[0], "({} $d$-dim support vectors, plus their labels and coefficients)".format(len(self.ySV)))
            st.write('* _Training time (seconds):_', round(T, 4))

    def calc_gramMatrix(self, inputs):
        return self.kernel.k(inputs,inputs)

    def f(self,x):
        # decision function f_{w,b}
        return (tf.reduce_sum(self.lambdaSV*self.ySV*self.kernel.k(self.xSV,x),axis=-1)+self.b).numpy()

    def plot_hyperplane(self, plot, fig):
        # change plotting from hyperplane to a contour plot to account for the nonlinear shape
        xx = np.linspace(0, 1, 100)
        yy = np.linspace(0, 1, 100)
        YY, XX = np.meshgrid(yy, xx)                # XX = [[0,...,1],...,[0,...,1]] = YY.T
        xy = np.vstack([XX.ravel(), YY.ravel()]).T  # .ravel() = cheap .flatten(), xy = coordinate list of grid
        Z = self.f(xy).reshape(XX.shape) 
        fig.axes[0].contour(XX, YY, Z, colors=self.get_color(), levels=[-1, 0, 1], alpha=0.4, linestyles=['--', '-', '--'])
        fig.axes[0].contourf(XX, YY, Z, colors=self.get_color(), levels=[-1, 0, 1], alpha=0.2, linestyles=['--', '-', '--'])
        fig.legend()
        plot.pyplot(fig)



class Kernel(object):
    def __init__(self, label, equation, params):
        self.label = label
        self.equation = equation
        self.params = params

    def kernel_rbf(self, inputs, x):
        gamma = self.params[0]
        exponent = tf.reduce_sum((tf.expand_dims(inputs, axis=0)-tf.expand_dims(x,axis=-2))**2,axis=-1)
        return tf.squeeze(tf.exp(-gamma*exponent))

    def kernel_tanh(self, inputs, x):
        kappa, c = self.params
        arg = tf.reduce_sum(tf.expand_dims(inputs, axis=0)*tf.expand_dims(x,axis=-2),axis=-1)
        return tf.tanh(kappa*arg - c)

    def k(self, inputs, x):
        if self.label=="RBF":
            return self.kernel_rbf(inputs, x)
        else:
            return self.kernel_tanh(inputs, x)



st.title('LS2: Instance-based learning')
"""
## 
"""

st.markdown("**Linear Classification**")
show_lsvm = st.checkbox("Linear Support Vector Machine")
show_gd = st.checkbox("Gradient Descent Hyperplane Classifier (= 1 Neuron)")
st.markdown("**Nonlinear Classification**")
show_svm1 = st.checkbox("Support Vector Machine with Gaussian RBF kernel")
show_svm2 = st.checkbox("Support Vector Machine with Tanh kernel")
show_mlp = st.checkbox("MLP (3 Neurons, 2xhidden+1xoutput)")


num  = st.slider('Number of datapoints', 5, 50, 10, key='num_linear')


if show_lsvm or show_gd:
    st.markdown("## Linear Classification")
    # data
    ftrue = lambda x: .9*x + 0.2
    X,Y = gen_data(num, ftrue)
    fig, plot = show_data(X,Y)
    
    if show_lsvm:
        st.write("### Linear Support Vector Machine")
        lsvm = LinearSVM("Linear" ,X)
        train_svm(lsvm, X,Y, plot, fig)
    if show_gd:
        st.write("### Gradient Descent Hyperplane Classifier")
        hc = GradientDescentHC(maxEpochs=10000)
        train_gd(hc,X,Y,plot,fig)


if show_svm1 or show_svm2 or show_mlp:
    st.markdown("## Nonlinear Classification")

    # data
    ftrue = lambda x: -2*(x-0.35)**2 + 0.7
    X1,Y1 = gen_data(num, ftrue)
    fig1, plot1 = show_data(X1,Y1)
    
    if show_svm1:
        st.write('### RBF Support Vector Machine')
        gamma = st.slider("Gamma", 0.05, 2.0, value = 1.0, step=0.05)
        svm1 = NonlinearSVM("RBF", X1, Kernel("RBF","K(x,y) = e^{-\\gamma \\|x-y\\|^2}", params=[gamma]))
        train_svm(svm1,X1,Y1, plot1, fig1)

    if show_svm2:
        st.write('### Tanh Support Vector Machine')
        col1, col2 = st.beta_columns(2)
        kappa = col1.slider("Kappa", 0.05, 1.8, value = 1.0, step=0.05)
        c = col2.slider("c", max(kappa,0.2), 8.0, value = 2.0, step=0.1)
        svm2 = NonlinearSVM("Tanh", X1, Kernel("Tanh","K(x,y) = \\tanh(\\kappa \\langle x,y\\rangle -c)", params=[kappa, c]))
        train_svm(svm2,X1,Y1, plot1, fig1)

    if show_mlp:
        st.write('### Multi-Layer-Perceptron')
        mlp = MLP(maxEpochs = 10000)
        train_gd(mlp,X1,Y1,plot1,fig1)









