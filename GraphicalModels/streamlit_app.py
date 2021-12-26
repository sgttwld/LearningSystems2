import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import _lib.pr_func as pr
from _lib.aux import *
from _lib.bms import *

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

N = 3
pr.set_dims([('a',N),('b',N),('c',N),('d',N)])

st.title('LS2: Learning in Graphical Models')
"""
## 
"""

show_examplemodel = st.checkbox("Example: Model mutual dependencies by undirected graphs")
show_boltzmannmachine = st.checkbox("Boltzmann machine")
show_rbm = st.checkbox("Restricted Boltzmann machine (standard learning rule)")
show_rbmcd = st.checkbox("Restricted Boltzmann machine (approx. contrastive divergence)")

if show_examplemodel:
        
    """
    Anton (A), Barbara (B), Claudia (C) and Dennis (D) choose between pasta, fish, and meat: 
    * Barbara dislikes fish and likes both meat and pasta.
    * Anton, Barbara, and Claudia prefer to choose the same.
    * Claudia likes to eat low carb.
    * Dennis slightly prefers meat over fish and pasta, but he is also a hippster, so he likes to choose 
    something else than all the others.

    We model this by a fully connected undirected graph with the four nodes $A,B,C,D$ and energy functions with 
    energy $0$ for the preferred food, and energy $10$ for the less preferred (except the slider values below).

    """

    varnames = ['a','b','c','d']
    valnames = ['pasta','fish','meat']

    b1 = st.slider("How much Barbara dislikes fish", 0.0, 10.0, value = 5.0, step=1.0)
    d1 = st.slider("How much Dennis prefers meat over fish and pasta", 0.0, 10.0, value = 5.0, step=1.0)


    Eb = pr.func('f(b)', val=np.array([0,b1,0]))
    Ec = pr.func('f(c)', val=np.array([10,0,0]))
    Ed = pr.func('f(d)', val=np.array([d1,d1,0]))
    Eabc = pr.func('f(a,b,c)', val=10*np.ones((N,N,N)))
    Eabcd = pr.func('f(a,b,c,d)', val=np.zeros((N,N,N,N)))

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if i==j and j==k:
                    Eabc.val[i,j,k] = 0
                for l in range(N):
                    if l==i or l==j or l==k: 
                        Eabcd.val[i,j,k,l] = 10

    expE = pr.exp(-Eb-Eabc-Ec-Eabcd-Ed)

    joint = expE.normalize(varnames)
    marginals = []
    for var in varnames:
        marginals.append(pr.sum(expE,[var]).normalize())

    st.write('### Marginals')
    fig, ax = plt.subplots(2,2)
    fig.subplots_adjust(hspace=0.4)
    for k in range(4):
        i,j = np.unravel_index(k,(2,2))
        ax[i,j].bar(range(N),marginals[k].val)
        ax[i,j].set_title('p({})'.format(varnames[k].upper()))
        ax[i,j].set_xticks([0,1,2])
        ax[i,j].set_xticklabels(valnames)
    st.pyplot(fig)

    st.write('### Inference')
    pdga = pr.sum(expE,['a','d']).normalize(['d'])
    fig1,ax = plt.subplots(1,3, figsize=(8,2))
    ax[0].bar(range(N),pdga.eval('a',0).val)
    ax[0].set_title('p(D|A=pasta)')
    ax[1].bar(range(N),pdga.eval('a',1).val)
    ax[1].set_title('p(D|A=fish)')
    ax[2].bar(range(N),pdga.eval('a',2).val)
    ax[2].set_title('p(D|A=meat)')    
    st.pyplot(fig1)


######################
### boltzmann machines
######################


fun = {
    'OR': lambda x,y: x+y > 0.5,
    'AND': lambda x,y: x+y > 1.5,
    'XOR': lambda x,y: (x+y)%2,
}
numEpochs = 1500
numTrainingData = 1000

if show_boltzmannmachine:
    st.markdown("## Boltzmann machine")
    fname = st.selectbox('Which function should be learned?',('OR', 'AND', 'XOR'))
    burnin = 10 
    hiddenUnits = st.slider("Hidden units", 0, 7, value = 0, step=1, key='hbm_hiddenUnits')
    beta = st.slider("Inverse temperature", 0.25, 5.0, value = 2.5, step=0.25, key='hbm_beta')
    samplesPerEpoch = st.slider("Samples per epoch", 1, 100, value = 50, step=1, key='hbm_samplesPerEpoch')
    learningRate = [
        lambda n: .2 if n < 1000 else (.1 if n<1500 else .05),      # hiddenUnits = 0
        lambda n: .2 if n < 1000 else (.1 if n<1500 else .05),      # hiddenUnits = 1
        lambda n: .1 if n < 1000 else (.05 if n<2000 else .01),     # hiddenUnits = 2
        lambda n: .1 if n < 400 else (.05 if n<1500 else .01),      # hiddenUnits = 3
        lambda n: .1 if n < 400 else (.05 if n<750 else .01),       # hiddenUnits = 4
        lambda n: .1 if n < 250 else (.05 if n<500 else .01),       # hiddenUnits = 5
        lambda n: .1 if n < 150 else (.05 if n<250 else .01),       # hiddenUnits = 6
        lambda n: .1 if n < 100 else (.05 if n<200 else .01),       # hiddenUnits = 7
        ]
    st.markdown("### Learning {} ({} hidden unit{})".format(fname,hiddenUnits,"s" if hiddenUnits>1 else ""))
    bm = BoltzmannMachine(numVisible=3,numHidden=hiddenUnits,beta=beta)
    trainingData = get_trainingData(f=fun[fname],Dsize=numTrainingData)
    ll = bm.train(trainingData, learningRate[hiddenUnits], samplesPerEpoch, numEpochs, burnin)
    show_evaluation(ll,trainingData,bm.get_marginal_visible())


if show_rbm:
    st.markdown("## Restricted Boltzmann machine (standard learning rule)")
    fname = st.selectbox('Which function should be learned?',('XOR','OR', 'AND'))
    burnin = 100
    hiddenUnits = st.slider("Hidden units", 0, 9, value = 4, step=1, key='rbm_hiddenUnits')
    beta = st.slider("Inverse temperature", 0.25, 5.0, value = 2.5, step=0.25, key='rbm_beta')
    samplesPerEpoch = st.slider("Samples per epoch", 10, 100, value = 50, step=5, key='rbm_samplesPerEpoch')
    learningRate = [
        lambda n: .2,                                             # hiddenUnits = 0
        lambda n: .3 if n<500 else (.2 if n<1500 else .1),      # hiddenUnits = 1
        lambda n: .2 if n<500 else (.1 if n<1500 else .025),      # hiddenUnits = 2
        lambda n: .2 if n<500 else (.1 if n<1000 else .025),      # hiddenUnits = 3
        lambda n: .2 if n<500 else (.1 if n<1000 else .025),      # hiddenUnits = 4
        lambda n: .1 if n<500 else (.05 if n<1500 else .01),      # hiddenUnits = 5
        lambda n: .1 if n<300 else (.05 if n<1000 else .01),      # hiddenUnits = 6
        lambda n: .1 if n<200 else (.05 if n<700 else .01),       # hiddenUnits = 7
        lambda n: .075 if n<500 else (.05 if n<1000 else .01),    # hiddenUnits = 8
        lambda n: .075 if n<500 else (.05 if n<800 else .01),     # hiddenUnits = 9
        ]
    st.markdown("### Learning {} ({} hidden unit{})".format(fname,hiddenUnits,"s" if hiddenUnits>1 else ""))    
    rbm = RestrictedBoltzmannMachine(numVisible=3,numHidden=hiddenUnits,beta=beta)
    trainingData = get_trainingData(Dsize=numTrainingData,f=fun[fname])
    ll = rbm.train(trainingData, learningRate[hiddenUnits], numEpochs, samplesPerEpoch, burnin=burnin, rule='standard')
    show_evaluation(ll,trainingData,rbm.get_marginal_visible())




if show_rbmcd:
    st.markdown("## Restricted Boltzmann machine (contrastive divergence)")
    burnin = 100
    hiddenUnits = st.slider("Hidden units", 0, 9, value = 4, step=1, key='rbmcd_hiddenUnits')
    beta = st.slider("Inverse temperature", 0.25, 5.0, value = 2.5, step=0.25, key='rbmcd_beta')
    samplesPerEpoch = st.slider("Samples per epoch", 10, 100, value = 50, step=5, key='rbmcd_samplesPerEpoch')
    learningRate = [
        lambda n: .2,                                             # hiddenUnits = 0
        lambda n: .3 if n<500 else (.2 if n<1500 else .1),        # hiddenUnits = 1
        lambda n: .2 if n<500 else (.1 if n<1500 else .025),      # hiddenUnits = 2
        lambda n: .2 if n<250 else (.1 if n<500 else .025),       # hiddenUnits = 3
        lambda n: .2 if n<400 else (.1 if n<1000 else .025),      # hiddenUnits = 4
        lambda n: .2 if n<400 else (.1 if n<900 else .01),      # hiddenUnits = 5
        lambda n: .1 if n<300 else (.05 if n<1000 else .01),      # hiddenUnits = 6
        lambda n: .1 if n<200 else (.05 if n<700 else .01),       # hiddenUnits = 7
        lambda n: .075 if n<500 else (.05 if n<1000 else .01),    # hiddenUnits = 8
        lambda n: .075 if n<500 else (.05 if n<800 else .01),     # hiddenUnits = 9
        ]
    fname = st.selectbox('Which function should be learned?',('XOR','OR', 'AND'))
    st.markdown("### Learning {} ({} hidden unit{})".format(fname,hiddenUnits,"s" if hiddenUnits>1 else ""))    
    trainingData = get_trainingData(Dsize=numTrainingData,f=fun[fname])
    rbmcd = RestrictedBoltzmannMachine(numVisible=3,numHidden=hiddenUnits,beta=beta)
    ll = rbmcd.train(trainingData, learningRate[hiddenUnits], numEpochs, samplesPerEpoch, burnin=burnin, rule='cd')
    show_evaluation(ll,trainingData,rbmcd.get_marginal_visible())






























