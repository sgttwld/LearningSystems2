import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

def get_trainingData(Dsize,f):
    D = np.random.randint(0,2,[Dsize,2])
    Y = np.array([f(d[0],d[1]) for d in D]).reshape([Dsize,1])
    return np.append(D,Y, axis=1)


def get_XiXj(data,i,j):
    return np.sum([d[i]*d[j] for d in data])/len(data)

def get_Xi(data,i):
    return np.sum([d[i] for d in data])/len(data)

def get_Vi(data,i):
    return np.sum([d[i] for d in data])/len(data)

def get_Hj(data,j,numVisible):
    return np.sum([d[numVisible+j] for d in data])/len(data)

def get_ViHj(data,i,j,numVisible):
    return np.sum([d[i]*d[numVisible+j] for d in data])/len(data)

def get_loglikelihood(pmodel,data):
    return np.sum([np.log(pmodel[d[0],d[1],d[2]]) for d in data])/len(data)


## for evaluation
def get_pfromdata(data):
    xData,freqs = np.unique(data, return_counts = True, axis=0)
    pdata = np.zeros([2 for i in range(np.shape(data)[-1])])
    for xdat, freq in zip(xData,freqs):
        pdata[tuple(xdat)] = freq
    return pdata/np.sum(pdata)


# sampleVisible = samples[:,np.array([0,1,2])]



# def get_pfromdata_old(data):
#     xData,freqs = np.unique(data, return_counts = True, axis=0)
#     pdata = np.zeros([2 for i in range(np.shape(data)[-1])])
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 for xdat, freq in zip(xData,freqs):
#                     if i == xdat[0] and j == xdat[1] and k == xdat[2]:
#                         pdata[i,j,k] = freq
#     return pdata/np.sum(pdata)


def get_flat(p):
    n = np.prod(np.shape(p))
    return p.reshape((n))
            
def barplot(p):
    pflat = get_flat(p)
    fig, ax = plt.subplots()
    ax.bar(range(len(pflat)), pflat)
    st.pyplot(fig)

def barplots(ps, labels):
    fig, ax = plt.subplots(1,len(ps), figsize=(8,2))
    pflats = [get_flat(p) for p in ps]
    for i,p in enumerate(pflats):
        ax[i].set_title(labels[i])
        ax[i].bar(range(len(p)),p)
        ax[i].set_xticks(range(len(p)))
        ax[i].set_xticklabels([(i,j,k) for i in range(2) for j in range(2) for k in range(2)], Fontsize=5)
    st.pyplot(fig)


def show_evaluation(ll,trainingData,marginalVisible):
    llchance = get_loglikelihood((np.ones(8)/8).reshape([2,2,2]), trainingData) 
    barplots([get_pfromdata(trainingData),marginalVisible], 
        ['Training Distribution $p_\\mathrm{data}$', 'Equilibrium Distribution $p_\\mathrm{model}$'])
    st.markdown("### log likelihood")
    fig, ax = plt.subplots()
    ax.plot(range(len(ll)), llchance*np.ones(len(ll)), '--' ,label="chance level")
    ax.plot(range(len(ll)), ll, label="f")
    ax.set_ylim([-4,0])
    ax.legend()
    st.pyplot(fig)



# ### just for testing:
# import _lib.pr_func as pr
# def get_pfromdata(data):
#     xData,freqs = np.unique(data, return_counts = True, axis=0)
#     pdata = np.zeros([2 for i in range(np.shape(data)[-1])])
#     for i in range(2):
#         for j in range(2):
#             for k in range(2):
#                 for xdat, freq in zip(xData,freqs):
#                     if i == xdat[0] and j == xdat[1] and k == xdat[2]:
#                         pdata[i,j,k] = freq
#     return pdata/np.sum(pdata)
#
# def get_XiXj_prob(p,i,j):
#     X = [pr.func('f(x{})'.format(i), val=np.array([0,1])) for i in range(N)]
#     return pr.sum(X[i]*X[j]*p).val
# def get_Xi_prob(p,i):
#     X = [pr.func('f(x{})'.format(i), val=np.array([0,1])) for i in range(N)]
#     return pr.sum(X[i]*p).val