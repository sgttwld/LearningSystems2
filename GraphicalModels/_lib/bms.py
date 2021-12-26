import numpy as np
import streamlit as st
import _lib.pr_func as pr
from _lib.aux import *
from time import time

class BoltzmannMachine(object):
    def __init__(self, numVisible=3, numHidden=0, beta=1.0):
        self.numVisible = numVisible
        self.numHidden = numHidden
        self.beta = beta
        self.N = numVisible+numHidden
        self.b = np.random.normal(size=[self.N])
        self.wmat = self.get_wmat(np.random.normal(size=[int(self.N*(self.N-1)/2)]))
        pr.set_dims([('x{}'.format(i),2) for i in range(self.N)])  


    def get_wmat(self,w):
        upper = np.triu_indices(self.N, k=1)
        U = np.zeros((self.N, self.N))   
        U[upper] = w
        return U+U.T


    def get_joint(self):
        ## just for reference, this is not needed in practice 
        ## (too hard to normalize and/or numerical instabilities due to many products of small numbers)
        X = [pr.func('f(x{})'.format(i), val=np.array([0,1])) for i in range(self.N)]
        logp = 0
        for i in range(self.N):
            logp += -self.b[i]*X[i]
            for j in range(self.N):
                logp += self.wmat[i,j]*X[i]*X[j]/2
        return pr.exp(self.beta*logp).normalize()

    def get_marginal_visible(self):
        visibleUnits = ['x{}'.format(i) for i in range(self.numVisible)]
        return pr.sum(self.get_joint(),visibleUnits).val


    def gibbsSampling(self,num,burnin=10):
        tot = burnin+num
        x = np.random.randint(0,2, size=[self.N])
        data = []
        for n in range(tot):
            for k in range(self.N):
                deltaE = self.b[k] - np.sum([self.wmat[k,i]*x[i] for i in range(self.N)])
                prob = 1/(1+np.exp(self.beta*deltaE))
                x[k] = np.random.choice([0,1],p=np.array([1-prob,prob]))
            data.append(np.copy(x))
        return np.array(data)[burnin:]


    def conditionalGibbsSampling(self,trainingData,num,burnin=10):
        data = []
        for n in range(num):
            x = np.random.randint(0,2, size=[self.N])
            v = trainingData[np.random.choice(len(trainingData))] 
            for idx in range(self.numVisible):
                x[idx] = v[idx]
            for iter in range(burnin):
                for k in range(self.numVisible,self.N):
                    deltaE = self.b[k] - np.sum([self.wmat[k,i]*x[i] for i in range(self.N)])
                    prob = 1/(1+np.exp(self.beta*deltaE))
                    x[k] = np.random.choice([0,1],p=np.array([1-prob,prob]))
            data.append(np.copy(x))
        return np.array(data)


    def train(self, trainingData, learningRate, samplesPerEpoch=50, epochs=300, burnin=10):
        if np.shape(trainingData)[-1] != self.numVisible:
            print("ERROR: Dimensionality of learning data needs to coincide with the number of visible units")
            return []
        lls = []
        placeholder = st.empty()
        progressBar = st.progress(0)
        deltat = 0
        for n in range(epochs):
            lr = learningRate(n)
            t0 = time()
            simulatedData = self.gibbsSampling(num=samplesPerEpoch,burnin=burnin)
            if self.numHidden > 0:
                conditionalData = self.conditionalGibbsSampling(trainingData,num=samplesPerEpoch,burnin=burnin) 
            else:
                conditionalData = trainingData   
            # update biases
            Xidata = np.sum(conditionalData,axis=0)/len(conditionalData)
            Ximodel = np.sum(simulatedData,axis=0)/len(simulatedData)
            self.b += lr*(Ximodel-Xidata)
            # update weights
            for i in range(self.N):
                for j in range(self.N):
                    if i != j:
                        XiXjdata = get_XiXj(conditionalData,i,j)
                        XiXjmodel = get_XiXj(simulatedData,i,j)
                        self.wmat[i,j] = self.wmat[i,j] + lr*(XiXjdata - XiXjmodel)
            t1 = time()
            ## evaluate
            p = self.get_marginal_visible()
            ll = get_loglikelihood(p,trainingData)
            llchance = get_loglikelihood((np.ones(8)/8).reshape([2,2,2]), trainingData) 
            lls.append(ll)
            deltat = deltat + (t1-t0 - deltat)/(n+1)
            placeholder.text('log likelihood: {:.3f}, (chance: {:.3f}), t/epoch: {:.2f} ms'.format(ll,llchance,1000*deltat))
            progressBar.progress(n/epochs)
        return lls



class RestrictedBoltzmannMachine(object):
    def __init__(self, numVisible=3, numHidden=0, beta=1):
        self.numVisible = numVisible
        self.numHidden = numHidden
        self.beta = beta
        self.a = np.random.normal(size=[self.numVisible])
        self.b = np.random.normal(size=[self.numHidden])
        self.wmat = np.random.normal(size=[self.numVisible,self.numHidden])
        pr.set_dims([('v{}'.format(i),2) for i in range(self.numVisible)]
            + [('h{}'.format(i),2) for i in range(self.numHidden)])         # just for analysis  

    def get_joint(self):
        ## just for reference, this is not needed in practice 
        ## (too hard to normalize and/or numerical instabilities due to many products of small numbers)
        V = [pr.func('f(v{})'.format(i), val=np.array([0,1])) for i in range(self.numVisible)]
        H = [pr.func('f(h{})'.format(i), val=np.array([0,1])) for i in range(self.numHidden)]
        logp = 0
        logp += sum([-self.a[i]*V[i] for i in range(self.numVisible)])
        logp += sum([-self.b[j]*H[j] for j in range(self.numHidden)])
        for i in range(self.numVisible):
            for j in range(self.numHidden):
                logp += self.wmat[i,j]*V[i]*H[j]
        return pr.exp(self.beta*logp).normalize()

    def get_marginal_visible(self):
        visibleUnits = ['v{}'.format(i) for i in range(self.numVisible)]
        return pr.sum(self.get_joint(),visibleUnits).val

    def sampleHidden(self,v):
        deltaEh = self.b - np.matmul(self.wmat.T,v)
        probh = 1/(1+np.exp(self.beta*deltaEh))
        r = np.random.rand(self.numHidden)
        return r<probh

    def sampleVisible(self,h):
        deltaEv = self.a - np.matmul(self.wmat,h)
        probv = 1/(1+np.exp(self.beta*deltaEv))
        r = np.random.rand(self.numVisible)
        return r<probv

    def gibbsSampling(self,num,burnin=10):
        tot = burnin+num
        data = []
        v = np.random.randint(0,2, size=[self.numVisible])
        h = np.zeros([self.numHidden])
        for n in range(tot):
            h = self.sampleHidden(v)
            v = self.sampleVisible(h)
            data.append(np.hstack([v,h]).astype(int))
        return np.array(data)[burnin:]

    def conditionalSampling(self,trainingData, num):
        data = []
        for n in range(num):            
            v = trainingData[np.random.choice(len(trainingData))] 
            h = self.sampleHidden(v)
            data.append(np.hstack([v,h]).astype(int))
        return data

    def get_reconData(self,trainingData, num):
        data = []
        for n in range(num):            
            v = trainingData[np.random.choice(len(trainingData))] 
            h = self.sampleHidden(v)
            v1 = self.sampleVisible(h)
            h1 = self.sampleHidden(v1)
            data.append(np.hstack([v1,h1]).astype(int))
        return data

    def train(self, trainingData, learningRate, epochs=300, samplesPerEpoch=50, burnin=10, rule='standard'):
        ## straight log-likelihood minimization
        placeholder = st.empty()
        progressBar = st.progress(0)
        lls = []
        deltat = 0
        for n in range(epochs):
            t0 = time()
            lr = learningRate(n)
            conditionalData = self.conditionalSampling(trainingData,num=samplesPerEpoch)
            
            if rule == 'standard':
                simulatedData = self.gibbsSampling(num=samplesPerEpoch,burnin=burnin)
            elif rule == 'cd':
                simulatedData = self.get_reconData(trainingData,num=samplesPerEpoch)

            Xidata = np.sum(conditionalData,axis=0)/len(conditionalData)
            Ximodel = np.sum(simulatedData,axis=0)/len(simulatedData)
            diff = Ximodel-Xidata
            self.a += lr*diff[:self.numVisible]
            self.b += lr*diff[self.numVisible:] 

            for i in range(self.numVisible):
                for j in range(self.numHidden):
                    ViHjdata = get_ViHj(conditionalData,i,j,self.numVisible)
                    ViHjmodel = get_ViHj(simulatedData,i,j, self.numVisible)
                    self.wmat[i,j] = self.wmat[i,j] + lr*(ViHjdata - ViHjmodel)
            t1 = time()
            ## marginal of visible units:
            p = self.get_marginal_visible()
            ll = get_loglikelihood(p,trainingData)
            llchance = get_loglikelihood((np.ones(8)/8).reshape([2,2,2]), trainingData) 
            lls.append(ll)
            deltat = deltat + (t1-t0 - deltat)/(n+1)
            placeholder.text('log likelihood: {:.3f}, (chance: {:.3f}), t/epoch: {:.2f} ms'.format(ll,llchance,1000*deltat))
            progressBar.progress(n/epochs)
        return lls





