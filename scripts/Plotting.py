import torch as t
import matplotlib.pyplot as plt


class Plotter():
    def __init__(self):
        super().__init__()

    def plot_restrict_time(self):
        assert self.restrict_time
        tmp = t.linspace(-20,self.T+20,100)
        fig, ax = plt.subplots(figsize=(15,5),ncols=3)
        ax[0].plot(tmp,self.restrict_time(tmp))
        ax[0].hlines([0,self.T],tmp.min(),tmp.max(),color='k',linestyle='--')
        zoom = 5
        mask = (tmp>self.T-zoom)&(tmp<self.T+zoom)
        ax[1].plot(tmp[mask],self.restrict_time(tmp)[mask])
        ax[1].hlines(self.T,self.T-zoom,self.T+zoom,'k',linestyle='--')
        ax[1].vlines(self.T,self.T-0.5,self.T+0.5,'b',linestyle='--')
        mask = (tmp>0-zoom)&(tmp<0+zoom)
        ax[2].plot(tmp[mask],self.restrict_time(tmp)[mask])
        ax[2].hlines(0,0-zoom,0+zoom,'k',linestyle='--')
        ax[2].vlines(0,0-0.5,0+0.5,'b',linestyle='--')
        return
    
    def plot_activation_func(self):
        assert self.activation_func
        plt.plot(self.times,self.activation_func(self.times).detach(),'b.')
        plt.ylim(0,None)
        try:
            plt.plot(self.times, self.get_control().detach()+0.5)
            plt.plot(self.times, self.envelope_func().detach()+0.5)
            plt.plot(self.times, -self.envelope_func().detach()+0.5)
        except:
            pass
        return
    
    def plot_occupancy(self): #FIX!
        alphas = self.activation_func(self.times).detach()
        Hs = self.get_H(alphas.flip(0),self.get_control().flip(0)).detach()
        occ = t.zeros((3,Hs.shape[0]))
        exp_mat = t.matrix_exp(-1j*Hs*self.dt)
        wavefunc = self.eigvecs[:,0]
        eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=alphas))
        for i, mat in enumerate(exp_mat.flip(0)):
            wavefunc = mat@wavefunc
            
            occ[0,i] = t.square(t.abs(eigvecs[i,:,0].unsqueeze(1).adjoint()@wavefunc))
            occ[1,i] = t.square(t.abs(eigvecs[i,:,1].unsqueeze(1).adjoint()@wavefunc))
        occ[2] = 1-occ.sum(0)

        fig, ax = plt.subplots()
        ax.plot(self.times,occ.T)
