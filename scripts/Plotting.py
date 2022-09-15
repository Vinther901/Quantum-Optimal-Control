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
        fig, ax = plt.subplots()
        ax.plot(self.times,self.activation_func(self.times).detach(),'b')
        ax.set_ylim(0,None)
        try:
            ax.plot(self.times, self.get_control().detach()+0.5)
            ax.plot(self.times, self.envelope_func().detach()+0.5)
            ax.plot(self.times, -self.envelope_func().detach()+0.5)
        except:
            print("Didn't plot pulse in .plot_activation_func")
            pass
        return
    
    def plot_occupancy(self,indices=[0,1]):
        occ = self.get_occupancy(indices).detach()
        fig, ax = plt.subplots()
        ax.plot(self.times,occ[:-1].T)
        ax.plot(self.times,occ[-1],'k--')#,zorder=-1)
    
    def plot_potential(self,alpha=1):
        phi = t.linspace(-t.pi,t.pi,self.NHilbert)
        eigvals, eigvecs = t.linalg.eigh(self.get_H(t.tensor([alpha])).squeeze())

        fig, ax = plt.subplots()
        potential = (-2*self.EJ*t.cos(phi) + alpha*self.EJ*t.cos(2*phi - self.phi_ext)).real
        ax.plot(phi, potential,'k')
        indices = [0,1,2,3]
        for i in indices:
            eigvec = t.fft.fftshift(t.fft.ifft(eigvecs[:,i])*t.sqrt(t.tensor(self.NHilbert)))
            ax.fill_between(phi,t.abs(eigvec)**2*self.NHilbert+eigvals[i],eigvals[i],label='n=%d'%i,edgecolor='k',alpha=0.8)
        ax.set_ylim(potential.min()*1.01,potential[self.q_max]*0.95)
        return
    
    def plot_run(self, save=False, fig_name="Optimized.pdf"):
        fig, ax = plt.subplots(figsize=(15,10),ncols=3,nrows=2)
        ax[0,0].plot(self.times, self.get_init_pulse())
        pulse_lim = (min(self.get_init_pulse().min(),self.get_control().detach().min()),
                     max(self.get_init_pulse().max(),self.get_control().detach().max()))
        ax[0,0].set(title="Initial Pulse",
                    ylim=pulse_lim)
        ax00_2nd = ax[0,0].twinx()
        ax00_2nd.plot(self.times,self.init_activation_func(self.times),'k--')
        ax00_2nd.set_ylim(-0.01,1.01)
        

        ax[0,1].plot(self.stored_weights.T)
        ax[0,1].set(title="Loss Weights",
                    yscale='log',
                    ylim=(None,1))
        ax[1,1].plot(self.stored_losses.T,label=self.params_dict['loss_funcs'])
        ax[1,1].set_prop_cycle(None)
        ax[1,1].plot((self.stored_losses*self.stored_weights).T,linestyle='--')
        ax[1,1].plot((self.stored_losses*self.stored_weights).mean(0),'k--')
        ax[1,1].set(xlabel="Epoch",
                    title="Losses")
        ax[1,1].legend()
        ax[1,1].set_yscale('log')

        ax[0,2].plot(self.times, self.get_control().detach())
        ax[0,2].set(title="Optimized Pulse",
                    ylim=pulse_lim)
        ax02_2nd = ax[0,2].twinx()
        ax02_2nd.plot(self.times,self.activation_func(self.times).detach(),'k--')
        ax02_2nd.set_ylim(-0.01,1.01)
        ax[1,2].plot(self.times, self.get_occupancy().detach().T)
        ax[1,2].set(xlabel="Time [ns]",
                    title="Occupation (optimized pulse)")

        self.latest_matrix_exp = t.matrix_exp(-1j*self.dt*self.get_H(self.activation_func(self.times),self.get_init_pulse()))
        ax[1,0].plot(self.times, self.get_occupancy().detach().T,label=["$\psi_0$","$\psi_1$","$\psi_{rest}$"])
        _ = self()
        ax[1,0].legend()
        ax[1,0].set(xlabel="Time [ns]",
                    title="Occupation (initial pulse)")
        
        fig.tight_layout()

        if save:
            from os.path import join
            fig.savefig(join(self.params_dict['exp_path'],fig_name))
