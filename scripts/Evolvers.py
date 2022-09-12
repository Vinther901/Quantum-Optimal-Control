import torch as t


class QTrotter():
    def __init__(self):
        self.init_wavefuncs = self.eigvecs
        self.subNHilbert = self.NHilbert**int(self.params_dict['dim'][0])
        super().__init__()
    
    def get_H(self,alphas=t.tensor([1]),control = t.tensor([0])):
        # assert type(alpha) == t.Tensor
        return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)
    

class ETrotter():
    def __init__(self):
        H0 = self.KinE.repeat((self.NTrot,1,1)) + self.V(alphas=self.activation_func(self.times),control=t.zeros(self.NTrot))
        E0, U0s = t.linalg.eigh(H0)
        U0s = t.concat([U0s[[0]],U0s,U0s[[-1]]],0)
     
        fphase = U0s[:,self.NHilbert//2 - 2].angle() #8 #self.NHilbert//2 - 2
        U0s = U0s*t.exp(-1j*fphase).unsqueeze(1)

        # U0dot = 1/(2*self.dt)*(U0s[2:].adjoint()@U0s[1:-1] - U0s[1:-1].adjoint()@U0s[2:])
        U0dot = 1/(4*self.dt)*(U0s[2:].adjoint()@U0s[:-2] - U0s[:-2].adjoint()@U0s[2:])
        self.H0_term = t.diag_embed(E0).type(t.cfloat) + 1j*U0dot
        self.U0s = U0s[1:-1,:,:self.subNHilbert]
        # self.basis_change = self.U0s[-(i+1)].adjoint()@self.U0s[-i]
        self.H0_term = self.H0_term[:,:self.subNHilbert,:self.subNHilbert]
        self.init_wavefuncs = self.U0s[0].adjoint()@self.eigvecs
        super().__init__()

    def get_H(self,alphas=t.tensor([1]), control=t.tensor([0])):
        if alphas.shape[0] > 1:
            V = self.EJ*control.view(-1,1,1)*self.q_mat
            return self.H0_term + self.U0s.adjoint()@V@self.U0s
        else:
            # return t.diag(t.linalg.eigvalsh(self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)).squeeze())
            return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)

    def _get_H2(self,alphas=t.tensor([1]),control = t.tensor([0])):
        # assert type(alpha) == t.Tensor
        return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)

class ETrotter3():
    def __init__(self):
        from tqdm import tqdm
        # import numpy as np
        extra = 0
        times = t.linspace(0,self.T,self.NTrot + (self.NTrot-1)*extra)
        H0 = self.KinE.repeat((times.shape[0],1,1)) + self.V(alphas=self.init_activation_func(times),control=t.zeros(times.shape[0]))
        # H0 = self.KinE.repeat((self.NTrot,1,1)) + self.V(alphas=self.activation_func(self.times),control=t.zeros(self.NTrot))
        
        # H0 = self.get_H(alphas=self.activation_func(self.times),control=t.zeros(self.NTrot))

        E0s = t.linalg.eigvalsh(H0)
        Fs = 1/(E0s.unsqueeze(2) - E0s.unsqueeze(1))
        self.E0s = E0s

        # E0s = np.linalg.eigvalsh(H0).astype(np.float64)
        # Fs = 1/(E0s.reshape(self.NTrot,self.NHilbert,1) - E0s.reshape(self.NTrot,1,self.NHilbert))
        # self.Fs = Fs
        # Fs = t.as_tensor(Fs)
        # E0s = t.as_tensor(E0s)
        # print(self.NHilbert**2*self.NTrot - Fs.isfinite().sum())
        # Fs[~Fs.isfinite()] = 0
        
        diag_inds = [_ for _ in range(self.NHilbert)]
        Fs[:,diag_inds,diag_inds] = 0
        print(self.NHilbert**2*times.shape[0] - Fs.isfinite().sum())
        # self.Fs = Fs
        # Fs = t.concat([Fs,Fs[[-1]]],0)
        H0 = t.concat([H0,H0[[-1]]],0)

        U0s_adj = t.zeros((times.shape[0]+2,self.NHilbert,self.NHilbert),dtype=t.cfloat)
        _, tmp_eigvec = t.linalg.eigh(H0[0])
        U0s_adj[0] = tmp_eigvec.adjoint()
        for i in tqdm(range(times.shape[0])):
            dH = H0[i+1] - H0[i]
            U0s_adj[i+1] = (self.Id + Fs[i]*(U0s_adj[i]@dH@U0s_adj[i].adjoint()))@U0s_adj[i]
        self.U0s_adj = U0s_adj
        U0s = U0s_adj.adjoint()
        
        U0dot = 1/(4*self.dt)*(U0s[2:].adjoint()@U0s[:-2] - U0s[:-2].adjoint()@U0s[2:])
        self.H0_term = t.diag_embed(E0s).type(t.cfloat) + 1j*U0dot
        self.U0s = U0s[1:-1,:,:self.subNHilbert]

        self.H0_term = self.H0_term[::(extra+1)]
        self.U0s = self.U0s[::(extra+1)]
        
        self.H0_term = self.H0_term[:,:self.subNHilbert,:self.subNHilbert]
        self.init_wavefuncs = self.U0s[0].adjoint()@self.eigvecs
        super().__init__()

    def get_H(self,alphas=t.tensor([1]), control=t.tensor([0])):
        if alphas.shape[0] > 1:
            V = self.EJ*control.view(-1,1,1)*self.q_mat
            return self.H0_term + self.U0s.adjoint()@V@self.U0s
        else:  
            return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)


class ETrotter2():
    def __init__(self):
        H0 = self.KinE.repeat((self.NTrot,1,1)) + self.V(alphas=t.ones(self.NTrot),control=t.zeros(self.NTrot))
        E0, U0s = t.linalg.eigh(H0)
        U0s = t.concat([U0s[[0]],U0s,U0s[[-1]]],0)
        # U0dot = 1/(2*self.dt)*(U0s[2:].adjoint() - U0s[:-2].adjoint())@(U0s[:-2] + U0s[2:])/2
        self.H0_term = t.diag_embed(E0).type(t.cfloat) #+ 1j*U0dot

        # self.U0 = U0s[1]
        # self.U0s = U0s[1:]
        self.U0s = U0s[1:-1]
        self.U0s = self.U0s[:,:,:self.subNHilbert]
        self.H0_term = self.H0_term[:,:self.subNHilbert,:self.subNHilbert]

        self.init_wavefuncs = self.U0s[-1].adjoint()@self.eigvecs
        super().__init__()

    def get_H(self,alphas=t.tensor([1]), control=t.tensor([0])):
        if alphas.shape[0] > 1:
            V = self.EJ*((alphas.view(-1,1,1) - 1)/2*self.cos2_mat + control.view(-1,1,1)*self.q_mat)
            return self.H0_term + self.U0s.adjoint()@V@self.U0s
        else:
            # return t.diag(t.linalg.eigvalsh(self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)).squeeze())
            return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)


class BasisChanges(): #Not really implemented
    def __init__(self):
        self.t1 = t.exp(-1j*self.dt*4*self.params_dict['EC']*q**2)
        self.t2 = t.exp(-1j*self.dt*self.EJ*q)
        t3 = t.matrix_exp(1j*self.dt*self.EJ*self.cos_mat).cfloat()
        self.t4, u4 = t.linalg.eig(t.matrix_exp(-1j*self.dt*self.EJ*self.cos2_mat).cfloat())
        self.t3 = t3@u4
        self.u4_adj = u4.adjoint().cfloat()

        super().__init__()
    
    def get_H(self,alphas=t.tensor([1]), control=t.tensor([0])):
        if alphas.shape[0] > 1:
            alphas = self.activation_func(self.times)
            pulse = self.get_control()
            # M2s = self.t1*t.pow(self.t2,pulse.view(-1,1))
            # matrices = t.zeros((2*self.NTrot,self.NHilbert,self.NHilbert)).type(t.complex128)
            # matrices[0::2] = t.diag_embed(self.t1*t.pow(self.t2,pulse.view(-1,1)))
            # matrices[1::2] = self.t3@t.diag_embed(t.pow(self.t4,alphas.view(-1,1)))@self.u4_adj
            M1s = t.diag_embed(self.t1*t.pow(self.t2,pulse.view(-1,1)))
            M2s = self.t3@t.diag_embed(t.pow(self.t4,alphas.view(-1,1)))@self.u4_adj
            return M1s@M2s
        else:
            return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)




###############Graveyard###################

        # tmp = U0s[2:].adjoint()@U0s[1:-1]
        # tmp2 = U0s[2:].adjoint().sum(2,keepdim=True)*U0s[1:-1].sum(1,keepdim=True)
        # tmp = tmp/tmp2
        # self.tmp = tmp
        # H = self._get_H2(alphas=self.activation_func(self.times),control=t.zeros(self.NTrot))
        # eigvals, eigvecs = t.linalg.eigh(H[0])
        # mat_exp = t.matrix_exp(-1j*H*self.dt).flip(0)
        # cum_mat_exp = [t.linalg.multi_dot(list(mat for mat in mat_exp[-i:])) for i in range(2,self.NTrot+1)]+ [mat_exp[-1]]+[self.Id]
        # cum_mat_exp = t.concat([tensor.unsqueeze(0) for tensor in cum_mat_exp],0).flip(0)
        # tmp = cum_mat_exp@eigvecs
        # tmp = ((tmp[1:].adjoint()@tmp[:-1])[:,[_ for _ in range(self.NHilbert)],[_ for _ in range(self.NHilbert)]]).angle()
        # # self.tmp = tmp
        # angles_0 = t.concat([t.zeros(2,21),tmp],0)
        # tmp_U0s = U0s*t.exp(+1j*angles_0).unsqueeze(1)

        # angles = (tmp_U0s[2:].adjoint()@tmp_U0s[1:-1])[:,[_ for _ in range(self.NHilbert)],[_ for _ in range(self.NHilbert)]].angle()
        # angles = angles.cumsum(0)# - tmp.cumsum(0)
        # angles = t.concat([t.zeros((2,21)),angles],0)

        # U0s = tmp_U0s*t.exp(+1j*(angles+angles_0)).unsqueeze(1)
        # U0s = t.concat([tmp,tmp[[-1]]],0)
        # print((U0s[2:].adjoint()@U0s[1:-1])[:,[_ for _ in range(self.NHilbert)],[_ for _ in range(self.NHilbert)]].angle())
        # print(angles_0)
        # self.U0dot = 1/(2*self.dt)*(U0s[2:].adjoint() - U0s[:-2].adjoint())@U0s[1:-1]
        # self.U0dot = 1/self.dt*(U0s[2:].adjoint() - U0s[1:-1].adjoint())@U0s[1:-1]
        # self.U0dot = 1/(self.dt)*(-0.5*U0s[2:]-1.5*U0s[:-2]+2*U0s[1:-1]).adjoint()@U0s[1:-1]
        # U0dot = 1/(2*self.dt)*(U0s[2:].adjoint() - U0s[:-2].adjoint())@(U0s[:-2] + U0s[2:])/2
               # + 1j/(self.dt)*(U0s[2:].adjoint()@U0s[1:-1] - t.eye(self.NHilbert))).flip(0)
            # + 1j/(4*self.dt)*(U0s[2:].adjoint()@U0s[:-2] - U0s[:-2].adjoint()@U0s[2:])).flip(0)
            #  - 1j/(2*self.dt)*(U0s[1:].adjoint()@U0s[:-1] - U0s[:-1].adjoint()@U0s[1:])
        
        # self.U0 = U0s[1]
        # self.U0s = U0s[1:-1]
        # self.U0s = 0.5*(U0s[2:] + U0s[1:-1])
        # self.U0s = 0.5*(U0s[:-2] + U0s[2:])
        # self.U0 = self.U0s[0]
        # self.H0_term = self.U0s.adjoint()@H0@self.U0s \
        #     + 1j/(4*self.dt)*(U0s[2:].adjoint()@U0s[:-2] - U0s[:-2].adjoint()@U0s[2:])
        
        # self.basis_change = (U0s[2:,:,:self.subNHilbert].adjoint()@U0s[1:-1,:,:self.subNHilbert])

       
        # print(t.exp(-1j*angles).unsqueeze(2).repeat(1,1,21).shape,U0s[1:-1,:,:self.subNHilbert].shape)
