import torch as t


class QTrotter():
    def __init__(self):
        super().__init__()
    
    def get_H(self,alphas=t.tensor([1]),control = t.tensor([0])):
        # assert type(alpha) == t.Tensor
        return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)
    

class ETrotter():
    def __init__(self):
        H0 = self.KinE.repeat((self.NTrot,1,1)) + self.V(alphas=self.activation_func(self.times),control=t.zeros(self.NTrot))
        E0, U0s = t.linalg.eigh(H0)
        U0s = t.concat([U0s[[0]],U0s,U0s[[-1]]],0)
        # self.U0dot = 1j/(2*self.dt)*(U0s[2:].adjoint() - U0s[:-2].adjoint())@U0s[1:-1]
        self.U0dot = 1/(self.dt)*(-0.5*U0s[2:]-1.5*U0s[:-2]+2*U0s[1:-1]).adjoint()@U0s[1:-1]
        self.H0_term = t.diag_embed(E0).type(t.cfloat) \
            + 1j*self.U0dot
            # + 1j/(self.dt)*(U0s[2:].adjoint()@U0s[1:-1] - t.eye(self.NHilbert))).flip(0)
            # + 1j/(4*self.dt)*(U0s[2:].adjoint()@U0s[:-2] - U0s[:-2].adjoint()@U0s[2:])).flip(0)
            #  - 1j/(2*self.dt)*(U0s[1:].adjoint()@U0s[:-1] - U0s[:-1].adjoint()@U0s[1:])
        
        self.U0s = U0s[1:-1]
        self.U0 = self.U0s[0]
        # self.U0s = 0.5*(U0s[:-2] + U0s[2:])
        # self.H0_term = self.U0s.adjoint()@H0@self.U0s \
        #     + 1j/(4*self.dt)*(U0s[2:].adjoint()@U0s[:-2] - U0s[:-2].adjoint()@U0s[2:])
        
        self.U0s = self.U0s.flip(0)[:,:,:self.subNHilbert]
        self.H0_term = self.H0_term.flip(0)[:,:self.subNHilbert,:self.subNHilbert]
        super().__init__()

    def get_H(self,alphas=t.tensor([1]), control=t.tensor([0])):
        if alphas.shape[0] > 1:
            V = control.view(-1,1,1)*self.q_mat
            return self.H0_term + self.U0s.adjoint()@V@self.U0s
        else:
            return t.diag(t.linalg.eigvalsh(self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)).squeeze())
            # return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)

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
            alphas = self.activation_func(self.times).flip(0)
            pulse = self.get_control().flip(0)
            # M2s = self.t1*t.pow(self.t2,pulse.view(-1,1))
            # matrices = t.zeros((2*self.NTrot,self.NHilbert,self.NHilbert)).type(t.complex128)
            # matrices[0::2] = t.diag_embed(self.t1*t.pow(self.t2,pulse.view(-1,1)))
            # matrices[1::2] = self.t3@t.diag_embed(t.pow(self.t4,alphas.view(-1,1)))@self.u4_adj
            M1s = t.diag_embed(self.t1*t.pow(self.t2,pulse.view(-1,1)))
            M2s = self.t3@t.diag_embed(t.pow(self.t4,alphas.view(-1,1)))@self.u4_adj
            return M1s@M2s
        else:
            return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)

