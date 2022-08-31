import torch as t


class Periodic_System():
    def __init__(self):
        self.T = self.params_dict['T']
        self.NTrot = self.params_dict['NTrot']
        self.q_max = self.params_dict['q_max']
        self.NHilbert = self.q_max*2 + 1

        self.times = t.linspace(0,self.T,self.NTrot)
        self.dt = (self.times[1:] - self.times[:-1]).mean().item()
        self.Id = t.eye(self.NHilbert, dtype=t.complex128)

        q = t.arange(-self.q_max,self.q_max+1,1)
        self.q_mat = t.diag(q)
        self.cos_mat = (t.diag(t.ones(self.NHilbert-1),-1) \
                        + t.diag(t.ones(self.NHilbert-1),1)).type(t.complex128)
        try:
            self.phi_ext = self.params_dict['phi_ext']
            self.cos2_mat = (t.diag(t.ones(self.NHilbert-2,dtype=t.complex128),-2)*t.exp(t.tensor(-1j*self.phi_ext)) \
                            + t.diag(t.ones(self.NHilbert-2),2)*t.exp(t.tensor(1j*self.phi_ext)))
        except:
            print("Found no phi_ext parameter, assuming it to be 0")
            self.cos2_mat = (t.diag(t.ones(self.NHilbert-2,dtype=t.complex128),-2) \
                            + t.diag(t.ones(self.NHilbert-2),2))

        ones = t.ones(self.NTrot-1)

        print("Right now the dimensions of the derivatives are not correct.")
        # self.diff = 1/(2*self.dt)*(t.diag(ones,1)+t.diag(-ones,-1))
        self.diff = 1/self.dt*(t.diag(-t.ones(self.NTrot)) + t.diag(ones,1))
        self.diff[-1,-1] = 0

        self.ddiff = 1/(self.dt**2)*(t.diag(-2*t.ones(self.NTrot)) + t.diag(ones,1) + t.diag(ones,-1))    
        # self.ddiff = t.diag(-2*t.ones(self.NTrot)) + t.diag(ones,1) + t.diag(ones,-1)
        self.ddiff[0] = 0
        self.ddiff[-1] = 0

        self.prepare_KinE()
        self.set_eig_H()
        super().__init__()

    def get_H(self,alphas=t.tensor([1]),control = t.tensor([0])):
        # assert type(alpha) == t.Tensor
        return self.KinE.repeat((alphas.shape[0],1,1)) + self.V(alphas=alphas,control=control)
    
    def set_eig_H(self,alpha = 1):
        eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=t.tensor([alpha])).squeeze())
        self.eigvals = eigvals
        self.eigvecs = eigvecs
    
    def get_occupancy(self, indices = [0,1]):
        alphas = self.activation_func(self.times)
        occ = t.zeros((len(indices)+1,self.NTrot))
        # try:
        exp_mat = self.latest_matrix_exp
        # except:
        #     print("No self.latest_matrix_exp")
        #     Hs = self.get_H(alphas.flip(0),self.get_control().flip(0))
        #     exp_mat = t.matrix_exp(-1j*Hs*self.dt)

        wavefunc = self.eigvecs[:,[0]]
        eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=alphas))
        for i, mat in enumerate(exp_mat.flip(0)):
            wavefunc = mat@wavefunc
            for j, ind in enumerate(indices):
                occ[j,i] = t.abs(eigvecs[i,:,[ind]].adjoint()@wavefunc)
                # occ[ind,i] = eigvecs[i,:,[ind]].adjoint()@wavefunc
            
        occ = t.square(occ)
        occ[len(indices)] = 1-occ.sum(0)
        return occ