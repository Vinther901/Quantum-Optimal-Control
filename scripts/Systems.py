import torch as t


class Periodic_System():
    def __init__(self):
        # self.device = t.device('cuda')

        self.ReLU = t.nn.ReLU()
        self.T = self.params_dict['T']
        self.NTrot = self.params_dict['NTrot']
        self.q_max = self.params_dict['q_max']
        self.NHilbert = self.q_max*2 + 1
        self.subNHilbert = self.params_dict['subNHilbert']

        self.times = t.linspace(0,self.T,self.NTrot)
        self.dt = (self.times[1:] - self.times[:-1]).mean().item()
        self.Id = t.eye(self.NHilbert).cfloat()

        q = t.arange(-self.q_max,self.q_max+1,1)
        self.q_mat = t.diag(q).cfloat()
        self.cos_mat = (t.diag(t.ones(self.NHilbert-1),-1) \
                        + t.diag(t.ones(self.NHilbert-1),1)).cfloat()
        try:
            self.phi_ext = self.params_dict['phi_ext']
            self.cos2_mat = (t.diag(t.ones(self.NHilbert-2).cfloat(),-2)*t.exp(t.tensor(-1j*self.phi_ext)) \
                            + t.diag(t.ones(self.NHilbert-2),2)*t.exp(t.tensor(1j*self.phi_ext)))
        except:
            print("Found no phi_ext parameter, assuming it to be 0")
            self.cos2_mat = (t.diag(t.ones(self.NHilbert-2).cfloat(),-2) \
                            + t.diag(t.ones(self.NHilbert-2),2))

        ones = t.ones(self.NTrot-1)

        # print("Right now the dimensions of the derivatives are not correct.")
        # self.diff = 1/(2*self.dt)*(t.diag(ones,1)+t.diag(-ones,-1))
        self.diff = 1/self.dt*(t.diag(-t.ones(self.NTrot)) + t.diag(ones,1))
        self.diff[-1,-1] = 0

        self.ddiff = 1/(self.dt**2)*(t.diag(-2*t.ones(self.NTrot)) + t.diag(ones,1) + t.diag(ones,-1))    
        # self.ddiff = t.diag(-2*t.ones(self.NTrot)) + t.diag(ones,1) + t.diag(ones,-1)
        self.ddiff[0] = 0
        self.ddiff[-1] = 0

        self.prepare_KinE()
        self.set_eig_H()
        ####################################################EXPERIMENTAL#
        # self.t1 = t.exp(-1j*self.dt*4*self.params_dict['EC']*q**2)
        # self.t2 = t.exp(-1j*self.dt*self.EJ*q)
        # t3 = t.matrix_exp(1j*self.dt*self.EJ*self.cos_mat).cfloat()
        # self.t4, u4 = t.linalg.eig(t.matrix_exp(-1j*self.dt*self.EJ*self.cos2_mat).cfloat())
        # self.t3 = t3@u4
        # self.u4_adj = u4.adjoint().cfloat()

        # self.t1 = self.t1.to(self.device)
        # self.t2 = self.t2.to(self.device)
        # self.t3 = self.t3.to(self.device)
        # self.t4 = self.t4.to(self.device)
        # self.u4_adj = self.u4_adj.to(self.device)

        #################################################################
        super().__init__()
    
    def prepare_KinE(self):
        self.KinE = 4*self.params_dict['EC']*self.q_mat**2
        self.EJ = self.params_dict['EJ']
    
    def set_eig_H(self,alpha = 1):
        eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=t.tensor([alpha])).squeeze())
        self.eigvals = eigvals
        self.eigvecs = eigvecs.cfloat()
    
    def get_occupancy(self, indices = [0,1], init_ind = 0):
        alphas = self.activation_func(self.times)
        occ = t.zeros((len(indices)+1,self.NTrot))
        exp_mat = self.latest_matrix_exp
        #BAAAAAAAAAAAAAAAAAAD
        eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=alphas.detach())) #This is probably what makes alpha regression slow
        eigvecs = eigvecs.cfloat()
        try:
            print("try (occupation)")
            U = t.eye(self.subNHilbert).type(t.cfloat)
            for i, mat in enumerate(exp_mat.flip(0)):
                # tmp = self.U0s[-i].adjoint()@self.U0
                # tmp = self.U0
                # init_wavefunc = tmp@self.eigvecs[:,[init_ind]]
                init_wavefunc = self.eigvecs[:self.subNHilbert,[init_ind]]
                # init_wavefunc = t.zeros((1,self.NHilbert),dtype=t.cfloat)
                # init_wavefunc[0,0] = 1
                U = self.U0s[-(i+1)].adjoint()@self.U0s[-i]@mat@U
                for j, ind in enumerate(indices):
                    # if i == 100:
                        # print(t.abs((U@init_wavefunc)[ind]))
                    occ[j,i] = t.abs((U@init_wavefunc)[ind])
        except:
            print("except (occupation)")
            wavefunc = self.eigvecs[:,[init_ind]]
            for i, mat in enumerate(exp_mat.flip(0)):
                wavefunc = mat@wavefunc
                for j, ind in enumerate(indices):
                    occ[j,i] = t.abs(eigvecs[i,:,[ind]].adjoint()@wavefunc)
                    # occ[ind,i] = eigvecs[i,:,[ind]].adjoint()@wavefunc
            
        occ = t.square(occ)
        occ[len(indices)] = 1-occ.sum(0)
        return occ
    
    # def restrict_time(self, time_point):
    #     return self.Softplus(time_point) - self.Softplus(time_point - self.T)
    
    def restrict_output(self,inp,Min,Max):
        return self.ReLU(inp - Min) - self.ReLU(inp - Max)
    
    # def custom_Hardsigmoid(self,x): #restrict linearly to [0,1]
    #     return self.Hardsigmoid(6*(x-0.5)) #Perhaps should be restricted to [0.5,1]
