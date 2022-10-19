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
        print("I added a factor half to the cosines")
        self.cos_mat = 0.5*(t.diag(t.ones(self.NHilbert-1),-1) \
                        + t.diag(t.ones(self.NHilbert-1),1)).cfloat()
        try:
            self.phi_ext = t.tensor(self.params_dict['phi_ext']).cfloat()
        except:
            print("Found no phi_ext parameter, assuming it to be 0")
            self.phi_ext = t.tensor(0).cfloat()
            # self.cos2_mat = (t.diag(t.ones(self.NHilbert-2).cfloat(),-2) \
            #                 + t.diag(t.ones(self.NHilbert-2),2))
        self.cos2_mat = 0.5*(t.diag(t.ones(self.NHilbert-2).cfloat(),-2)*t.exp(-1j*self.phi_ext) \
                            + t.diag(t.ones(self.NHilbert-2),2)*t.exp(1j*self.phi_ext))

        
        
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
        if self.params_dict['dim'] == '2d':
            self.prepare_2d()
        self.set_eig_H()

        super().__init__()
    
    def prepare_KinE(self):
        self.EJ = self.params_dict['EJ']
        if self.params_dict['dim'] == '1d':
            self.KinE = 4*self.params_dict['EC']*self.q_mat**2
            self.V = self.V_1d
        elif self.params_dict['dim'] == '2d':
            self.KinE = 4*self.params_dict['EC']*(t.kron(self.q_mat**2,self.Id) + t.kron(self.Id,self.q_mat**2))
            self.V = self.V_2d
        else:
            print(self.params_dict['dim'] + ", is not a proper input (either '1d' or '2d')")
        
    
    def set_eig_H(self,alpha = 1):
        eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=t.tensor([alpha])).squeeze())
        self.eigvals = eigvals
        self.eigvecs = eigvecs.cfloat()
    
    def prepare_2d(self):
        self.q_mat = t.kron(self.q_mat,self.Id) #+ t.kron(self.Id,self.q_mat)

        self.cos_mat = t.kron(self.cos_mat,self.Id) + t.kron(self.Id,self.cos_mat)

        upper = t.diag(t.ones(self.NHilbert-1).cfloat(),1)
        lower = t.diag(t.ones(self.NHilbert-1).cfloat(),-1)
        self.cos2_mat = 0.5*(t.exp(1j*self.phi_ext)*t.kron(lower,upper) + t.exp(-1j*self.phi_ext)*t.kron(upper,lower))
        return
    
    #Made for only ETrotter atm, un-comment stuff for QTrotter
    def get_occupancy(self, indices = [0,1], init_inds = [0]):
        # alphas = self.activation_func(self.times)
        occ = t.zeros((len(indices)+1,self.NTrot,len(init_inds)))

        # eigvals, eigvecs = t.linalg.eigh(self.get_H(alphas=alphas.detach())) #This is probably what makes alpha regression slow
        # eigvecs = eigvecs[:,:self.subNHilbert].cfloat()
        # wavefuncs = self.init_wavefuncs[:self.subNHilbert,init_inds]
        wavefuncs = self.init_wavefuncs[:,init_inds]
        # evolve = self.Id[:self.subNHilbert]

        ############TEST#######
        # print(((eigvecs[:-1].adjoint()@eigvecs[1:]).abs().diagonal(offset=0,dim1=-2,dim2=-1).sum(dim=-1)<self.NHilbert-0.1).sum())
        
        # energies = self.H0_term[:,[_ for _ in range(self.subNHilbert)],[_ for _ in range(self.subNHilbert)]].real
        # sorted_inds = energies.sort().indices
        for i, mat in enumerate(self.latest_matrix_exp):
            wavefuncs = mat@wavefuncs
            for j, ind in enumerate(indices):
                    # occ[j,i] = t.abs(eigvecs[i,:,[ind]].adjoint()@wavefunc)
                    occ[j,i] = t.abs(wavefuncs[ind])
                    # occ[j,i] = t.abs(wavefuncs[sorted_inds[i,ind]])


        # try:
        #     print("try (occupation)")
        #     init_wavefunc = self.U0s[0].adjoint()@self.eigvecs[:,[init_ind]] #self.subNHilbert
        #     U = t.eye(self.subNHilbert).type(t.cfloat)
        #     for i, mat in enumerate(exp_mat.flip(0)):
        #         # tmp = self.U0s[-i].adjoint()@self.U0
        #         # tmp = self.U0
        #         # init_wavefunc = tmp@self.eigvecs[:,[init_ind]]
                
        #         # init_wavefunc = t.zeros((1,self.NHilbert),dtype=t.cfloat)
        #         # init_wavefunc[0,0] = 1
        #         # U = self.U0s[-(i+1)].adjoint()@self.U0s[-i]@mat@U
        #         U = mat@U
        #         for j, ind in enumerate(indices):
        #             # if i == 100:
        #                 # print(t.abs((U@init_wavefunc)[ind]))
        #             # occ[j,i] = t.abs((U@init_wavefunc)[ind])
        #             occ[j,i] = t.abs((eigvecs[i,:self.subNHilbert,[ind]].adjoint()@U@init_wavefunc))
        # except:
        #     print("except (occupation)")
        #     wavefunc = self.eigvecs[:,[init_ind]]
        #     for i, mat in enumerate(exp_mat.flip(0)):
        #         wavefunc = mat@wavefunc
        #         for j, ind in enumerate(indices):
        #             occ[j,i] = t.abs(eigvecs[i,:,[ind]].adjoint()@wavefunc)
        #             # occ[ind,i] = eigvecs[i,:,[ind]].adjoint()@wavefunc
            
        occ = t.square(occ)
        occ[len(indices)] = 1-occ.sum(0)
        return occ.squeeze()
    
    # def restrict_time(self, time_point):
    #     return self.Softplus(time_point) - self.Softplus(time_point - self.T)
    
    def restrict_output(self,inp,Min,Max):
        return self.ReLU(inp - Min) - self.ReLU(inp - Max)
    
    # def custom_Hardsigmoid(self,x): #restrict linearly to [0,1]
    #     return self.Hardsigmoid(6*(x-0.5)) #Perhaps should be restricted to [0.5,1]

    def Print(self,statement):
        from time import localtime, strftime
        print("{} - {}".format(strftime("%H:%M:%S", localtime()),statement))
    
    def lin_interpolate(self,vals):
        out = t.zeros(vals.shape[0]*2-1)
        out[::2] = vals
        out[1::2] = 0.5*(vals[1:] + vals[:-1])
        return out
