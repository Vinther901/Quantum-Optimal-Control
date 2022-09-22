import torch as t
from numpy import exp



class Trainer():
    def __init__(self):
        self.N_epoch = 1
        self.lr = self.params_dict['lr']

        self.optimizer = t.optim.Adam(self.parameters(), lr=self.lr)
        #Using stochastic gradient descent is no bueno.
        # self.optimizer = t.optim.SGD(self.parameters(), lr=self.lr,momentum=0.0)
        if self.params_dict['Scheduler']:
            self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_func)
            self.step = self.step_with_scheduler
        else:
            self.step = self.step_without_scheduler

        # self.loss_funcs = [self.C1_state,self.C4,self.C5,self.C6,self.C7,self.C8]
        self.prepare_loss_funcs()

        N_loss_funcs = len(self.loss_funcs)
        self.loss_weights = t.ones(N_loss_funcs)/N_loss_funcs
        # self.loss_means = t.zeros(N_loss_funcs)
        self.loss_ratio_means = t.ones(N_loss_funcs)
        self.loss_vars = t.zeros(N_loss_funcs)

        self.stored_weights = self.loss_weights.unsqueeze(1)
        self.prepare_target_state_adj()
        self.prepare_target_gate_adj()
        super().__init__()
        self.initialize_loss_means()
        self.stored_losses = self.loss_means.unsqueeze(1)

    def prepare_target_state_adj(self):
        self.target_state_adj = self.init_wavefuncs[:self.subNHilbert,[1]].adjoint()#self.eigvecs[:,[1]].adjoint()
    
    def prepare_target_gate_adj(self):
        tmp = t.eye(self.subNHilbert,dtype=t.cfloat)
        tmp[0,0] = 0
        tmp[0,1] = 1
        tmp[1,0] = 1
        tmp[1,1] = 0
        self.target_gate_adj = tmp.adjoint()

    def C1_state(self,U):
        return 1 - t.square(t.abs(self.target_state_adj@U@self.init_wavefuncs[:self.subNHilbert,[0]])).squeeze()
    
    def C1_gate(self,U): #Perhaps do weighted sum instead of dim<subNHilbert
        # transformed = (self.eigvecs.adjoint()@U@self.eigvecs)[:self.subNHilbert,:self.subNHilbert]
        # wavefunc = U@self.init_wavefuncs[:self.subNHilbert]
        # return 1 - 1/self.subNHilbert**2*t.square(t.abs(t.trace(self.target_gate_adj@U)))
        dim = 3#self.subNHilbert
        return 1 - 1/dim**2*t.square(t.abs(t.trace(self.target_gate_adj[:dim]@U[:,:dim])))

    # def C2(self,U):
    #     return t.square(self.ascend_start - self.decline_end)
    
    def C4(self,U):
        return t.mean(t.square(self.diff@self.get_control()))
    
    def C5(self,U):
        return t.mean(t.abs(self.ddiff@self.get_control()))

    def C6(self,U):
        return t.mean(t.square(self.get_control()))

    def C7(self,U):
        self.occ = self.get_occupancy(indices=[0,1])
        return 1 - self.occ[1].mean()
    
    def C8(self,U):
        # self.occ = self.get_occupancy(indices=[0,1])
        return self.occ[2].mean()
    
    def C7_gate(self,U): #occupation[Bra,t,Ket]
        self.occ = self.get_occupancy(indices=[0,1],init_inds=[0,1])
        return 1 - self.occ[1,:,0].mean() + 1 - self.occ[0,:,1].mean()
    
    def C8_gate(self,U): #Should change 5 -> 6, consult the alpha dependent spectrum.
        #Perhaps it is enough to minimize only what is missing from the first 5/6
        #Instead of minimize every occupation from 5/6 and up.
        return self.occ[-1].mean(0).sum()
        # occ = self.get_occupancy(indices=[_ for _ in range(self.subNHilbert)],init_inds=[0,1])
        # return occ[5:-1].sum(2).sum(0).mean()
    
    def loss_func(self,U):
        self.losses = t.hstack([loss_func(U) for loss_func in self.loss_funcs])#/self.stored_losses[:,0]
        self.update_weights()
        return t.sum(self.loss_weights*self.losses)
    
    def initialize_loss_means(self):
        U  = self()
        self.loss_means = t.hstack([loss_func(U) for loss_func in self.loss_funcs]).detach()

    def update_weights(self):
        fac2 = 1/self.N_epoch
        fac1 = 1 - fac2
        loss = self.losses#.detach()

        old_lm = self.loss_means
        new_lm = fac1*old_lm + fac2*loss

        loss_ratio = loss/old_lm
        old_lrm = self.loss_ratio_means.detach()
        new_lrm = fac1*old_lrm + fac2*loss_ratio

        new_var = fac1*self.loss_vars.detach() + fac2*(loss_ratio - old_lrm)*(loss_ratio - new_lrm)
        
        self.loss_means = new_lm.detach()
        self.loss_ratio_means = new_lrm
        self.loss_vars = new_var
        
        # cs = t.sqrt(self.loss_vars + 10**(-5-self.N_epoch))/self.loss_ratio_means
        cs = self.loss_ratio_means/t.sqrt(self.loss_vars + 10**(-5-self.N_epoch))
        self.loss_weights = cs/t.sum(cs)#/old_lm

        # w = 10**(-100/self.N_epoch)
        # self.loss_weights = w*self.loss_weights + (1-w)*self.stored_weights[:,0]

        self.stored_weights = t.cat([self.stored_weights,self.loss_weights.detach().unsqueeze(1)],dim=1)#[:500]
        self.stored_losses = t.cat([self.stored_losses,self.losses.detach().unsqueeze(1)],dim=1)

        w = 10**(-1000/self.N_epoch)
        self.loss_weights = w*self.loss_weights + (1-w)*self.stored_weights[:,0]


    def step_with_scheduler(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
    
    def step_without_scheduler(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def lr_func(self,epoch):
        # return np.log(epoch+1)
        # tmp = epoch - 4
        # return max(0.005,1*(epoch-4)/(300 + (epoch-4)))
        return 0.9/(1+((epoch-15)/5)**2) + 0.1
    
    def minimize(self, threshold, max_steps):
        from time import time
        max_steps += self.N_epoch
        self.optimizer.zero_grad()
        loss = self.loss_func(self())

        print(f"initial loss: {loss.item()}")

        start_time = time()
        while loss >= threshold and self.N_epoch <= max_steps:
            loss.backward()
            self.step()
            loss = self.loss_func(self())
            self.N_epoch += 1
            print(f"loss: {loss.item()}, step: {self.N_epoch}", end='\r')
        
        print(f"Ended at step: {self.N_epoch}, with loss: {loss.item()} and runtime: {time() - start_time}")

    def prepare_loss_funcs(self):
        string = "[self." + ", self.".join(self.params_dict['loss_funcs']) + "]"
        exec("self.loss_funcs = " + string)
    