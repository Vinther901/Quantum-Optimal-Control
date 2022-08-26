import torch as t



class Trainer():
    def __init__(self):
        self.N_epoch = 0
        self.lr = self.params_dict['lr']

        self.optimizer = t.optim.Adam(self.parameters(), lr=self.lr)
        if self.params_dict['Scheduler']:
            self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer,self.lr_func)
            self.step = self.step_with_scheduler
        else:
            self.step = self.step_without_scheduler
        self.prepare_target_state_adj()
        super().__init__()

    # def prepare_loss_func(self):
    #     #loss_funcs = [self.C1,self.C2,self.C3,..]
    #     for i in range(len(loss_funcs))

    def prepare_target_state_adj(self):
        self.target_state_adj = self.eigvecs[:,[1]].adjoint()

    def C1_state(self,U):
        return 1 - t.square(t.abs(self.target_state_adj@U@self.eigvecs[:,[0]]))

    def loss_func(self,U):
        return self.C1_state(U)

    def step_with_scheduler(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
    
    def step_without_scheduler(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def lr_func(self,epoch):
        # return np.log(epoch+1)
        return 10*epoch/(300 + epoch)
    
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

    