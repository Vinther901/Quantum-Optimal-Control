import torch as t

class RampDownUpPulse():
    def __init__(self):
        self.ReLU = t.nn.ReLU()
        self.Softplus = t.nn.Softplus(beta=4)
        self.Sigmoid = t.nn.Sigmoid()
        self.Hardsigmoid = t.nn.Hardsigmoid()

        self.decline_end = t.nn.parameter.Parameter(t.tensor(self.params_dict['decline_end']))
        self.ascend_start = t.nn.parameter.Parameter(t.tensor(self.params_dict['ascend_start']))
        self.level = t.nn.parameter.Parameter(t.tensor(self.params_dict['level']))
        self.envelope_amp = t.nn.parameter.Parameter(t.tensor(self.params_dict['envelope_amp']))
        self.detuning = t.nn.parameter.Parameter(t.tensor(self.params_dict['detuning']))
        self.phase = t.nn.parameter.Parameter(t.tensor(self.params_dict['phase']))
        super().__init__()
    
    def get_control(self):
        eigvals = t.linalg.eigvalsh(self.get_H(self.level.view(1)).squeeze())
        omega_d = eigvals[1] - eigvals[0]
        pulse = t.cos(self.detuning*omega_d*(self.times - self.decline_end - self.phase))
        envelope = self.envelope_func()
        return envelope*pulse
    
    def envelope_func(self):
        return self.envelope_amp*self.custom_Sigmoid(self.times-self.decline_end)*self.custom_Sigmoid(self.ascend_start - self.times)

    def custom_Sigmoid(self,x):
        return self.Sigmoid(5*t.pi/4*(x-1))
    
    def custom_Hardsigmoid(self,x): #restrict linearly to [0,1]
        return self.Hardsigmoid(6*(x-0.5)) #Perhaps should be restricted to [0.5,1]

    def activation_func(self,time):
        decline_end = self.restrict_time(self.decline_end)
        ascend_start = self.restrict_time(self.ascend_start)
        level = self.custom_Hardsigmoid(self.level)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level
    
    def restrict_time(self, time_point):
        return self.Softplus(time_point) - self.Softplus(time_point - self.T)

class FreePulse():
    def __init__(self):
        self.pulse = t.nn.parameter.Parameter(self.init_pulse())
        self.alphas = t.ones(self.NTrot)
        super().__init__()
    
    def init_pulse(self):
        # return t.ones(self.NTrot)
        return 1.5*t.pi/(2*self.T)*t.sin((self.eigvals[1] - self.eigvals[0])*self.times-0.1*t.pi)
    
    def get_control(self):
        return self.pulse
    
    def activation_func(self,time):
        return self.alphas
    
    def envelope_func(self):
        return t.zeros(self.NTrot)