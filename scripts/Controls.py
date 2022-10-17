import torch as t

class RampDownUpPulse():
    def __init__(self):
        self.Sigmoid = t.nn.Sigmoid()
        # self.decline_end = t.nn.parameter.Parameter(t.tensor(self.params_dict['decline_end']))
        # self.ascend_start = t.nn.parameter.Parameter(t.tensor(self.params_dict['ascend_start']))
        # self.level = t.nn.parameter.Parameter(t.tensor(self.params_dict['level']))
        self.decline_end = t.tensor(self.params_dict['decline_end'])
        self.ascend_start = t.tensor(self.params_dict['ascend_start'])
        self.level = t.tensor(self.params_dict['level'])

        self.envelope_amp = t.nn.parameter.Parameter(t.tensor(self.params_dict['envelope_amp']))
        self.detuning = t.nn.parameter.Parameter(t.tensor(self.params_dict['detuning']))
        self.phase = t.nn.parameter.Parameter(t.tensor(self.params_dict['phase']))
        super().__init__()
    
    def get_control(self):
        eigvals = t.linalg.eigvalsh(self.get_H(self.level.view(1)).squeeze())
        omega_d = (eigvals[1] - eigvals[0])
        pulse = t.cos(self.detuning*omega_d*(self.times - self.decline_end - self.phase))
        envelope = self.envelope_func()
        return envelope*pulse
    
    def envelope_func(self):
        return self.envelope_amp*self.custom_Sigmoid(self.times-self.decline_end)*self.custom_Sigmoid(self.ascend_start - self.times)

    def custom_Sigmoid(self,x):
        return self.Sigmoid(5*t.pi/4*(x-1))

    def activation_func(self,time):
        decline_end = self.restrict_output(self.decline_end,0,self.T)
        ascend_start = self.restrict_output(self.ascend_start,0,self.T)
        level = self.restrict_output(self.level,0,1)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level
    
    def init_activation_func(self,time):
        decline_end = self.restrict_output(t.tensor(self.params_dict['decline_end']),0,self.T)
        ascend_start = self.restrict_output(t.tensor(self.params_dict['ascend_start']),0,self.T)
        level = self.restrict_output(t.tensor(self.params_dict['level']),0,1)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level

    
    def get_init_pulse(self):
        decline_end = t.tensor(self.params_dict['decline_end'])
        ascend_start = t.tensor(self.params_dict['ascend_start'])
        level = t.tensor(self.params_dict['level'])
        envelope_amp =t.tensor(self.params_dict['envelope_amp'])
        detuning = t.tensor(self.params_dict['detuning'])
        phase = t.tensor(self.params_dict['phase'])
        envelope = envelope_amp*self.custom_Sigmoid(self.times-decline_end)*self.custom_Sigmoid(ascend_start - self.times)
        eigvals = t.linalg.eigvalsh(self.get_H(level.view(1)).squeeze())
        omega_d = (eigvals[1] - eigvals[0])
        pulse = t.cos(detuning*omega_d*(self.times - decline_end - phase))
        return envelope*pulse

    def restrict_time(self, time_point):
        return self.Softplus(time_point) - self.Softplus(time_point - self.T)

class CauchyPulse():
    def __init__(self):
        # self.decline_end = t.nn.parameter.Parameter(t.tensor(self.params_dict['decline_end']))
        # self.ascend_start = t.nn.parameter.Parameter(t.tensor(self.params_dict['ascend_start']))
        # self.level = t.nn.parameter.Parameter(t.tensor(self.params_dict['level']))
        self.alphas = self.init_activation_func(self.times)

        self.heights = t.nn.parameter.Parameter(self.init_heights())#.to(self.device)
        # self.heights = self.init_heights()
        
        sqrd_dists = (self.times.view(-1,1) - self.times.view(1,-1))**2/0.001
        self.height_weights = (1/(1+sqrd_dists)/t.sum(1/(1+sqrd_dists),1))
        super().__init__()
    
    def get_init_pulse(self):
        return t.sum(self.init_heights()*self.height_weights,1)
    
    def get_control(self):
        # return t.cumsum(self.restrict_diff(),0)
        return t.sum(self.heights*self.height_weights,1)
    
    def init_heights(self):
        # return 0.5*t.exp(-(self.times - self.T/2)**2/20)
        return 0.05*t.exp(-(self.times - self.T/2)**2/10)*t.sin(10*self.times)
    
    def activation_func(self,time):
        # decline_end = self.restrict_output(self.decline_end,0,self.T)
        # ascend_start = self.restrict_output(self.ascend_start,0,self.T)
        # level = self.restrict_output(self.level,0,1)

        # left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        # right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        # return left_slope + right_slope + level
        return self.alphas
    
    def init_activation_func(self,time):
        decline_end = self.restrict_output(t.tensor(self.params_dict['decline_end']),0,self.T)
        ascend_start = self.restrict_output(t.tensor(self.params_dict['ascend_start']),0,self.T)
        level = self.restrict_output(t.tensor(self.params_dict['level']),0,1)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level
        # return self.alphas
    
    def envelope_func(self):
        return t.zeros(self.NTrot)

class ConstrainedPulse():
    def __init__(self):
        # self.decline_end = t.nn.parameter.Parameter(t.tensor(self.params_dict['decline_end']))
        # self.ascend_start = t.nn.parameter.Parameter(t.tensor(self.params_dict['ascend_start']))
        # self.level = t.nn.parameter.Parameter(t.tensor(self.params_dict['level']))
        # self.alphas = self.init_activation_func(self.times)

        self.heights = t.nn.parameter.Parameter(self.init_heights())#.to(self.device)
        # heights = self.init_heights()
        # self.diffs = t.nn.parameter.Parameter(heights[1:] - heights[:-1])
        # self.max_diff = 0.005
        super().__init__()

    # # def restrict_diff(self,diff):
    # def restrict_diff(self,heights):
    #     max_diff = 0.005
    #     # scaling = 0.001
    #     # constrained = self.restrict_output(scaling*diff,0,max_diff) + self.restrict_output(scaling*diff,-max_diff,0)-max_diff
    #     diff = heights[1:] - heights[:-1]
    #     constrained = self.restrict_output(diff,0,max_diff) + self.restrict_output(diff,-max_diff,0)-max_diff

    #     return t.concat([heights[[0]],constrained])
    #     # return t.concat([t.tensor([0.]),constrained])
    
    def restrict_diff(self,heights):
        max_diff = 0.005
        out = t.zeros(heights.shape[0])
        out[0] = heights[0]
        for i in range(heights.shape[0]-1):
            diff = heights[i+1] - out[i]
            out[i+1] = out[i] + self.restrict_output(diff,0,max_diff) + self.restrict_output(diff,-max_diff,0)-max_diff
        return out

    def get_init_pulse(self):
        heights = self.init_heights()
        # return t.cumsum(self.restrict_diff(heights[1:] - heights[:-1]),0)
        # return t.cumsum(self.restrict_diff(heights),0)
        return self.restrict_diff(heights)
    
    def get_control(self):
        # return t.cumsum(self.restrict_diff(self.heights),0)
        # return t.cumsum(self.restrict_diff(self.diffs),0)
        return self.restrict_diff(self.heights)
    
    def init_heights(self):
        # return 0.5*t.exp(-(self.times - self.T/2)**2/20)
        # return 0.0*t.exp(-(self.times - self.T/2)**2/10)*t.sin(5*self.times)
        # return 0.0*t.sin(1*self.times)*t.exp(-(self.times - self.T/2)**2/100)
        return 0.*t.sin(2*t.pi/30*self.times)*t.cos(5*2*t.pi/30*self.times) #0.1
    
    def activation_func(self,time):
        # decline_end = self.restrict_output(self.decline_end,0,self.T)
        # ascend_start = self.restrict_output(self.ascend_start,0,self.T)
        # level = self.restrict_output(self.level,0,1)

        # left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        # right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        # return left_slope + right_slope + level
        return self.alphas
    
    def init_activation_func(self,time):
        decline_end = self.restrict_output(t.tensor(self.params_dict['decline_end']),0,self.T)
        ascend_start = self.restrict_output(t.tensor(self.params_dict['ascend_start']),0,self.T)
        level = self.restrict_output(t.tensor(self.params_dict['level']),0,1)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level
        # alphas = t.sin(time)*t.exp(-(time-15)**2/10) + t.exp(-time/3) + t.exp(-(30 - time)/3)
        # alphas = alphas - alphas.min()
        # alphas = alphas/alphas.max()/2
        # alphas = alphas + 0.5
        # print("Setting self.alphas")
        # self.alphas = alphas
        # return alphas
        # return self.alphas
        # return time*(time-self.T)/500+1
    
    def envelope_func(self):
        return t.zeros(self.NTrot)

class ConstrainedAlpha():
    def __init__(self):
        self.alphas = t.nn.parameter.Parameter(self.init_activation_func(self.times[1:-1]))
        super().__init__()
    
    def restrict_diff(self,heights):
        max_diff = 0.01
        out = t.zeros(heights.shape[0])
        out[0] = heights[0]
        for i in range(heights.shape[0]-1):
            diff = heights[i+1] - out[i]
            out[i+1] = out[i] + self.restrict_output(diff,0,max_diff) + self.restrict_output(diff,-max_diff,0)-max_diff
        return out
    
    def get_init_pulse(self):
        return self.init_heights()

    def get_control(self):
        return self.init_heights()

    def init_heights(self):
        return t.zeros_like(self.times)
    
    def activation_func(self,time):
        # return self.restrict_output(self.restrict_diff(self.alphas),0.5,1) + 0.5
        one = t.tensor([1.])
        return t.concat([one,self.restrict_output(self.alphas,0.5,1)+0.5,one])
    
    def init_activation_func(self,time):
        decline_end = self.restrict_output(t.tensor(self.params_dict['decline_end']),0,self.T)
        ascend_start = self.restrict_output(t.tensor(self.params_dict['ascend_start']),0,self.T)
        level = self.restrict_output(t.tensor(self.params_dict['level']),0,1)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level

        # alphas = t.sin(time)*t.exp(-(time-15)**2/10) + t.exp(-time/3) + t.exp(-(30 - time)/3)
        # alphas = alphas - alphas.min()
        # alphas = alphas/alphas.max()/2
        # alphas = alphas + 0.5
        # return self.restrict_output(self.restrict_diff(alphas),0.5,1) + 0.5

        # return time*(time-self.T)/500+1
    
    def envelope_func(self):
        return t.zeros(self.NTrot)

class FourierPulse():
    def __init__(self):
        # self.decline_end = t.nn.parameter.Parameter(t.tensor(self.params_dict['decline_end']))
        # self.ascend_start = t.nn.parameter.Parameter(t.tensor(self.params_dict['ascend_start']))
        # self.level = t.nn.parameter.Parameter(t.tensor(self.params_dict['level']))
        self.alphas = self.init_activation_func(self.times)

        amps, freqs, phases = self.init()
        self.amps = t.nn.parameter.Parameter(amps)
        self.freqs = t.nn.parameter.Parameter(freqs)
        self.phases = t.nn.parameter.Parameter(phases)
        super().__init__()
    
    def get_init_pulse(self):
        amps, freqs, phases = self.init()
        return (amps*t.sin(freqs*self.times + phases)).mean(0)
    
    def get_control(self):
        return (self.amps*t.sin(self.freqs*self.times + self.phases)).mean(0)
    
    def init(self):
        N = 5
        # return t.zeros((N,1)), t.zeros((N,1)), t.zeros((N,1))
        return t.rand((N,1))*0.1, t.rand((N,1))*10, t.rand((N,1))
    
    def activation_func(self,time):
        # decline_end = self.restrict_output(self.decline_end,0,self.T)
        # ascend_start = self.restrict_output(self.ascend_start,0,self.T)
        # level = self.restrict_output(self.level,0,1)

        # left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        # right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        # return left_slope + right_slope + level
        return self.alphas
    
    def init_activation_func(self,time):
        decline_end = self.restrict_output(t.tensor(self.params_dict['decline_end']),0,self.T)
        ascend_start = self.restrict_output(t.tensor(self.params_dict['ascend_start']),0,self.T)
        level = self.restrict_output(t.tensor(self.params_dict['level']),0,1)

        left_slope = self.ReLU(1-level - (1-level)/decline_end*time)
        right_slope = self.ReLU((1-level)/(self.T - ascend_start)*(time - ascend_start))
        return left_slope + right_slope + level
        # return self.alphas
    
    def envelope_func(self):
        return t.zeros(self.NTrot)

class FreePulse():
    def __init__(self):
        self.pulse = t.nn.parameter.Parameter(self.get_init_pulse())
        self.alphas = t.ones(self.NTrot)
        super().__init__()
    
    def get_init_pulse(self):
        # return t.ones(self.NTrot)
        # return 5*t.pi/(2*self.T)*t.sin((self.eigvals[1] - self.eigvals[0])*self.times-0.1*t.pi)**2
        # return 0.1*t.exp(-(self.times - self.T/2)**2/20)
        return  0.1*t.exp(-(self.times - self.T/2)**2/10)*t.sin(17.6*self.times)

        # return t.tensor([ 4.1660e-04, -2.4065e-03, -1.5939e-03,  9.2546e-04,  1.3162e-04,
        # -1.8205e-03, -1.6280e-04,  2.7743e-03,  1.7387e-03, -6.9497e-04,
        #  4.3951e-04,  2.8324e-03,  1.2600e-03, -1.1482e-03, -9.8995e-04,
        # -1.5482e-04, -8.1502e-04, -1.0501e-03,  2.9334e-04,  1.5840e-03,
        #  2.2249e-03,  3.5546e-03,  4.9430e-03,  3.4209e-03, -8.9537e-04,
        # -2.4358e-03, -4.0107e-04, -3.4498e-04, -1.8031e-03,  9.4654e-04,
        #  5.5067e-03,  5.5429e-03,  2.0506e-03, -4.2348e-04, -1.0100e-03,
        # -2.8407e-03, -5.6696e-03, -5.6136e-03, -2.1391e-03,  4.0869e-04,
        #  2.0274e-03,  3.8396e-03,  3.3357e-03,  8.6723e-04, -1.6108e-03,
        # -4.5333e-03, -7.9088e-03, -9.5082e-03, -7.1917e-03, -1.7542e-03,
        #  3.2780e-03,  6.0944e-03,  7.9030e-03,  7.4403e-03,  4.0438e-03,
        # -6.9613e-04, -4.7044e-03, -6.8058e-03, -7.1500e-03, -4.6963e-03,
        #  1.8001e-03,  8.6257e-03,  1.1450e-02,  9.9057e-03,  7.9144e-03,
        #  6.4559e-03,  1.8643e-03, -3.9861e-03, -5.9761e-03, -4.3549e-03,
        # -7.8004e-04,  2.2714e-03,  2.6930e-03,  1.5199e-03,  1.9968e-03,
        #  2.2908e-03, -3.4448e-03, -1.0942e-02, -1.2926e-02, -9.9628e-03,
        # -5.2865e-03, -1.8407e-03, -1.9652e-03, -2.4781e-03, -2.1924e-03,
        # -1.5152e-03, -2.3268e-03, -4.8547e-03, -6.7747e-03, -5.1812e-03,
        #  1.1335e-03,  6.4743e-03,  6.1383e-03,  3.9222e-03,  1.6506e-03,
        # -1.0477e-03, -2.0159e-03, -1.9863e-03, -3.7878e-03, -6.4675e-03,
        # -5.4034e-03,  5.6014e-04,  6.7849e-03,  6.4178e-03, -2.6657e-04,
        # -2.6842e-03,  2.7623e-03,  5.4407e-03,  2.1916e-03, -3.0257e-03,
        # -5.5647e-03, -4.3937e-03, -3.1287e-03, -6.5625e-03, -1.1306e-02,
        # -1.3444e-02, -1.1759e-02, -7.8900e-03, -3.5553e-03,  2.2733e-04,
        #  5.6978e-03,  1.1575e-02,  9.8092e-03,  4.4550e-03,  4.8523e-03,
        #  4.4216e-03, -2.9966e-03, -9.7389e-03, -8.9071e-03, -3.2890e-03,
        #  4.5425e-04,  3.9354e-04, -2.1177e-03, -3.8657e-03, -1.7723e-03,
        #  4.3036e-03,  9.7081e-03,  8.3266e-03,  1.2624e-03,  1.3450e-03,
        #  9.0314e-03,  1.0724e-02,  3.4326e-03, -3.1659e-03, -2.7428e-03,
        #  2.9409e-04,  8.3141e-04,  4.2717e-04,  5.6753e-04,  3.2590e-04,
        #  1.1267e-03,  6.7437e-04, -2.9192e-03, -7.1475e-03, -6.9642e-03,
        # -4.1493e-03, -3.8626e-03, -5.8179e-03, -6.1432e-03, -3.6555e-03,
        # -1.9716e-03, -3.4833e-03, -6.5574e-03, -8.7810e-03, -9.6680e-03,
        # -6.0306e-03,  3.7256e-03,  1.0994e-02,  8.7535e-03,  3.8336e-03,
        #  4.1219e-03,  6.1431e-03,  4.6070e-03,  4.8282e-05, -3.7087e-03,
        # -2.7955e-03,  8.7621e-04,  3.3048e-03,  2.1624e-03, -1.1780e-03,
        # -3.8955e-03, -3.2663e-03, -1.1194e-03,  4.1661e-04,  3.0578e-03,
        #  8.3171e-03,  9.8062e-03,  7.4004e-03,  6.7275e-03,  5.1340e-03,
        # -8.0823e-04, -6.1217e-03, -8.5215e-03, -9.0530e-03, -3.6659e-03,
        #  6.3641e-03,  1.3979e-02,  1.2329e-02,  4.3531e-03,  2.7938e-04,
        #  2.0631e-03,  3.1247e-03, -1.5672e-03, -6.8892e-03, -5.1192e-03,
        #  2.3696e-03,  5.8445e-03,  2.3956e-03, -3.0587e-03, -6.3182e-03,
        # -6.9801e-03, -3.2651e-03,  1.3712e-03,  1.5604e-03,  2.2072e-03,
        #  9.6071e-03,  1.8664e-02,  2.1507e-02,  1.7456e-02,  9.1067e-03,
        #  2.0832e-03,  2.0307e-03,  6.5179e-03,  5.9155e-03,  1.1440e-03,
        #  1.3219e-03,  5.3510e-03,  9.1483e-03,  7.6727e-03, -2.6475e-03,
        # -1.4374e-02, -1.7296e-02, -1.4814e-02, -1.5886e-02, -1.9765e-02,
        # -2.3372e-02, -2.4516e-02, -1.8627e-02, -9.0334e-03, -3.0748e-03,
        #  1.3728e-05,  1.6708e-03,  6.6662e-03,  1.1402e-02,  8.5627e-03,
        #  4.2385e-03,  1.9049e-03, -5.3504e-03, -1.1739e-02, -4.7057e-03,
        #  9.1196e-03,  1.6678e-02,  1.9620e-02,  2.7463e-02,  3.6445e-02,
        #  3.4824e-02,  2.8101e-02,  2.3047e-02,  1.8332e-02,  1.5521e-02,
        #  1.4872e-02,  1.3099e-02,  1.4478e-02,  1.5845e-02,  1.4645e-02,
        #  1.6248e-02,  1.5183e-02,  4.5299e-03, -6.1680e-03, -7.7634e-03,
        # -2.5357e-03,  2.2403e-03,  5.2255e-03,  6.6995e-03,  4.5837e-03,
        #  3.3292e-03,  8.6351e-04, -5.6747e-03, -1.2128e-02, -1.2865e-02,
        # -8.1407e-03, -4.3435e-03, -8.6727e-04,  6.7736e-03,  1.5232e-02,
        #  2.1417e-02,  2.6800e-02,  2.7156e-02,  1.6561e-02,  4.4507e-03,
        #  4.8346e-03,  1.3073e-02,  1.1047e-02, -4.4174e-03, -1.1943e-02,
        # -1.3829e-03,  8.8525e-03,  5.3278e-03, -3.2497e-03, -8.7951e-03,
        # -8.6378e-03, -3.1060e-03,  9.6443e-04,  1.2236e-03,  3.3405e-03,
        #  5.9293e-03,  3.7127e-03,  1.4857e-03,  2.2786e-03,  1.8898e-03,
        # -1.7023e-04, -3.1649e-03, -8.2299e-03, -1.1910e-02, -1.2680e-02,
        # -1.1367e-02, -8.2750e-03, -7.5756e-03, -1.0038e-02, -8.4388e-03,
        # -2.7432e-03, -2.3500e-04, -2.8598e-04,  5.4566e-03,  1.3661e-02,
        #  1.1853e-02,  7.8081e-04, -7.7821e-03, -7.2256e-03, -3.0374e-03,
        # -5.4491e-03, -8.6414e-03, -1.1773e-03,  8.2905e-03,  7.0572e-03,
        #  3.4829e-03,  6.4351e-03,  6.5669e-03, -2.5454e-03, -9.4217e-03,
        # -3.9690e-03,  5.2549e-03,  7.8276e-03,  5.8042e-03,  6.5863e-03,
        #  7.7807e-03,  5.6660e-03,  1.8561e-03, -4.8591e-03, -1.1914e-02,
        # -1.5486e-02, -1.3534e-02, -8.2080e-03,  2.1343e-04,  6.9449e-03,
        #  3.7090e-03, -5.7195e-03, -7.6749e-03, -2.8731e-03, -4.7155e-03,
        # -1.2434e-02, -9.0978e-03,  8.0119e-03,  2.1242e-02,  1.9638e-02,
        #  8.6673e-03, -6.1709e-04, -6.0440e-03, -1.1722e-02, -2.0627e-02,
        # -3.1244e-02, -3.2716e-02, -1.9001e-02, -1.3284e-03,  1.3051e-02,
        #  2.3262e-02,  2.9610e-02,  3.2668e-02,  2.8988e-02,  2.0341e-02,
        #  1.2753e-02,  4.7608e-03, -5.6621e-03, -1.6407e-02, -2.1448e-02,
        # -1.8292e-02, -7.3504e-03,  5.3872e-03,  5.3956e-03, -3.1328e-03,
        # -2.2284e-03,  5.9846e-03,  5.8009e-03, -3.7591e-03, -1.1563e-02,
        # -1.3053e-02, -1.2173e-02, -7.4989e-03,  3.5805e-03,  1.7035e-02,
        #  2.3847e-02,  1.9204e-02,  1.6659e-02,  2.6519e-02,  2.7870e-02,
        #  9.0001e-03, -9.5531e-03, -9.9421e-03, -6.9960e-03, -7.8153e-03,
        # -7.0314e-03, -4.2752e-03, -4.2786e-04,  4.2293e-03,  9.8474e-03,
        #  1.5192e-02,  1.5952e-02,  5.2942e-03, -9.3836e-03, -9.3095e-03,
        #  6.0004e-03,  1.9813e-02,  3.6631e-02,  6.0904e-02,  7.4429e-02,
        #  4.8255e-02, -3.5890e-02, -9.6289e-02, -5.1123e-02,  1.9998e-02,
        #  6.1147e-02,  8.7683e-02,  9.6229e-02,  7.7069e-02,  2.5798e-02,
        # -3.4080e-02, -8.0655e-02, -8.3544e-02, -4.1420e-02,  1.3539e-02,
        #  5.1546e-02,  5.7607e-02,  3.6130e-02,  6.4701e-03, -2.0229e-03,
        #  1.3761e-02,  2.5576e-02,  2.0463e-02,  1.9012e-02,  3.2136e-02,
        #  5.6036e-02,  7.7448e-02,  9.7600e-02,  1.0691e-01,  7.8862e-02,
        #  4.3664e-02,  4.0778e-02,  4.4051e-02,  2.3648e-02, -5.5505e-03,
        # -2.4855e-02, -3.5660e-02, -4.1442e-02, -3.5681e-02, -2.2063e-02,
        # -7.5395e-03,  1.2595e-02,  3.2770e-02,  4.1828e-02,  4.1031e-02,
        #  3.4693e-02,  2.7855e-02,  2.3127e-02,  1.4856e-02,  5.8488e-04,
        # -1.1280e-02, -1.2282e-02, -8.6135e-03, -9.1436e-03, -8.4798e-03,
        # -1.3629e-03,  2.3824e-03, -3.3926e-03, -4.0134e-03,  3.0601e-03,
        #  3.3491e-03, -2.6719e-03, -2.9368e-03,  1.3279e-03,  5.5181e-04,
        # -4.7091e-03, -4.6957e-03, -5.7774e-04, -1.6653e-03, -4.2016e-03,
        # -6.5600e-04,  3.1431e-03, -1.0714e-03, -3.7102e-03,  1.6577e-03])
    
    def get_control(self):
        return self.pulse
    
    def activation_func(self,time):
        return self.alphas
    
    def init_activation_func(self,times):
        return self.alphas
    
    def envelope_func(self):
        return t.zeros(self.NTrot)