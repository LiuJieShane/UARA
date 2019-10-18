import numpy as np
import matplotlib.pyplot as plt

#NUM_MACRO = 1
#SECTOR_PER_MACRO = 3
#NUM_PICO_PER_SECTOR = 2
#NUM_UE_PER_SECTOR = 30
NUM_PICO = 6
NUM_UE = 90
INTERSITE_DIS = 500   # meters, intersite distance
MIN_DIS_MP = 75       # meters, minimum distance between macrocell and picocell
MIN_DIS_PP = 40       # meters, minimum distance between picocell and picocell
MIN_DIS_MU = 35
MIN_DIS_PU = 10

SUB_BW = 180          # kHz, subchannel bandwidth
NUM_SUB = 100         # number of subchannels
REUSE = 1             # frequency reuse for picocells
P_MAX_MACRO = 46        # dBm, maximum transmit power of macrocell
P_MAX_PICO = 30       # dBm, maximum transmit power of picocell
ANT_GAIN_MACRO = 15   # dBi, macrocell antenna gain
ANT_GAIN_PICO = 5     # dBi, picocell antenna gain
SHADOW_FADING = 8    # dB,  shadow fading
# path_loss_macro = 128.1 + 37.6 lg(R) dB
# path_loss_pico = 140.7 + 36.7 lg(R) dB
# feasible pico_cell_position 
RANDOM_SEED = 77
TOTAL_SCHEDULING_INT = 60
NOISE_DENSITY = -174   #themal noise, -174dBm/Hz
MEGA = 1000000.0

PICO_DEPLOY = './pico_deploy/pico_pos.txt'
UE_TRACES = './user_traces/'

all_pico_deploy = np.loadtxt(PICO_DEPLOY)
#pico_deploy_idx = np.random.randint(len(all_pico_deploy))
pico_deploy_idx = 0


class Environment:
    def __init__(self, all_user_pos, random_seed = RANDOM_SEED ):
        np.random.seed(random_seed)
        self.macrocell = np.zeros((1,2))    # macrocell position
        self.picocells = all_pico_deploy.reshape((6,2))
        self.all_user_pos = all_user_pos
        
        self.trace_idx =  0
        self.user_pos = self.all_user_pos[self.trace_idx]  
        self.scheduling_ptr = 0
        
        self.current_user_pos = np.zeros((90,2))
        self.K = 10     # subchannel number of frequency reuse-1 for picocells
        self.association = np.zeros((90,7))  # UE-BS association array 0 for macrocell, 1-6 for picocell
        
    def scheduling_and_association(self, association, num_shared):
        
        current_user_pos = self.user_pos[:, self.scheduling_ptr*2:(self.scheduling_ptr+1)*2]
        self.current_user_pos = current_user_pos
        self.association = association
        self.K = num_shared 
        distance = []
        mu_relative = current_user_pos - np.tile(self.macrocell,(90,1))  #macrocell user relative position
        mu_dis = np.sqrt(np.sum(np.square(mu_relative),axis=1))          #macrocell user distance
        distance.append(mu_dis)
        for i in xrange(NUM_PICO):                 
            pu_relative = current_user_pos - np.tile(self.picocells[i],(90,1)) #picocell user relative position
            pu_dis = np.sqrt(np.sum(np.square(pu_relative),axis=1))            #picocell user distance
            distance.append(pu_dis)
        distance = np.array(distance)
        
        path_loss_macro = (128.1+37.6*np.log10((distance[0]+35)/1000)).reshape((1,90))
        path_loss_pico = 140.7+36.7*np.log10((distance[1:7]+10)/1000)
        path_loss = np.concatenate([path_loss_macro,path_loss_pico])
        channel_gain_macro =  - path_loss_macro - SHADOW_FADING +  ANT_GAIN_MACRO 
        channel_gain_pico  = -path_loss_pico -SHADOW_FADING + ANT_GAIN_PICO
        channel_gain = np.concatenate([channel_gain_macro, channel_gain_pico]) 
        G = np.power(10, (channel_gain/10.0))
        
        M = NUM_SUB
        K = num_shared
        assert K <= M
        # using Partially Shared Deployment 
        Pm = np.power(10, (P_MAX_MACRO/10.0))          # maximum macrocell transmit power in mW 
        Pp = np.power(10, (P_MAX_PICO/10.0))
        Pmc = (Pm-Pp)/(M-K)
        Ppc = Pp/K
        N0 = np.power(10, NOISE_DENSITY/10.0) * SUB_BW * 1000       # noise power in mW 
        
        gamma_macro = {}         # dictionary
        gamma_pico = {}          # dictionary
        # compute SINR for user associated to macrocell
        gamma_macro['exclusive'] = []
        gamma_macro['shared'] = []
        for i in xrange(NUM_UE):
            gamma1 = Pmc*G[0][i]/N0
            gamma_macro['exclusive'].append(gamma1)
            interference = 0
            for j in xrange(1,7):
                interference += Ppc*G[j][i] 
            gamma2 = Ppc*G[0][i]/(N0 + interference)
            gamma_macro['shared'].append(gamma2)
        
        # compute SINR for user associated to picocell
        for num in xrange(1,7):
            gamma_pico[num] = []
            for i in xrange(NUM_UE):
                interference = 0
                for j in xrange(0,7):
                    if j!=num:
                        interference += Ppc*G[j][i]
                gamma = Ppc*G[num][i]/(N0+interference)
                gamma_pico[num].append(gamma)
        
        R = np.zeros((7,90))
        R[0] = (M-K)*SUB_BW*1000*np.log(1+np.array(gamma_macro['exclusive'])) + K*SUB_BW*1000*np.log(1+np.array(gamma_macro['shared'])) 
        for num in xrange(1,7):
            R[num] = K*SUB_BW*1000*np.log(1+np.array(gamma_pico[num]))
            
        num_user_bs = np.zeros((7,))
        for j in xrange(0,7):
            num_user_bs[j] = np.sum(association[j]) 
        N = num_user_bs
        rate = np.zeros((NUM_UE,))
        for i in xrange(NUM_UE):
            a_index = association[:,i].tolist().index(1)
            rate[i] = R[a_index][i]/N[a_index]
         
        self.scheduling_ptr += 1
        
        end_of_trace = False
        if self.scheduling_ptr >= TOTAL_SCHEDULING_INT:
            end_of_trace = True
            self.scheduling_ptr = 0
            self.trace_idx +=1
            if self.trace_idx >= len(self.all_user_pos):
                self.trace_idx = 0    
            self.user_pos = self.all_user_pos[self.trace_idx]
        #reward = np.sum(np.log(rate))             # using proportional fairness objective as reward function 
        
        return channel_gain, num_user_bs, rate/MEGA, end_of_trace  #rate in Mbps