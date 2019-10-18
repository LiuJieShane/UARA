import os
import numpy as np
import load_trace
import fixed_env
from plot_results import plot_cellular_network

RANDOM_SEED = 77

NUM_PICO = 6 
NUM_UE = 90
SUB_BW = 180          # kHz, subchannel bandwidth
NUM_SUB = 100    # number of subchannels
P_MAX_MACRO = 46        # dBm, maximum transmit power of macrocell
P_MAX_PICO = 30       # dBm, maximum transmit power of picocell
NOISE_DENSITY = -174   #themal noise, -174dBm/Hz

TEST_TRACES = './user_traces/validation/'
TEST_LOG_FOLDER = './pf_results/results/'
LOG_FILE = './pf_results/log'

def one_hot(values = None):
    if values is None:
        values = np.random.randint(0,7,90)
    n_values = 7
    return np.eye(n_values)[values]

def picocell_first(channel_gain, K, beta):
    G = np.power(10, (channel_gain/10.0))
    M = NUM_SUB
    assert K <= M
    Pm = np.power(10, (P_MAX_MACRO/10.0))          # maximum macrocell transmit power in mW 
    Pp = np.power(10, (P_MAX_PICO/10.0))
    Pmc = (Pm-Pp)/(M-K)
    Ppc = Pp/K
    N0 = np.power(10, NOISE_DENSITY/10.0) * SUB_BW * 1000       # noise power in mW
    
    gamma_pico = np.zeros((NUM_PICO+1, NUM_UE))          # User-Picocell SINR dictionary, the firt void row for macrocell
    for num in xrange(1,7):
        for i in xrange(NUM_UE):
            interference = 0
            for j in xrange(0,7):                      # we also need to consider interference from the macro cell, thus 0~6
                if j!=num:
                    interference += Ppc*G[j][i]        
            gamma = Ppc*G[num][i]/(N0+interference)        
            gamma_pico[num][i] = gamma
     
    SINR = 10*np.log10(gamma_pico.T+1e-6)     # SINR in dB, we use SINR threshold to decide whether
                                              # a user associates to picocell or macrocell
    
    user_bs = np.argmax(SINR,axis=1)
    for i in range(NUM_UE):
        if SINR[i][user_bs[i]] <= beta:
            user_bs[i] = 0
     
    association = one_hot(user_bs).T
    
    return association
 
    
def main():
    for num_shared in range(5,100,10):
        for beta in range(-6,14,2):
            #num_shared = 55
            #beta = 2
            print "num_shared, beta: ", num_shared, beta
            os.system('rm -r ' + TEST_LOG_FOLDER)
            os.system('mkdir ' + TEST_LOG_FOLDER)

            np.random.seed(RANDOM_SEED)
            all_user_pos, all_file_names = load_trace.load_trace(TEST_TRACES)
            net_env = fixed_env.Environment(all_user_pos=all_user_pos)
            log_path = TEST_LOG_FOLDER + 'log_sim_pf_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')

            association = one_hot().T 
            trace_count = 0

            while True:  # serve video forever
                # the action is from the last decision
                # this is to make the framework similar to the real
                channel_gain, num_user_bs, rate, end_of_trace = \
                    net_env.scheduling_and_association(association, num_shared)
                
                reward = np.mean(np.log(rate)) 

                # log time_stamp, bit_rate, buffer_size, reward
                log_file.write(str(reward) + '\n')
                log_file.flush()

                association = picocell_first(channel_gain,num_shared,beta)

                if end_of_trace:
                    #plot_cellular_network(net_env.macrocell, net_env.picocells, net_env.current_user_pos, association)
                    log_file.write('\n')
                    log_file.close()
                    association = one_hot().T 

                    print "trace_count", trace_count, all_file_names[net_env.trace_idx]
                    trace_count += 1
                    if trace_count >= len(all_file_names):
                        break

                    log_path = TEST_LOG_FOLDER + 'log_sim_pf_' + all_file_names[net_env.trace_idx]
                    log_file = open(log_path, 'wb')

            # append test performance to the log
            with open(LOG_FILE + '_test', 'ab') as log_file:    
                rewards = []
                test_log_files = os.listdir(TEST_LOG_FOLDER)
                for test_log_file in test_log_files:
                    reward = []
                    with open(TEST_LOG_FOLDER + test_log_file, 'rb') as f:
                        for line in f:
                            parse = line.split()
                            try:
                                reward.append(float(parse[0]))
                            except IndexError:
                                break
                    rewards.append(np.sum(reward[1:]))

                rewards = np.array(rewards)
                rewards_min = np.min(rewards)
                rewards_5per = np.percentile(rewards, 5)
                rewards_mean = np.mean(rewards)
                rewards_median = np.percentile(rewards, 50)
                rewards_95per = np.percentile(rewards, 95)
                rewards_max = np.max(rewards)
                
                log_file.write(str(num_shared) + '\t' +
                               str(beta) + '\t' +
                               str(rewards_mean) + '\n')
                
                '''
                log_file.write(str(num_shared) + '\t' +
                               str(beta) + '\t' +
                               str(rewards_min) + '\t' +
                               str(rewards_5per) + '\t' +
                               str(rewards_mean) + '\t' +
                               str(rewards_median) + '\t' +
                               str(rewards_95per) + '\t' +
                               str(rewards_max) + '\n')
                '''
                log_file.flush()

                print 'testing results' + '\t average rewards: ' + str(rewards_mean)

if __name__ == '__main__':
    main()
