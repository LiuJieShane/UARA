import os
import logging
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

import numpy as np
import a3c
import load_trace
import fixed_env
from picocell_first import picocell_first
from plot_results import plot_cellular_network

S_INFO = 7
S_LEN = 90

ACTOR_LR_RATE = 1e-5
CRITIC_LR_RATE = 1e-4
RANDOM_SEED = 77

NN_MODEL = './a3c_results/nn_model_ep_10000.ckpt'
TEST_TRACES = './user_traces/validation/'
TEST_LOG_FOLDER = './a3c_results/results/'
LOG_FILE = './a3c_results/log'

K_SET = range(5, 100, 10)
BETA_SET = range(-6, 14, 2)
K_DIM = len(K_SET)
BETA_DIM = len(BETA_SET)
A_DIM = K_DIM + BETA_DIM


def epsilon_greedy(action_prob, epsilon):
    hint = np.random.rand()
    shared_prob = action_prob[0][0:K_DIM]
    channel_prob = action_prob[0][K_DIM:A_DIM]
    shared = np.zeros(K_DIM)
    channel = np.zeros(BETA_DIM)
    if hint < epsilon:
        index1 = np.random.randint(0,K_DIM)
        index2 = np.random.randint(0,BETA_DIM)
        shared[index1] = 1
        channel[index2] = 1
    else:
        index1 = np.argmax(shared_prob)
        index2 = np.argmax(channel_prob)
        shared[index1] = 1
        channel[index2] = 1
    action = np.concatenate((shared,channel))
    return action

def rl_scheduling(channel_gain, action):
    K = K_SET[np.argmax(action[0:K_DIM])]
    beta = BETA_SET[np.argmax(action[K_DIM:A_DIM])]
    association = picocell_first(channel_gain, K, beta)
    return association, K

def one_hot(values = None):
    if values is None:
        values = np.random.randint(0,7,90)
    n_values = 7
    return np.eye(n_values)[values]

def main():
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    np.random.seed(RANDOM_SEED)
    all_user_pos, all_file_names = load_trace.load_trace(TEST_TRACES)
    net_env = fixed_env.Environment(all_user_pos=all_user_pos)
    log_path = TEST_LOG_FOLDER + 'log_sim_rl_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')
    
    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()  # save neural net parameters

        # restore neural net parameters
        if NN_MODEL is not None:  # NN_MODEL is the path to file
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")
        
        # initializing
        association = one_hot().T 
        num_shared = 50
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

            state_p1 = (channel_gain-np.mean(channel_gain.reshape((-1))))/(np.std(channel_gain.reshape((-1)))+1e-6)
            state_p2 = ((num_user_bs-np.mean(num_user_bs))/(np.std(num_user_bs)+1e-6)).reshape((7,1))
            #state = np.concatenate([state_p1,state_p2],axis = 1)     # state shape (7, 91)
            state = state_p1


            # compute action probability vector
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))  
            action = epsilon_greedy(action_prob, 0)          # set epsilon to zero when testing

            association, num_shared = rl_scheduling(channel_gain, action)

            if end_of_trace:
                print all_file_names[net_env.trace_idx-1],net_env.scheduling_ptr,'number of shared subchannels:', num_shared, 'SINR threshold:', BETA_SET[np.argmax(action[K_DIM:A_DIM])]
                #plot_cellular_network(net_env.macrocell, net_env.picocells, net_env.current_user_pos, association)
                log_file.write('\n')
                log_file.close()
                association = one_hot().T 
                num_shared = 50
                
                trace_count += 1
                if trace_count >= len(all_file_names):
                    break

                log_path = TEST_LOG_FOLDER + 'log_sim_rl_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'wb')

        # append test performance to the log
    with open(LOG_FILE + '_rl_test', 'ab') as log_file:        
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

        log_file.write(str(rewards_min) + '\t' +
                       str(rewards_5per) + '\t' +
                       str(rewards_mean) + '\t' +
                       str(rewards_median) + '\t' +
                       str(rewards_95per) + '\t' +
                       str(rewards_max) + '\n')
        log_file.flush()
        
        print 'testing results' + '\t average rewards: ' + str(rewards_mean)

if __name__ == '__main__':
    main()