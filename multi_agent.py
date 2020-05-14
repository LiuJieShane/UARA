import os
import logging
import multiprocessing as mp
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf
from sklearn.model_selection import GridSearchCV

import numpy as np
import a3c
import load_trace
import env
import fixed_env
from picocell_first import picocell_first

S_INFO = 7
S_LEN = 90

ACTOR_LR_RATE = 1e-5
CRITIC_LR_RATE = 1e-4
NUM_AGENTS = 1
EPSILON_BEGIN = 1
EPSILON_STEP = 1e-4
EPOCH_START = 0
EPOCH_END = 10000
MODEL_SAVE_INTERVAL = 1000
TESTING_INTERVAL = 10
RANDOM_SEED = 77

#NN_MODEL = './a3c_results/nn_model_ep_2000.ckpt'
NN_MODEL = None
TRAIN_TRACES = './user_traces/training/'
TEST_TRACES = './user_traces/validation/'
SUMMARY_DIR = './a3c_results'
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


def testing(epoch, actor):
    # clean up the test results folder
    os.system('rm -r ' + TEST_LOG_FOLDER)
    os.system('mkdir ' + TEST_LOG_FOLDER)
    
    # run test script
    np.random.seed(RANDOM_SEED)

    all_user_pos, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = fixed_env.Environment(all_user_pos=all_user_pos)

    log_path = TEST_LOG_FOLDER + 'log_sim_rl_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')


    # initializing
    association = one_hot().T 
    num_shared = 50
    trace_count = 0
    # time_stamp = 0
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
            log_file.write('\n')
            log_file.close()
            association = one_hot().T 
            num_shared = 50
       
            trace_count += 1
            if trace_count >= len(all_file_names):
                break

            log_path = TEST_LOG_FOLDER + 'log_sim_rl_' + all_file_names[net_env.trace_idx]
            log_file = open(log_path, 'wb')
    
    with open(LOG_FILE + '_test', 'ab') as log_file:
        # append test performance to the log
        
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

        log_file.write(str(epoch) + '\t' +
                       str(rewards_min) + '\t' +
                       str(rewards_5per) + '\t' +
                       str(rewards_mean) + '\t' +
                       str(rewards_median) + '\t' +
                       str(rewards_95per) + '\t' +
                       str(rewards_max) + '\n')
        log_file.flush()
        print 'epoch:' + str(epoch) + '\t average rewards: ' + str(rewards_mean)


def central_agent(net_params_queues, exp_queues):

    assert len(net_params_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    #logging.basicConfig(filename=LOG_FILE + '_central', filemode='a', level=logging.INFO)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim= [S_INFO,S_LEN], action_dim = A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO,S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()  # save neural net parameters
        # restore neural net parameters
        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            saver.restore(sess, nn_model)
            print("Model restored.")

        epoch = EPOCH_START
        #testing(epoch, actor)

        # assemble experiences from agents, compute the gradients
        while True:
            # synchronize the network parameters of work agent
            actor_net_params = actor.get_network_params()
            critic_net_params = critic.get_network_params()
            for i in xrange(NUM_AGENTS):
                net_params_queues[i].put([actor_net_params, critic_net_params])
                # Note: this is synchronous version of the parallel training,
                # which is easier to understand and probe. The framework can be
                # fairly easily modified to support asynchronous training.
                # Some practices of asynchronous training (lock-free SGD at
                # its core) are nicely explained in the following two papers:
                # https://arxiv.org/abs/1602.01783
                # https://arxiv.org/abs/1106.5730

            # record average reward and td loss change
            # in the experiences from the agents
            total_batch_len = 0.0
            total_reward = 0.0
            total_td_loss = 0.0
            total_agents = 0.0 

            # assemble experiences from the agents
            actor_gradient_batch = []
            critic_gradient_batch = []

            for i in xrange(NUM_AGENTS):
                s_batch, a_batch, r_batch, terminal  = exp_queues[i].get()

                actor_gradient, critic_gradient, td_batch = \
                    a3c.compute_gradients(
                        s_batch=np.stack(s_batch, axis=0),
                        a_batch=np.vstack(a_batch),
                        r_batch=np.vstack(r_batch),
                        terminal=terminal, actor=actor, critic=critic)

                actor_gradient_batch.append(actor_gradient)
                critic_gradient_batch.append(critic_gradient)

                total_reward += np.sum(r_batch)
                total_td_loss += np.sum(td_batch)
                total_batch_len += len(r_batch)
                total_agents += 1.0

            # compute aggregated gradient
            assert NUM_AGENTS == len(actor_gradient_batch)
            assert len(actor_gradient_batch) == len(critic_gradient_batch)
            # assembled_actor_gradient = actor_gradient_batch[0]
            # assembled_critic_gradient = critic_gradient_batch[0]
            # for i in xrange(len(actor_gradient_batch) - 1):
            #     for j in xrange(len(assembled_actor_gradient)):
            #             assembled_actor_gradient[j] += actor_gradient_batch[i][j]
            #             assembled_critic_gradient[j] += critic_gradient_batch[i][j]
            # actor.apply_gradients(assembled_actor_gradient)
            # critic.apply_gradients(assembled_critic_gradient)
            for i in xrange(len(actor_gradient_batch)):
                actor.apply_gradients(actor_gradient_batch[i])
                critic.apply_gradients(critic_gradient_batch[i])

            # log training information
            epoch += 1
            avg_reward = total_reward  / total_agents
            avg_td_loss = total_td_loss / total_batch_len

            print 'epoch:' + str(epoch) + ' average_reward:' + str(avg_reward)
            if epoch % TESTING_INTERVAL == 0:
                testing(epoch, actor)
                
            if epoch % MODEL_SAVE_INTERVAL == 0:
                save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                
            if epoch == EPOCH_END:
                #save_path = saver.save(sess, SUMMARY_DIR + "/nn_model_ep_" + str(epoch) + ".ckpt")
                #logging.info("Model saved in file: " + save_path)
                break



def agent(agent_id, all_user_pos, net_params_queue, exp_queue):
	
    net_env = env.Environment(all_user_pos = all_user_pos,
                              random_seed=agent_id)

    with tf.Session() as sess, open(LOG_FILE + '_agent_' + str(agent_id), 'wb') as log_file:
        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim = A_DIM,
                                 learning_rate=ACTOR_LR_RATE)
        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        # initial synchronization of the network parameters from the coordinator
        actor_net_params, critic_net_params = net_params_queue.get()
        actor.set_network_params(actor_net_params)
        critic.set_network_params(critic_net_params)     
        
        # initializing
        association = one_hot().T 
        num_shared = 50
        
        s_batch = [np.zeros((S_INFO,S_LEN))]
        a_batch = [np.zeros(A_DIM,)]
        r_batch = []
        
        epsilon = EPSILON_BEGIN

        while True:  # experience scheduling and allocation forever

            # the action is from the last decision
            # this is to make the framework similar to the real
            channel_gain, num_user_bs, rate, end_of_trace =  net_env.scheduling_and_association(association, num_shared)

            reward = np.mean(np.log(rate)) 
            
            '''
            if reward < 0.3:
                reward = -1
            else:
                reward = 1
            '''
            r_batch.append(reward)

            #last_bit_rate = bit_rate
            
            '''
            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write(str(time_stamp) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n')
            log_file.flush()
            '''

            state_p1 = (channel_gain-np.mean(channel_gain.reshape((-1))))/(np.std(channel_gain.reshape((-1)))+1e-6)
            state_p2 = ((num_user_bs-np.mean(num_user_bs))/(np.std(num_user_bs)+1e-6)).reshape((7,1))
            #state = np.concatenate([state_p1,state_p2],axis = 1)     # state shape (7, 91)
            state = state_p1

            # compute action probability vector   action_prob of size (1,A_DIM)
            action_prob = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))  
            action = epsilon_greedy(action_prob, epsilon)    
            
            association, num_shared = rl_scheduling(channel_gain, action)

            # report experience to the coordinator
            if end_of_trace:
                exp_queue.put([s_batch[1:],  # ignore the first chuck
                               a_batch[1:],  # since we don't have the
                               r_batch[1:],  # control over it
                               end_of_trace])

                # synchronize the network parameters from the coordinator
                actor_net_params, critic_net_params = net_params_queue.get()
                actor.set_network_params(actor_net_params)
                critic.set_network_params(critic_net_params)

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                #log_file.write('\n')  # so that in the log we know where video ends

            # store the state and action into batches
            if end_of_trace:
                association = one_hot().T 
                num_shared = 50
                s_batch = [np.zeros((S_INFO,S_LEN))]
                a_batch = [np.zeros(A_DIM,)]
                
                if epsilon > 0:
                    epsilon -= EPSILON_STEP

            else:
                s_batch.append(state)
                a_batch.append(action)


def main():

    np.random.seed(RANDOM_SEED)

    # create result directory
    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    # inter-process communication queues
    net_params_queues = []
    exp_queues = []
    for i in xrange(NUM_AGENTS):
        net_params_queues.append(mp.Queue(1))
        exp_queues.append(mp.Queue(1))

    # create a coordinator and multiple agent processes
    # (note: threading is not desirable due to python GIL)
    coordinator = mp.Process(target=central_agent,
                             args=(net_params_queues, exp_queues))
    coordinator.start()

    all_user_pos, _ = load_trace.load_trace(TRAIN_TRACES)
    agents = []
    for i in xrange(NUM_AGENTS):
        agents.append(mp.Process(target=agent,
                                 args=(i, all_user_pos,
                                       net_params_queues[i],
                                       exp_queues[i])))
    for i in xrange(NUM_AGENTS):
        agents[i].start()

    # wait unit training is done
    coordinator.join()

if __name__ == '__main__':
    main()
