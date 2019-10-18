import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import load_trace
import fixed_env
import a3c

NUM_UE = 90
NUM_PICO = 6

S_INFO = 7
S_LEN = 91
A_DIM = 20

ACTOR_LR_RATE = 1e-4
CRITIC_LR_RATE = 1e-3
RANDOM_SEED = 77
NN_MODEL = './a3c_results/nn_model_ep_100000.ckpt'
PLOT_TRACES = './user_traces/plotting/'

def one_hot(values = None):
    if values is None:
        values = np.random.randint(0,7,90)
    n_values = 7
    return np.eye(n_values)[values]

#1 macrocell, 3 sector, each sector with 2 two picocells
def plot_cellular_network(macrocell, picocells, users, association):

    x1 = np.array(range(-1000,-750))
    y1 =  -(x1+1000)*np.sqrt(3)
    x2 = np.array(range(-1000,-750))
    y2 =  (x2+1000)*np.sqrt(3)

    x3 = np.array(range(-750,-250))
    y3 = np.tile(-250*np.sqrt(3),x3.shape)
    x4 = np.array(range(-750,-250))
    y4 = np.tile(250*np.sqrt(3),x4.shape)

    x5 = np.array(range(-250,0))
    y5 = -x5*np.sqrt(3)
    x6 = np.array(range(-250,0))
    y6 =  x6*np.sqrt(3)
    x7 = np.array(range(-250,0))
    y7= -(x7+500)*np.sqrt(3)
    x8 = np.array(range(-250,0))
    y8 = (x8+500)*np.sqrt(3)

    x9 = np.array(range(0,500))
    y9 = np.tile(0,x9.shape)
    x10 = np.array(range(0,500))
    y10 = np.tile(500*np.sqrt(3),x10.shape)
    x11 = np.array(range(0,500))
    y11 = np.tile(-500*np.sqrt(3),x11.shape)

    x12 = np.array(range(500,751))
    y12 = (x12-1000)*np.sqrt(3)
    x13 = np.array(range(500,751))
    y13 = -(x13-1000)*np.sqrt(3)
    x14 = np.array(range(500,751))
    y14 = (x14-500)*np.sqrt(3)
    x15 = np.array(range(500,751))
    y15 = -(x15-500)*np.sqrt(3)

    for i in range(1,16):
        if i == 1:
            X = eval('x'+str(i))
            Y = eval('y'+str(i))
        else:
            X = np.concatenate([X,eval('x'+str(i))])
            Y = np.concatenate([Y,eval('y'+str(i))])
    
    legends = ['macrocell','picocells','mobile users']
    label = np.argwhere(association.T == 1)[:,1]
    colors = np.array(['orange','red', 'peru', 'gold', 'olivedrab', 'aqua', 'purple'])
    plt.scatter(X,Y)
    p3 = plt.scatter(macrocell[:,0], macrocell[:,1], c = colors[0], s = 500)
    p4 = plt.scatter(picocells[:,0], picocells[:,1], c = colors[1:7], s = 100)
    p5 = plt.scatter(users[:,0], users[:,1], c = colors[label], s = 10)
    
    plt.legend([p3, p4, p5], legends, loc='lower left', scatterpoints=1)
    plt.show()


'''
def main():
    np.random.seed(RANDOM_SEED)
    all_user_pos, all_file_names = load_trace.load_trace(PLOT_TRACES)
    net_env = fixed_env.Environment(all_user_pos=all_user_pos)
    #plot_cellular_network(net_env.macrocell, net_env.picocells)

    with tf.Session() as sess:

        actor = a3c.ActorNetwork(sess,
                                 state_dim=[S_INFO, S_LEN], action_dim=A_DIM,
                                 learning_rate=ACTOR_LR_RATE)

        critic = a3c.CriticNetwork(sess,
                                   state_dim=[S_INFO, S_LEN],
                                   learning_rate=CRITIC_LR_RATE)

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver() 

        if NN_MODEL is not None:  
            saver.restore(sess, NN_MODEL)
            print("Testing model restored.")


        association = one_hot().T 
        num_shared = 50
        trace_count = 0
        scheduling_ptr = 0
        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            channel_gain, num_user_bs, rate, end_of_trace = \
                net_env.scheduling_and_association(association, num_shared)

            plot_cellular_network(net_env.macrocell, net_env.picocells, net_env.current_user_pos, association)

            reward = np.mean(np.log(rate)) 

            state_p1 = (channel_gain-np.mean(channel_gain.reshape((-1))))/(np.std(channel_gain.reshape((-1)))+1e-6)
            state_p2 = ((num_user_bs-np.mean(num_user_bs))/(np.std(num_user_bs)+1e-6)).reshape((7,1))
            state = np.concatenate([state_p1,state_p2],axis = 1)     # state shape (7, 91)


            # compute action probability vector
            action = actor.predict(np.reshape(state, (1, S_INFO, S_LEN)))    

            user_bs_prob = action[0][0:630].reshape((90,7))
            association = one_hot(np.argmax(user_bs_prob,axis=1)).T
            num_shared = 10*action[0][630:640].argmax() + 5

            scheduling_ptr += 1
            if scheduling_ptr >= 3:
                break

            if end_of_trace:
                association = one_hot().T 
                num_shared = 50

                trace_count += 1
                if trace_count >= len(all_file_names):
                    break

if __name__ == '__main__':
    main()

'''

