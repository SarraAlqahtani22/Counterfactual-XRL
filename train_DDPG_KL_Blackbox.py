"""
    Blackbox use prediction model for transition model to predict agent's a' and s' and then compute KL divergence


        TO DO: lstm for last 3 timesteps
"""

import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import copy


import wandb
wandb.init(name='Counterfactual - KL_Blackbox - new', project="counterfactual_based_attack")


import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Model

from DDPG import DDPG

from scipy.special import softmax
from scipy.special import kl_div

# global variables for DDPG

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.90     # reward discount   originally 0.9
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
# ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################



def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple_adversary", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=40000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./weights_save/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=500, help="save model once every time this many episodes are completed") #originally 1000
    parser.add_argument("--load-dir", type=str, default="./weights_final/", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=True)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def get_logits(model, obses, in_length, actions=None, model_type="other_policy"):
    if model_type == "other_policy":
        obs = np.reshape(obses, [-1, 3, in_length])
        action = np.reshape(actions, [-1, 3, 1])
        model_input = np.concatenate([obs, action], axis=2)
        logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
        return logits
    elif model_type == "transition":
        obs = np.reshape(obses, [-1, 3, in_length])   ## why this is 3 * 28
        action = np.reshape(actions, [-1, 3, 3])      ## why this is 3 * 3
        model_input = np.concatenate([obs, action], axis=2)
        logits = np.asarray(model.predict_on_batch(model_input.astype('float64')))
        return logits
    elif model_type == "policy":
        obs = np.reshape(obses, [-1, 3, in_length])
        logits = np.asarray(model.predict_on_batch(obs.astype('float64')))
        return logits
    else:
        raise NotImplementedError

def build_model(scope, in_length, fname=None, model_type="other_policy"):
    if model_type == "other_policy":
        with tf.variable_scope(scope):
            # build functional model
            visible = Input(shape=(3, in_length))
            hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
            hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)
            # left branch decides second agent action
            hiddenLeft = LSTM(10, name='leftBranch')(hidden2)
            agent2 = Dense(5, activation='softmax', name='agent2classifier')(hiddenLeft)
            # right branch decides third agent action
            hiddenRight = LSTM(10, name='rightBranch')(hidden2)
            agent3 = Dense(5, activation='softmax', name='agent3classifier')(hiddenRight)

            model = Model(inputs=visible, outputs=[agent2, agent3])

            model.compile(optimizer='adam',
                          loss={'agent2classifier': 'categorical_crossentropy',
                                'agent3classifier': 'categorical_crossentropy'},
                          metrics={'agent2classifier': ['acc'],
                                   'agent3classifier': ['acc']})

            model.summary()

            U.initialize()

            if fname is not None:
                model.load_weights(fname)

        return model

    elif model_type == "transition":
        visible = Input(shape=(3, in_length))
        hidden1 = LSTM(100, return_sequences=True)(visible)
        hidden2 = LSTM(64, return_sequences=True)(hidden1)
        hiddenObservation = LSTM(64, name='observationBranch')(hidden2)
        observation = Dense(in_length-3, name='observationScalar')(hiddenObservation)

        # model = Model(inputs=visible,outputs=[agent1,agent2,agent3,observation])
        model = Model(inputs=visible, outputs=observation)
        model.compile(optimizer='adam',
                      loss={'observationScalar': 'mse'},
                      metrics={'observationScalar': ['mae']})

        model.summary()

        U.initialize()

        if fname is not None:
            model.load_weights(fname)

        return model

    elif model_type == "policy":

        with tf.variable_scope(scope):
            # build functional model
            visible = Input(shape=(3, in_length))
            hidden1 = LSTM(32, return_sequences=True, name='firstLSTMLayer')(visible)
            hidden2 = LSTM(16, name='secondLSTMLayer', return_sequences=True)(hidden1)

            hidden_final = LSTM(10, name='leftBranch')(hidden2)
            agent0 = Dense(5, activation='softmax', name='agent0classifier')(hidden_final)

            model = Model(inputs=visible, outputs=agent0)

            model.compile(optimizer='adam',
                          loss={'agent0classifier': 'categorical_crossentropy'},
                          metrics={'agent0classifier': ['acc']})

            model.summary()

            U.initialize()

            if fname is not None:
                model.load_weights(fname)

        return model

    else:
        raise NotImplementedError




def train(arglist):
    # with U.single_threaded_session():

    tf.reset_default_graph()
        # Create environment
    env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # simulator_env = make_env(arglist.scenario, arglist, arglist.benchmark)
    marl_sess = U.single_threaded_session()

    if arglist.scenario == 'simple_spread':
        in_length = 54
        cor_agent_in_length = 18
    else:
        in_length = 28
        cor_agent_in_length = 10

    if arglist.scenario == "simple_adversary":
        other_act_model_fname = "Prediction_weights/actionMultiClassNetwork_simpadv"
        transition_model_fname = "Prediction_weights/StateTransitionNetwork_adv"
        corrupted_act_model_fname = "Prediction_weights/adv_agent1_policy_predictor"
    else:
        other_act_model_fname = "Prediction_weights/actionMultiClassNetwork"
        transition_model_fname = "Prediction_weights/StateTransitionNetwork"
        corrupted_act_model_fname = "Prediction_weights/agent0_policy_predictor"
        

    with marl_sess as sess:

        

        # U.initialize()



        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))
        
        ##################### DDPG #######################
        
        #env = gym.make(ENV_NAME)
        #env = env.unwrapped
        #env.seed(1)

        # s_dim = 12 #shape of the obs will be [[shape of agent 3], [action_1, [action_2]]]
        s_dim = 15
        a_dim = 5
        a_bound = 1    #upper limit of the action

        #ddpg = DDPG(a_dim, s_dim, a_bound)

        var = 3  # control exploration
        t1 = time.time()

        ##################### DDPG #######################


        
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        
        ddpg = DDPG(a_dim, s_dim, a_bound)   # define DDPG policy trainer here
        transition_model = build_model("Transition", in_length + 3, fname=transition_model_fname, model_type="transition")
        # U.initialize()
        """
        obs_n[0] has dimension of 8
        obs_n[1] has dimension of 10
        obs_n[2] has dimension of 10, which extra coordinate of target landmark
        """
        episode_step = 0
        train_step = 0
        t_start = time.time()

        
        #### DDPG parameters initialization ####

        DDPG_obs = obs_n[2]  # (10,)
        #DDPG_obs = np.append(DDPG_obs, np.array([0,0]))   #(12,)
        DDPG_obs = np.append(DDPG_obs, np.zeros((5,)))
        DDPG_rew = 0
        DDPG_ep_rw = 0
        ep_obs = []
        ep_act = []



        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]   # (3,5), three agent, each with a probability distribution of action #NONE, UP, DOWN, LEFT, RIGHT
            # environment step
            DDPG_act = ddpg.choose_action(DDPG_obs)

            ###### modified MADDPG env with output of DDPG network #####
            
            DDPG_act = softmax(DDPG_act)
            attack_action_n = copy.deepcopy(action_n)
            attack_action_n[2] = DDPG_act  # compromised agent attack

            # FIXME HERE *****************************************************
            
            new_obs_n, rew_n, done_n, info_n = env.step(attack_action_n)

            # Question 1: how to copy the simuator env
            # MADDPG_new_obs_n, _, _, _ =simulator_env.step(action_n) 

            # FIXME direct argmax?
            action_taken_0 = np.random.choice(5, p=action_n[0])
            action_taken_1 = np.random.choice(5, p=action_n[1])
            action_taken_2 = np.random.choice(5, p=action_n[2])
            action_to_simulate = [action_taken_0, action_taken_1, action_taken_2]

            #######
            obs_n = np.array(obs_n)
            obs_n_all = np.concatenate((obs_n[0],obs_n[1],obs_n[2])).reshape(1,-1)
            ep_obs.append(obs_n_all)
            ep_act.append(action_to_simulate)
            #######
            if episode_step > 2:
                MADDPG_new_obs_n = get_logits(transition_model, ep_obs[-3:], in_length, ep_act[-3:], model_type="transition")
                # a_t a'_t 
                # s_t+1, s'_t+1    use model 
                # a_t+1, a'_t+1

                # shape from transition model is (8, 10, 10)
                MADDPG_agent0_obs = MADDPG_new_obs_n[-1,0:8]
                MADDPG_agent1_obs = MADDPG_new_obs_n[-1,8:18]
                MADDPG_agent2_obs = MADDPG_new_obs_n[-1,18:28]
                MADDPG_new_obs_n = [MADDPG_agent0_obs, MADDPG_agent1_obs, MADDPG_agent2_obs]

                MADDPG_next_action_n = [agent.action(obs) for agent, obs in zip(trainers, MADDPG_new_obs_n)]
                MADDPG_next_action_n_atk = [agent.action(obs) for agent, obs in zip(trainers, new_obs_n)]

                # KL_0 = sum(kl_div(MADDPG_next_action_n[0], MADDPG_next_action_n_atk[0]))
                KL_1 = sum(kl_div(MADDPG_next_action_n[1], MADDPG_next_action_n_atk[1]))


                #***************************************************************

                DDPG_rew = KL_1 
                if train_step % (arglist.max_episode_len - 1) == 0:
                    DDPG_rew += 10 * rew_n[0]
            else:
                DDPG_rew = 0.001

   



            # DDPG_new_obs = np.append(new_obs_n[2], [action_taken_0, action_taken_1])
            DDPG_new_obs = np.append(new_obs_n[2], attack_action_n[1])
            # print("shape is " + str(DDPG_new_obs.shape))
            ddpg.store_transition(DDPG_obs, DDPG_act, DDPG_rew, DDPG_new_obs)


            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n


            if ddpg.pointer > MEMORY_CAPACITY:
                #var *= .9995    # decay the action randomness
                ddpg.learn()
            
            DDPG_obs = DDPG_new_obs
            DDPG_ep_rw += DDPG_rew 
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])

            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                print("episode reward for DDPG agent: {}".format(DDPG_ep_rw/arglist.save_rate))
                wandb.log({'Reward for DDPG': DDPG_ep_rw/arglist.save_rate})
                DDPG_ep_rw = 0
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                    wandb.log({'Reward for MADDPG': np.mean(episode_rewards[-arglist.save_rate:])})
                    # print("episode reward for DDPG agent: {}".format(DDPG_ep_rw/len(episode_rewards)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break

if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
