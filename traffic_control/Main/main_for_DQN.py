import os
import sys
import argparse
import time
from logger import logger
from traffic_env import traffic_env
import shutil
import numpy as np
import traci
import pickle
import utils
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='traffic_dqn')
parser.add_argument('--model_name', type=str, default='model')
parser.add_argument('--render', action='store_true')
parser.add_argument('--mpc_horizon', type=int, default=15)
parser.add_argument('--num_random_action_selection', type=int, default=4096)
parser.add_argument('--nn_layers', type=int, default=1)
args = parser.parse_args()
data_dir = '../DATA'
exp_dir = os.path.join(data_dir, args.exp_name)
# mod_dir = os.path.join(data_dir, args.model_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
else:
    os.makedirs(exp_dir, exist_ok=True)

import tensorflow as tf
import matplotlib.pyplot as plt
import gym, threading, queue
from myqueue import replay_buffer
from itertools import chain
import os
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Convolution2D, LSTM, TimeDistributed, Input, concatenate, Lambda
from tensorflow.contrib.keras.api.keras import backend as K

## PPO Parameter
EPSILON = 0.2

class DeepQNetWork(object):
    def __init__(self, n_actions = 2,
                 n_features = 0,
                 learning_rate=0.0002,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 tua = 0.001,
                 memory_size = 10000,
                 batch_size = 32,
                 e_greedy_increment=0.0002,
                 DDQN = False,
                 saving_loading = False
                 ):
        ## 超参数设置
        self.n_act = n_actions
        self.n_fea = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.decay_update = tua
        self.memory_size = memory_size
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.batch_size = batch_size
        self.road_size = 48
        self.state_size = (16, 20, 1)
        self.DDQN = DDQN
        self.cost_hist = []
        ## 网络搭建
        self.sess = tf.Session()
        keras.backend.set_session(self.sess)

        self.Buffer = replay_buffer(self.memory_size)
        self.model = self._build_net()
        self.target_model = self._build_net()

        ### 采用此种更新方式更容易收敛
        if saving_loading == True:
            self.load_model()
        self.sess.run(tf.global_variables_initializer())
        self.target_model.set_weights(self.model.get_weights())
        self.saving_or_loading = saving_loading
        self.optimizer = self.optimizer()
    def load_model(self):
        self.save_dir = '../checkpoints/traffic_dqn'
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
        self.save_path = os.path.join(self.save_dir, 'weights.h5')
        try:
            self.model.load_weights(self.save_path)
            print("Successfully loaded!!!")
        except:
            print("Could not find old weights!")
    def _build_net(self):
        pinput = Input(shape=self.state_size)
        sp = Convolution2D(16, (4, 4), strides=(2, 2), activation='relu')(pinput)
        sp = Convolution2D(32, (2, 2), strides=(1, 1), activation='relu')(sp)
        vinput = Input(shape=self.state_size)
        sv = Convolution2D(16, (4, 4), strides=(2, 2), activation='relu')(vinput)
        sv = Convolution2D(32, (2, 2), strides=(1, 1), activation='relu')(sv)
        pflatten = Flatten()(sp)
        vflatten = Flatten()(sv)
        ainput = Input(shape=(1,), dtype='int32')
        aonehot = Lambda(lambda i: K.one_hot(i,4))(ainput)
        aonehot = Flatten()(aonehot)
        sinput = concatenate([pflatten, vflatten, aonehot])
        sinput = Dense(128, activation='relu')(sinput)
        sinput = Dense(64, activation='relu')(sinput)
        out = Dense(self.n_act, activation='linear')(sinput)
        model = keras.models.Model(inputs=[pinput, vinput, ainput], outputs=out)
        return model
    def update_target_model(self):
        weights  = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.decay_update + target_weights[i] * (1 - self.decay_update)
        self.target_model.set_weights(target_weights)
    def optimizer(self):
        a = K.placeholder(shape=[None, ], dtype='int32')
        y = K.placeholder(shape=[None, ], dtype='float32')
        prediction = self.model.output
        a_one_hot = K.one_hot(a, self.n_act)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        optimizer = keras.optimizers.Adam(lr=self.lr)
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        train = K.function([self.model.input[0], self.model.input[1], self.model.input[2], a, y], [loss], updates=updates)
        return train
    def store_transition(self, state, action, reward, state_):
        self.Buffer.add(state, action, reward, state_)

    def choose_action(self, s, training=True):
        if training:
            if np.random.uniform() < self.epsilon:
                q_value = self.forward_e(s)[0]
                action = np.argmax(q_value)
            else:
                action = np.random.randint(0, self.n_act)
        else:
            q_value = self.forward_e(s)[0]
            action = np.argmax(q_value)
        return action
    def forward_e(self, s):
        SPV = np.reshape(s[0], newshape=[-1, 16, 20, 2])
        SP = np.reshape(SPV[:, :, :, 0], newshape=[-1, 16, 20, 1])
        SV = np.reshape(SPV[:, :, :, 1], newshape=[-1, 16, 20, 1])
        SA = np.reshape(s[1], newshape=[-1,1])
        q_values = self.model.predict([SP, SV, SA])
        return q_values

    def learn_forward_t(self, s_, phase):
        SPV = np.reshape(s_, newshape=[-1, 16, 20, 2])
        SP = np.reshape(SPV[:, :, :, 0], newshape=[-1, 16, 20, 1])
        SV = np.reshape(SPV[:, :, :, 1], newshape=[-1, 16, 20, 1])
        SA = np.reshape(phase, newshape=[-1,1])
        # s = np.reshape(s_, newshape=[self.batch_size, self.road_size, self.road_size, 2])
        # S1 = np.reshape(s[:, :, :, 0], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # S2 = np.reshape(s[:, :, :, 1], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        q_value = self.target_model.predict([SP, SV, SA])
        return q_value
    def learn_forward_e(self, s, phase):
        SPV = np.reshape(s[0], newshape=[-1, 16, 20, 2])
        SP = np.reshape(SPV[:, :, :, 0], newshape=[-1, 16, 20, 1])
        SV = np.reshape(SPV[:, :, :, 1], newshape=[-1, 16, 20, 1])
        SA = np.reshape(phase, newshape=[-1,1])
        q_value = self.model.predict([SP, SV, SA])
        return q_value

    def learn(self):
        bran_batch = self.Buffer.get_Batch(self.batch_size)
        state = [batch[0][0] for batch in bran_batch]
        state = np.reshape(state, newshape=[-1, 16, 20, 2])
        phase = [batch[0][1] for batch in bran_batch]
        phase = np.array(phase)
        action = np.array([batch[1] for batch in bran_batch])
        reward = [batch[2] for batch in bran_batch]
        state_ = [batch[3][0] for batch in bran_batch]
        state_ = np.reshape(state_, newshape=[-1, 16, 20, 2])
        phase_ = [batch[3][1] for batch in bran_batch]
        phase_ = np.array(phase_)
        q_next_t = self.learn_forward_t(state_, phase_)
        if self.DDQN:
            q_next_e = self.learn_forward_e(state_, phase_)
            max_act_next = np.argmax(q_next_e, axis=1)
            selected_q_next = q_next_t[range(self.batch_size), max_act_next]
        else:
            selected_q_next = np.max(q_next_t, axis=1)
        q_target = reward + self.gamma*selected_q_next
        # action = np.reshape(action, newshape=[-1, 1])
        phase = np.reshape(phase, newshape=[-1,1])
        SP = np.reshape(state[:, :, :, 0], newshape=[-1, 16, 20 ,1])
        SV = np.reshape(state[:, :, :, 1], newshape=[-1, 16, 20, 1])
        self.cost  = self.optimizer([SP, SV, phase, action, q_target])
        self.update_target_model()
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        return self.cost

def traffic_train(compare_with_regular = False):
    logger.setup(exp_dir, os.path.join(exp_dir, 'dqn_log.txt'), 'debug')
    num_episodes = 500
    max_epLength = 1800
    env = traffic_env()
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    dqn = DeepQNetWork(env.n_act, saving_loading=True)
    start_training_step = 50
    training_step = 0
    n_loss = []
    total_reward = []
    n_waiting_time = []
    n_arrived = []
    n_reward = []
    n_mean_reward = []
    n_std_reward = []
    vehicle_num = 1795
    per_waiting_time = []
    duration = 6
    for ep in range(num_episodes):
        # vehicle_num = env.generateTrips_e(max_epLength)
        Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
        env.reset(env.sumoCmd)
        reward = []
        env.start(Record.record)
        s = np.zeros((16, 20, 2))
        a_before = np.random.randint(0,2)
        a = a_before
        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[a_before])
        new_time_flag = False
        transition_flag = False
        now_step = 0
        time_step = 0
        sp_ = [s, a]
        sp = sp_
        tqdm_e = tqdm(range(max_epLength), desc='duration', leave=True, unit=" episodes")
        for t in tqdm_e:
            if new_time_flag == True:
                new_time_flag = False
                a = dqn.choose_action(sp)
                time_step = 0
                if a != a_before:
                    a_before = a
                    now_step = 0
                    transition_flag = True
            if transition_flag == True:
                now_step = env.transition(now_step, a)
                if now_step == 33:
                    time_step = 1
                    transition_flag = False
                    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[a])
            else:
                time_step += 1
            Record.simulation_step()
            if time_step == duration:
                r = env.get_RewardBaseRecord(Record.record, duration)
                s_ = env.get_state(DRQN=False)
                sp_ = [s_, a]
                dqn.store_transition(sp, a, r, sp_)
                sp = sp_
                new_time_flag = True
                if dqn.Buffer.count() > start_training_step:
                    training_step += 1
                    dqn.learn()
                    n_loss.append(dqn.cost)
                reward.append(r)
        env.end()
        # if (ep % 400 == 0 or ep == (num_episodes-1)) and ep > 0:
        #     utils.make_video(env, dqn, max_epLength, 'traffic_ep_' + str(ep) + '.avi')

        ### calculate the waiting time
        waiting_time = Record.calc_waiting()
        n_waiting_time.append(waiting_time)
        # arrived vehicle num
        arrived = Record.arrived_vehicle
        n_arrived.append(arrived)
        logger.debug(' train '  + ' meantime : ' + str(waiting_time) + ' arrived : ' + str(arrived))
        total_reward.append(sum(reward))
        ### calculate the reward
        n_reward.append(sum(reward))
        n_mean_reward.append(np.mean(reward))
        n_std_reward.append(np.std(reward))

        if dqn.saving_or_loading == True and ep > 100 and min(n_waiting_time) == waiting_time:
            dqn.model.save_weights(filepath=dqn.save_path)
        ### drawing the loss
        if len(n_loss) > 10 and ep % 100 == 0:
            utils.plot_line(n_loss, name='Loss' + str(ep), path=exp_dir, show=True)
        if ep % 50 == 0 and ep > 0:
            utils.plot_dot(n_waiting_time, name='waiting_time_' + str(ep), path=exp_dir, show=True)
        per_waiting_time.append(Record.record['vehicle']['waittime'][:, -3:-1])
    result = {}
    result['n_reward'] = n_reward
    result['n_arrived'] = n_arrived
    result['n_mean_reward'] = n_mean_reward
    result['n_std_reward'] = n_std_reward
    result['n_waiting_time'] = n_waiting_time
    result['n_loss'] = n_loss
    result['percar'] = per_waiting_time
    utils.plot_result(result, path=exp_dir)
    utils.plot_line(n_loss, name='Loss', path=exp_dir)
    utils.plot_line(n_waiting_time, name='Waiting time', path=exp_dir)
    with open(os.path.join(exp_dir, 'dqn_traffic_data.pkl'), "wb") as f:
        pickle.dump(result, f)
    ### 关闭ubuntu系统
    # command = 'shutdown +1'
    # os.system(command)

def run_main():
    traffic_train()
if __name__ == "__main__":
    run_main()







