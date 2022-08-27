import os
import sys
import argparse
import time
from logger import logger
from traffic_env import traffic_env
from DRLmodel.DQN import DeepQNetWork as DQN
import shutil
import numpy as np
import traci
import pickle
import utils
from tqdm import tqdm
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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
                 e_greedy_increment=0.0005,
                 DDQN = False,
                 saving_loading = False,
                 save_path = '../checkpoints/traffic_ca',
                 lstm_size=128,
                 n_frames=4
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
        self.lstm_size = lstm_size
        # self.road_size = 48
        # self.state_size = 48
        self.size0 = 16
        self.size1 = 20
        # self.state_size = (16, )
        # self.state_size = (self.road_size, self.road_size, 1)
        self.state_size = (n_frames, self.size0, self.size1, 1)
        self.n_frames = n_frames
        self.DDQN = DDQN
        self.cost_hist = []
        ## 网络搭建
        self.Buffer = replay_buffer(self.memory_size)
        self.model = self._build_net()
        self.target_model = self._build_net()

        ### 采用此种更新方式更容易收敛
        if saving_loading == True:
            self.save_dir = save_path
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, 'weights.h5')
        self.saving_or_loading = saving_loading
        self.optimizer = self.optimizer()
    def load_model(self):
        try:
            self.model.load_weights(self.save_path)
            print("Successfully loaded!!!")
        except:
            print("Could not find old weights!")
    def _build_net(self):
        # input = Input(shape=(self.state_size,))
        # shared = Dense(256, activation='relu')(input)
        # phase = Input(shape=(1,), dtype='int32')
        # embedding = keras.layers.Embedding(input_dim=4, output_dim=8, input_length=1)(phase)
        # embedding = Flatten()(embedding)
        # shared = concatenate([shared, embedding])
        # shared = Dense(128, activation='relu')(shared)
        # shared = Dense(64, activation='relu')(shared)
        # # agent i
        # value = Dense(self.n_act, activation='linear')(shared)
        # model = keras.models.Model(inputs=[input, phase], outputs=value)
        # # keras.utils.plot_model(model, to_file='../modelPlot/drqn1.png',show_shapes=True)
        # return model
        input = Input(shape=self.state_size)
        shared = TimeDistributed(Convolution2D(32, (4, 4), strides=(2, 2), activation='relu'))(input)
        shared = TimeDistributed(Convolution2D(64, (2, 2), strides=(1, 1), activation='relu'))(shared)
        # shared = TimeDistributed(Convolution2D(128, (2, 2), strides=(1, 1), activation='relu'))(shared)
        flatten = TimeDistributed(Flatten())(shared)
        phase = Input(shape=(4, 1), dtype='int32')
        embedding = keras.layers.Embedding(input_dim=4, output_dim=4, input_length=1)(phase)
        embedding = TimeDistributed(Flatten())(embedding)
        flatten = concatenate([flatten, embedding])
        # agent P
        lstm_layer_p = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        dense_p = Dense(64, activation='relu')(lstm_layer_p)
        dense_p = Dense(32, activation='relu')(dense_p)
        value_p = Dense(self.n_act, activation='linear')(dense_p)
        # agent A
        output = value_p
        model = keras.models.Model(inputs=[input, phase], outputs=output)
        # keras.utils.plot_model(model, to_file='../modelPlot/drqn1.png',show_shapes=True)
        return model

        # pinput = Input(shape=self.state_size)
        # sp = Convolution2D(16, (4, 4), strides=(2, 2), activation='relu')(pinput)
        # sp = Convolution2D(32, (2, 2), strides=(1, 1), activation='relu')(sp)
        # vinput = Input(shape=self.state_size)
        # sv = Convolution2D(16, (4, 4), strides=(2, 2), activation='relu')(vinput)
        # sv = Convolution2D(32, (2, 2), strides=(1, 1), activation='relu')(sv)
        # pflatten = Flatten()(sp)
        # vflatten = Flatten()(sv)
        # ainput = Input(shape=(1,), dtype='int32')
        # aonehot = Lambda(lambda i: K.one_hot(i,4))(ainput)
        # aonehot = Flatten()(aonehot)
        # sinput = concatenate([pflatten, vflatten, aonehot])
        # sinput = Dense(128, activation='relu')(sinput)
        # sinput = Dense(64, activation='relu')(sinput)
        # out = Dense(self.n_act, activation='linear')(sinput)
        # model = keras.models.Model(inputs=[pinput, vinput, ainput], outputs=out)
        # return model
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
        train = K.function([self.model.input[0], self.model.input[1], a, y], [loss], updates=updates)
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
        state = [batch[0] for batch in s]
        phase = [batch[1] for batch in s]
        sn = np.reshape(state, newshape=[1, self.n_frames, self.size0, self.size1, 1])
        sp = np.reshape(phase, newshape=[1, self.n_frames, 1])
        q_values = self.model.predict([sn, sp])
        return q_values
    def learn_forward_t(self, s_, phase):
        s_ = np.reshape(s_ , newshape=[-1, self.state_size])
        phase = np.reshape(phase, newshape=[-1, 1])
        # s = np.reshape(s_, newshape=[self.batch_size, self.road_size, self.road_size, 2])
        # S1 = np.reshape(s[:, :, :, 0], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # S2 = np.reshape(s[:, :, :, 1], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        q_value = self.target_model.predict([s_, phase])
        return q_value
    def learn_forward_e(self, s, phase):
        s = np.reshape(s , newshape=[-1, self.state_size])
        phase = np.reshape(phase, newshape=[-1, 1])
        # s = np.reshape(s_, newshape=[self.batch_size, self.road_size, self.road_size, 2])
        # S1 = np.reshape(s[:, :, :, 0], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # S2 = np.reshape(s[:, :, :, 1], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        q_value = self.target_model.predict([s, phase])
        # s = np.reshape(s, newshape=[self.batch_size, self.road_size, self.road_size, 2])
        # S1 = np.reshape(s[:, :, :, 0], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # S2 = np.reshape(s[:, :, :, 1], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # q_value = self.model.predict([S1, S2, a])
        return q_value

    def learn(self):
        bran_batch = self.Buffer.get_Batch(self.batch_size)
        state = [batch[0][0] for batch in bran_batch]
        state = np.reshape(state, newshape=[-1, self.state_size])
        phase = [batch[0][1] for batch in bran_batch]
        phase = np.array(phase)
        action = np.array([batch[1] for batch in bran_batch])
        reward = [batch[2] for batch in bran_batch]
        state_ = [batch[3][0] for batch in bran_batch]
        state_ = np.reshape(state_, newshape=[-1, self.state_size])
        phase_ = [batch[3][1] for batch in bran_batch]
        phase_ = np.array(phase_)
        q_next_t = self.target_model.predict([state_, phase_])
        if self.DDQN:
            q_next_e = self.model.predict([state_, phase_])
            max_act_next = np.argmax(q_next_e, axis=1)
            selected_q_next = q_next_t[range(self.batch_size), max_act_next]
        else:
            selected_q_next = np.max(q_next_t, axis=1)
        q_target = reward + self.gamma*selected_q_next
        # action = np.reshape(action, newshape=[-1, 1])
        phase = np.reshape(phase, newshape=[-1,1])
        self.cost  = self.optimizer([state, phase, action, q_target])
        self.update_target_model()
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        return self.cost
        # bran_batch = self.Buffer.get_Batch(self.batch_size)
        # state = [batch[0][0] for batch in bran_batch]
        # # state = np.reshape(state, newshape=[-1, self.n_frames, self.road_size, self.road_size, 1])
        # state = np.reshape(state, newshape=[-1, self.state_size])
        # phase = [batch[0][1] for batch in bran_batch]
        # phase = np.array(phase)
        # action = np.array([batch[1] for batch in bran_batch])
        # reward = [batch[2] for batch in bran_batch]
        # state_ = [batch[3][0] for batch in bran_batch]
        # # state_ = np.reshape(state_, newshape=[-1, self.n_frames, self.road_size, self.road_size, 1])
        # state_ = np.reshape(state_, newshape=[-1, self.state_size])
        # phase_ = [batch[3][1] for batch in bran_batch]
        # phase_ = np.array(phase_)
        # q_next_t = self.learn_forward_t(state_, phase_)
        # if self.DDQN:
        #     q_next_e = self.learn_forward_e(state_, phase_)
        #     max_act_next = np.argmax(q_next_e, axis=1)
        #     selected_q_next = q_next_t[range(len(q_next_t)), max_act_next]
        # else:
        #     selected_q_next = np.max(q_next_t, axis=1)
        # q_target = reward + self.gamma*selected_q_next
        # s = np.reshape(state, newshape=[self.batch_size, self.road_size, self.road_size, 2])
        # S1 = np.reshape(s[:, :, :, 0], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # S2 = np.reshape(s[:, :, :, 1], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # phase = np.reshape(phase, newshape=[-1, 1])
        # self.cost = self.optimizer([S1, S2, phase, action, q_target])
        # self.update_target_model()
        # # increasing epsilon
        # self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max


def Compare_CA(env=None, vehicle_num=1000, max_epLength=1800, dqn=None):
    rgy_state = ['rrrrrGGGGgrrrrrGGGGg', 'rrrrrrrrrGrrrrrrrrrG', 'GGGGgrrrrrGGGGgrrrrr', 'rrrrGrrrrrrrrrGrrrrr']
    duration = 6
    env.reset(env.sumoCmd)
    Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
    env.start(Record.record)
    s = np.zeros((16, 20))
    a_before = 0
    a = a_before
    phase = 0
    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[phase])
    new_time_flag = False
    transition_flag = False
    now_step = 0
    time_step = 0
    sp = s
    sp_ = sp
    transition_sp = [[sp, phase] for _ in range(dqn.n_frames)]
    transition_sp_ = [[sp_, phase] for _ in range(dqn.n_frames)]
    tqdm_e = tqdm(range(max_epLength), desc='CA', leave=True, unit=" episodes")
    for t in tqdm_e:
        if new_time_flag == True:
            new_time_flag = False
            a = dqn.choose_action(transition_sp, training=False)
            if a == 1:
                # a_before = a
                now_step = 0
                time_step = 0
                transition_flag = True
        if transition_flag == True:
            now_step = env.transitionF(now_step, phase)
            if now_step == 7:
                time_step = 1
                transition_flag = False
                phase += 1
                if phase >= 4:
                    phase = 0
                traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[phase])
        else:
            time_step += 1
        Record.simulation_step()
        if (time_step - 6) % duration == 0 and time_step >= 6:
            s_ = env.get_state(reward=False)
            transition_sp_ = [[s_, phase]] + transition_sp_[0:(dqn.n_frames - 1)]
            # r = env.get_Reward_queue_c(phase)
            transition_sp = transition_sp_
            new_time_flag = True
    env.end()

    ### calculate the waiting time
    waiting_time = Record.calc_waiting()
    arrived = Record.arrived_vehicle
    exists = Record.record['vehicle']['number']
    stop = Record.get_stop_times()
    return waiting_time, arrived, exists, stop






