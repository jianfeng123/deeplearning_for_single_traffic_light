import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from myqueue import replay_buffer
from itertools import chain
import os
import sys
import argparse
import time
from logger import logger
from traffic_env import traffic_env
import shutil
import traci
import pickle
import utils
import copy
from tensorflow.contrib.keras.api.keras.models import Sequential
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Convolution2D, LSTM, TimeDistributed, Input, concatenate
from tensorflow.contrib.keras.api.keras import backend as K
from DRLmodel.DQN import DeepQNetWork as DQN
from tqdm import tqdm
from Compare_Main.finished_DQN import Compare_DQN
from Compare_Main.finished_CA import DeepQNetWork as CA
from Compare_Main.finished_CA import Compare_CA
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
## PPO Parameter
EPSILON = 0.2

class DeepRQNetWork(object):
    def __init__(self, n_actions = 2,
                 n_features = 0,
                 learning_rate=0.0002,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 tua = 0.001,
                 memory_size = 10000,
                 batch_size = 32,
                 e_greedy_increment=0.0005,
                 DDQN = True,
                 saving_loading = False,
                 save_path = '../checkpoints/traffic_keras_dqn',
                 lstm_size=128,
                 n_frames=4,
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
        self.epsilon_p = 0 if e_greedy_increment is not None else self.epsilon_max
        self.epsilon_a = 0 if e_greedy_increment is not None else self.epsilon_max
        self.batch_size = batch_size
        self.DDQN = DDQN
        self.lstm_size = lstm_size
        self.save_path = save_path
        self.cost_hist = []
        self.road_size = 48
        self.state_size = 48
        self.size0 = 16
        self.size1 = 20
        # self.state_size = (16, )
        self.n_frames = n_frames
        self.state_size = (self.n_frames, self.size0, self.size1, 1)
        self.n_frames = n_frames
        self.sess = tf.Session()
        keras.backend.set_session(self.sess)
        self.Buffer_p = replay_buffer(self.memory_size)
        self.Buffer_a = replay_buffer(self.memory_size)
        ## 网络搭建
        self.model = self._build_net()
        self.target_model = self._build_net()
        if saving_loading == True:
            self.save_dir = self.save_path
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
        # share network
        # input = Input(shape=(self.state_size, ))
        # shared = Dense(256, activation='relu')(input)
        # # flatten = concatenate([shared, embedding])
        # # agent P
        # # lstm_layer_p = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        # dense_p = Dense(128, activation='relu')(shared)
        # dense_p = Dense(64, activation='relu')(dense_p)
        # value_p = Dense(self.n_act, activation='linear')(dense_p)
        # # agent A
        # # lstm_layer_a = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        # dense_a = Dense(128, activation='relu')(shared)
        # dense_a = Dense(64, activation='relu')(dense_a)
        # value_a = Dense(self.n_act, activation='linear')(dense_a)
        # shared = Convolution2D(32, (8, 8), strides=(4, 4), activation='relu')(input)
        # shared = Convolution2D(64, (4, 4), strides=(2, 2), activation='relu')(shared)
        # shared = Convolution2D(128, (3, 3), strides=(1, 1), activation='relu')(shared)
        # flatten = Flatten()(shared)
        # # agent P
        # # lstm_layer_p = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        # dense_p = Dense(64, activation='tanh')(flatten)
        # dense_p = Dense(32, activation='relu')(dense_p)
        # value_p = Dense(self.n_act, activation='linear')(dense_p)
        # # agent A
        # # lstm_layer_a = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        # dense_a = Dense(64, activation='tanh')(flatten)
        # dense_a = Dense(32, activation='relu')(dense_a)
        # value_a = Dense(self.n_act, activation='linear')(dense_a)
        input = Input(shape=self.state_size)
        shared = TimeDistributed(Convolution2D(32, (4, 4), strides=(2, 2), activation='relu'))(input)
        shared = TimeDistributed(Convolution2D(64, (2, 2), strides=(1, 1), activation='relu'))(shared)
        flatten = TimeDistributed(Flatten())(shared)
        # agent P
        lstm_layer_p = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        dense_p = Dense(64, activation='relu')(lstm_layer_p)
        dense_p = Dense(32, activation='relu')(dense_p)
        value_p = Dense(self.n_act, activation='linear')(dense_p)
        # agent A
        lstm_layer_a = LSTM(units=self.lstm_size, activation='tanh')(flatten)
        dense_a = Dense(64, activation='relu')(lstm_layer_a)
        dense_a = Dense(32, activation='relu')(dense_a)
        value_a = Dense(self.n_act, activation='linear')(dense_a)

        output = concatenate([value_p, value_a])
        model = keras.models.Model(inputs=[input], outputs=output)
        # keras.utils.plot_model(model, to_file='../modelPlot/drqn1.png',show_shapes=True)
        return model
        # keras.utils.plot_model(model, to_file='../modelPlot/drqn1.png',show_shapes=True)
    def update_target_model(self):
        weights  = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.decay_update + target_weights[i] * (1 - self.decay_update)
        self.target_model.set_weights(target_weights)
    def store_transition_p(self, state, action, reward, state_):
        self.Buffer_p.add(state, action, reward, state_)

    def store_transition_a(self, state, action, reward, state_):
        self.Buffer_a.add(state, action, reward, state_)

    def choose_action_p(self, s, training=True):
        if training:
            if np.random.uniform() < self.epsilon_p:
                q_value = self.forward_e(s)[0]
                action = np.argmax(q_value[0:2])
            else:
                action = np.random.randint(0, self.n_act)
        else:
            q_value = self.forward_e(s)[0]
            action = np.argmax(q_value[0:2])
        return action
    def choose_action_a(self, s, training=True):
        if training:
            if np.random.uniform() < self.epsilon_a:
                q_value = self.forward_e(s)[0]
                action = np.argmax(q_value[-2:])
            else:
                action = np.random.randint(0, self.n_act)
        else:
            q_value = self.forward_e(s)[0]
            action = np.argmax(q_value[-2:])
        return action

    def forward_e(self, s):
        # s = np.reshape(s, newshape=[-1, self.n_frames, self.road_size, self.road_size, 1])
        # state = [state[0] for state in s]
        # phase = [state[1] for state in s]
        sn = np.reshape(s, newshape=[-1, self.n_frames, self.size0, self.size1, 1])

        # sp = np.reshape(s[1], newshape=[-1, 1])
        # s = np.reshape(s, newshape=[-1, self.road_size, self.road_size, 1])
        q_values = self.model.predict([sn])
        return q_values
    def optimizer(self):
        a = K.placeholder(shape=[None, ], dtype='int32')
        y = K.placeholder(shape=[None, ], dtype='float32')
        prediction = self.model.output
        a_one_hot = K.one_hot(a, 4)
        q_value = K.sum(prediction * a_one_hot, axis=1)
        error = K.abs(y - q_value)
        quadratic_part = K.clip(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = K.mean(0.5 * K.square(quadratic_part) + linear_part)
        optimizer = keras.optimizers.Adam(lr=self.lr)
        updates = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        train = K.function([self.model.input, a, y], [loss, error], updates=updates)
        return train

    def learnp(self):
        bran_batch = self.Buffer_p.get_Batch(self.batch_size)
        # state = [[bitch[0] for bitch in batch[0]] for batch in bran_batch]
        state = [batch[0] for batch in bran_batch]
        state = np.reshape(state, newshape=[-1, self.n_frames, self.size0, self.size1, 1])
        # state = np.reshape(state, newshape=[-1, self.road_size, self.road_size, 1])
        action = [batch[1] for batch in bran_batch]
        reward = [batch[2] for batch in bran_batch]
        # state_ = [[bitch[0] for bitch in batch[0]] for batch in bran_batch]
        state_ = [batch[3] for batch in bran_batch]
        state_ = np.reshape(state_, newshape=[-1, self.n_frames, self.size0, self.size1, 1])
        # state_ = np.reshape(state_, newshape=[-1, self.road_size, self.road_size, 1])
        q_next_t = self.target_model.predict([state_])[:, 0:2]
        if self.DDQN:
            q_next_e = self.model.predict([state_])[:, 0:2]
            max_act_next = np.argmax(q_next_e, axis=1)
            selected_q_next = q_next_t[range(self.batch_size), max_act_next]
        else:
            selected_q_next = np.max(q_next_t, axis=1)
        q_target = reward + self.gamma*selected_q_next
        self.cost, abs_errors = self.optimizer([state, action, q_target])
        self.update_target_model()
        # increasing epsilon
        self.epsilon_p = self.epsilon_p + self.epsilon_increment if self.epsilon_p < self.epsilon_max else self.epsilon_max
    def learna(self):
        bran_batch = self.Buffer_a.get_Batch(self.batch_size)
        # state = [[bitch[0] for bitch in batch[0]] for batch in bran_batch]
        state = [batch[0] for batch in bran_batch]
        state = np.reshape(state, newshape=[-1, self.n_frames, self.size0, self.size1, 1])
        # state = np.reshape(state, newshape=[-1, self.road_size, self.road_size, 1])
        action = [batch[1] for batch in bran_batch]
        reward = [batch[2] for batch in bran_batch]
        # state_ = [[bitch[0] for bitch in batch[0]] for batch in bran_batch]
        state_ = [batch[3] for batch in bran_batch]
        state_ = np.reshape(state_, newshape=[-1, self.n_frames, self.size0, self.size1, 1])
        q_next_t = self.target_model.predict([state_])[:,-2:]
        if self.DDQN:
            q_next_e = self.model.predict([state_])[:,-2:]
            max_act_next = np.argmax(q_next_e, axis=1)
            selected_q_next = q_next_t[range(self.batch_size), max_act_next]
        else:
            selected_q_next = np.max(q_next_t, axis=1)
        q_target = reward + self.gamma*selected_q_next
        self.cost, abs_errors  = self.optimizer([state, action, q_target])
        self.update_target_model()
        # increasing epsilon
        self.epsilon_a = self.epsilon_a + self.epsilon_increment if self.epsilon_a < self.epsilon_max else self.epsilon_max


parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='traffic_Compare_Double_drqn')
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

def traffic_train():
    logger.setup(exp_dir, os.path.join(exp_dir, 'DDRQN_log.txt'), 'debug')
    num_episodes = 500
    max_epLength = 3600
    vehicle_num = 1795
    env = traffic_env(vehicle_num=1795)    ### 重新生成新的环境，可以将字符串中train删除
    # vehicle_num = env.generateTrips_e(max_epLength)
    ## P for main road , A for sub road
    # rgy_state_p = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    rgy_state_p = ['rrrrrGGGGgrrrrrGGGGg', 'GGGGgrrrrrGGGGgrrrrr']
    rgy_state_a = ['rrrrrrrrrGrrrrrrrrrG', 'rrrrGrrrrrrrrrGrrrrr']

    sess = tf.Session()
    keras.backend.set_session(sess)

    drqn = DeepRQNetWork(saving_loading=True, save_path='../checkpoints/traffic_CSDRQN')
    dqn = DQN(saving_loading=True)
    ca = CA(saving_loading=True)
    sess.run(tf.global_variables_initializer())
    drqn.load_model()
    dqn.load_model()
    ca.load_model()
    ## Principal
    # ddqn
    ddqn_n_waiting_time = []
    ddqn_n_arrived = []
    ddqn_n_exists = []
    ddqn_n_stop = []
    # fixed
    fixed_n_waiting_time = []
    fixed_n_arrived = []
    fixed_n_exists = []
    fixed_n_stop = []
    # dqn
    dqn_n_waiting_time = []
    dqn_n_arrived = []
    dqn_n_exists = []
    dqn_n_stop = []
    # change action
    ca_n_waiting_time = []
    ca_n_arrived = []
    ca_n_exists = []
    ca_n_stop = []
    durationp = 6
    durationa = 6
    #
    for ep in range(num_episodes):
        p = 0.1 * (int(ep / 20) + 1)
        vehicle_num = env.generateTrips_e(max_epLength, p=p)
        Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
        if ep > 30:
            env.reset(env.sumoCmd)
        else:
            env.reset(env.sumoCmd)
        env.start(Record.record)
        sp = np.zeros((16, 20))
        ap_before = 0                                                    #### 主道路
        ap = ap_before
        aa_before = ap                                                   #### 副道路
        aa = aa_before
        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state_p[ap_before])     #### 开始时的控制
        new_time_flag = False
        transition_flag = False
        Principal = True  ### 主干道 T为主干道, F为副道
        now_step = 0
        time_step_p = 0
        time_step_a = 0
        A_state_first_flag = True
        ## 构建时间序列
        sa = sp
        # transition_sp = sp
        # transition_sp_ = sp
        # transition_sa = sa
        # transition_sa_ = sa
        transition_sp = [sp  for _ in range(drqn.n_frames)]
        transition_sp_ = [sp for _ in range(drqn.n_frames)]
        transition_sa = [sa for _ in range(drqn.n_frames)]
        transition_sa_ = [sa for _ in range(drqn.n_frames)]
        tqdm_e = tqdm(range(max_epLength), desc='reward', leave=True, unit=" episodes")
        for t in tqdm_e:
            if new_time_flag == True:
                new_time_flag = False
                if Principal:
                    ap = drqn.choose_action_p(transition_sp, training=False)
                    time_step_p = 0
                    if ap != ap_before:
                        ap_before = ap
                        now_step = 0
                        transition_flag = True
                else:
                    aa = drqn.choose_action_a(transition_sa, training=False)
                    time_step_a = 0
                    if aa != aa_before:
                        aa_before = aa
                        now_step = 0
                        transition_flag = True
            if transition_flag == True:
                now_step = env.transitionD(now_step, ap, aa, Principal)
                if now_step == 7:
                    Principal = not Principal
                    transition_flag = False
                    if Principal:
                        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state_p[ap])
                        time_step_p = 1
                    else:
                        if A_state_first_flag:
                            A_state_first_flag = False
                            sa = env.get_state(reward=False)
                        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state_a[aa])
                        time_step_a = 1
            else:
                if Principal:
                    if time_step_p % (durationp/2) == 0 and time_step_p > 0:
                        transition_sp_ = [env.get_state(reward=False)] + transition_sp_[0:(drqn.n_frames - 1)]
                    time_step_p += 1
                else:
                    if time_step_a % (durationa/2) == 0 and time_step_a > 0:
                        transition_sa_ = [env.get_state(reward=False)] + transition_sa_[0:(drqn.n_frames - 1)]
                    time_step_a += 1
            Record.simulation_step()
            if time_step_p == durationp:
                sp_ = env.get_state(reward=False)
                transition_sp_ = [sp_] + transition_sp_[0:(drqn.n_frames - 1)]
                # transition_sp_ = sp_
                ## for get reward
                # drqn.store_transition_p(sp, ap, r, sp_)
                transition_sp = transition_sp_
                new_time_flag = True
            if time_step_a == durationa:
                sa_ = env.get_state(reward=False)
                transition_sa_ = [sa_] + transition_sa_[0:(drqn.n_frames - 1)]
                # transition_sa_ = sa_
                # drqn.store_transition_a(sa, aa + 2, r, sa_)
                transition_sa = transition_sa_
                new_time_flag = True
        env.end()
        # with open(os.path.join(exp_dir, 'wait_traffic_data_' + str(ep) + '.pkl'), "wb") as f:
        #     pickle.dump(step_wait, f)
        # if (ep % 400 == 0 or ep == (num_episodes-1)) and ep > 0:
        #     utils.make_video(env, dqn1, max_epLength, 'traffic_ep_' + str(ep) + '.avi')
        ### calculate the waiting time
        waiting_time = Record.calc_waiting()
        ddqn_n_waiting_time.append(waiting_time)
        arrived = Record.arrived_vehicle
        ddqn_stop = Record.get_stop_times()
        ddqn_n_arrived.append(arrived)
        ddqn_n_exists.append(Record.record['vehicle']['number'])
        ddqn_n_stop.append(ddqn_stop)
        if ep % 20 == 0 and ep > 0:
            utils.plot_dot(ddqn_n_waiting_time, name='waiting_time_' + str(ep), path=exp_dir, show=True)
        # save the per vehicle waiting time
        ## fixed
        fixed, fix_arrived, fix_exists, fix_stop = calc_normal_light_compare(env=env, max_epLength=max_epLength, vehicle_num=vehicle_num)
        fixed_n_waiting_time.append(fixed), fixed_n_arrived.append(fix_arrived), fixed_n_exists.append(fix_exists), fixed_n_stop.append(fix_stop)
        ## dqn
        dqn_waiting_time,dqn_arrived, dqn_exists, dqn_stop = Compare_DQN(env=env, max_epLength=max_epLength, vehicle_num=vehicle_num, dqn=dqn)
        dqn_n_waiting_time.append(dqn_waiting_time), dqn_n_arrived.append(dqn_arrived), dqn_n_exists.append(dqn_exists), ddqn_n_stop.append(dqn_stop)
        ## ca  change phase
        ca_waiting_time, ca_arrived, ca_exists, ca_stop =  Compare_CA(env=env, vehicle_num=vehicle_num, max_epLength=max_epLength, dqn=ca)
        ca_n_waiting_time.append(ca_waiting_time), ca_n_arrived.append(ca_arrived), ca_n_exists.append(ca_exists), ca_n_stop.append(ca_stop)
        # logger.debug(' train ' + str(ep) + ' ddqnwaitingtime : ' + str(waiting_time) + ' fixed : ' + str(fixed) + ' dqnwaitingtime : ' + str(dqn_waiting_time) + ' ca: ' + str(ca_waiting_time))
        logger.debug(' train ' + str(ep) + ' P: ' + str(p) + ' ddqnwaitingtime : ' + str(waiting_time) + ' fixed : ' + str(
            fixed) + ' dqnwaitingtime : ' + str(dqn_waiting_time) + ' ca: ' + str(ca_waiting_time))
        logger.debug(' train stop ' + str(ep) + ' P: ' + str(p) + ' ddqn_stop : ' + str(ddqn_stop) + ' fixed : ' + str(
            fix_stop) + ' dqn_stop : ' + str(dqn_stop) + ' ca: ' + str(ca_stop))

        if ep % 20 == 0 and ep > 0:
            result = {}
            result['ddqn_n_waiting_time'] = ddqn_n_waiting_time
            result['ddqn_n_arrived'] = ddqn_n_arrived
            result['ddqn_n_exists'] = ddqn_n_exists
            result['ddqn_n_stop'] = ddqn_n_stop

            result['fixed_n_waiting_time'] = fixed_n_waiting_time
            result['fixed_n_arrived'] = fixed_n_arrived
            result['fixed_n_exists'] = fixed_n_exists
            result['fixed_n_stop'] = fixed_n_stop

            result['dqn_n_waiting_time'] = dqn_n_waiting_time
            result['dqn_n_arrived'] = dqn_n_arrived
            result['dqn_n_exists'] = dqn_n_exists
            result['dqn_n_stop'] = dqn_n_stop

            result['ca_n_waiting_time'] = ca_n_waiting_time
            result['ca_n_arrived'] = ca_n_arrived
            result['ca_n_exists'] = ca_n_exists
            result['ca_n_stop'] = ca_n_stop

            # utils.plot_result(result, path=exp_dir)
            utils.plot_line(ddqn_n_waiting_time, name='DDQN Waiting time', path=exp_dir)
            utils.plot_line(fixed_n_waiting_time, name='Fixed Waiting time', path=exp_dir)
            utils.plot_line(dqn_n_waiting_time, name='DQN Waiting time', path=exp_dir)
            utils.plot_line(ca_n_waiting_time, name='CA Waiting time', path=exp_dir)
            with open(os.path.join(exp_dir, 'compare_traffic_data' + str(ep) + '.pkl'), "wb") as f:
                pickle.dump(result, f)
    ### 关闭ubuntu系统
    # command = 'shutdown +1'
    # os.system(command)

def calc_normal_light_compare(env=None, max_epLength=1000, vehicle_num=1000):
    env.reset(env.sumoCmd)
    record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'yyyygrrrrryyyygrrrrr', 'rrrrGrrrrrrrrrGrrrrr', 'rrrryrrrrrrrrryrrrrr',
                 'rrrrrGGGGgrrrrrGGGGg', 'rrrrryyyygrrrrryyyyg', 'rrrrrrrrrGrrrrrrrrrG', 'rrrrrrrrryrrrrrrrrry']
    p_t = 25
    a_t = 20
    y_t = 6
    st = [p_t, y_t, a_t, y_t, p_t, y_t, a_t, y_t]
    state_t = []
    t = 0
    for i in range(8):
        t += st[i]
        state_t.append(t)
    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[0])
    count = 0
    now_step = 0
    for i in range(max_epLength):
        record.simulation_step()
        count += 1
        if count == state_t[now_step]:
            now_step += 1
            if now_step == 8:
                now_step = 0
                count = 0
            traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[now_step])
    env.end()
    waiting_time = record.calc_waiting()
    fixed_arrived_vehicle = record.arrived_vehicle
    fixed_exits_vehicle = record.record['vehicle']['number']
    fixed_stop_times = record.get_stop_times()
    return waiting_time, fixed_arrived_vehicle, fixed_exits_vehicle, fixed_stop_times

if __name__ == "__main__":
    traffic_train()

