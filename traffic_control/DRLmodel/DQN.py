import tensorflow as tf
import numpy as np
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
                 saving_loading = False,
                 save_path = '../checkpoints/traffic_dqn'
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
        pinput = Input(shape=self.state_size)
        sp = Convolution2D(16, (4, 4), strides=(2, 2), activation='relu')(pinput)
        sp = Convolution2D(32, (2, 2), strides=(1, 1), activation='relu')(sp)
        vinput = Input(shape=self.state_size)
        sv = Convolution2D(16, (4, 4), strides=(2, 2), activation='relu')(vinput)
        sv = Convolution2D(32, (2, 2), strides=(1, 1), activation='relu')(sv)
        pflatten = Flatten()(sp)
        vflatten = Flatten()(sv)
        ainput = Input(shape=(1,), dtype='int32')
        aonehot = Lambda(lambda i: K.one_hot(i, 4))(ainput)
        aonehot = Flatten()(aonehot)
        sinput = concatenate([pflatten, vflatten, aonehot])
        sinput = Dense(128, activation='relu')(sinput)
        sinput = Dense(64, activation='relu')(sinput)
        out = Dense(self.n_act, activation='linear')(sinput)
        model = keras.models.Model(inputs=[pinput, vinput, ainput], outputs=out)
        return model

    def update_target_model(self):
        weights = self.model.get_weights()
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
        train = K.function([self.model.input[0], self.model.input[1], self.model.input[2], a, y], [loss],
                           updates=updates)
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
        SA = np.reshape(s[1], newshape=[-1, 1])
        q_values = self.model.predict([SP, SV, SA])
        return q_values

    def learn_forward_t(self, s_, phase):
        SPV = np.reshape(s_, newshape=[-1, 16, 20, 2])
        SP = np.reshape(SPV[:, :, :, 0], newshape=[-1, 16, 20, 1])
        SV = np.reshape(SPV[:, :, :, 1], newshape=[-1, 16, 20, 1])
        SA = np.reshape(phase, newshape=[-1, 1])
        # s = np.reshape(s_, newshape=[self.batch_size, self.road_size, self.road_size, 2])
        # S1 = np.reshape(s[:, :, :, 0], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        # S2 = np.reshape(s[:, :, :, 1], newshape=[self.batch_size, self.road_size, self.road_size, 1])
        q_value = self.target_model.predict([SP, SV, SA])
        return q_value

    def learn_forward_e(self, s, phase):
        SPV = np.reshape(s[0], newshape=[-1, 16, 20, 2])
        SP = np.reshape(SPV[:, :, :, 0], newshape=[-1, 16, 20, 1])
        SV = np.reshape(SPV[:, :, :, 1], newshape=[-1, 16, 20, 1])
        SA = np.reshape(phase, newshape=[-1, 1])
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
        q_target = reward + self.gamma * selected_q_next
        # action = np.reshape(action, newshape=[-1, 1])
        phase = np.reshape(phase, newshape=[-1, 1])
        SP = np.reshape(state[:, :, :, 0], newshape=[-1, 16, 20, 1])
        SV = np.reshape(state[:, :, :, 1], newshape=[-1, 16, 20, 1])
        self.cost = self.optimizer([SP, SV, phase, action, q_target])
        self.update_target_model()
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        return self.cost


