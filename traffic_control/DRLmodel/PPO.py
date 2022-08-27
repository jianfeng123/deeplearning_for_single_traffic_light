import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue
from myqueue import replay_buffer
from itertools import chain
import os
## PPO Parameter
EPSILON = 0.2

class PPO(object):
    def __init__(self, n_actions=13,
                 n_features=0,
                 c_learning_rate=0.0003,
                 a_learning_rate=0.0001,
                 reward_decay=0.9,
                 e_greedy=0.95,
                 tua = 0.001,
                 c_update_steps=10,
                 a_update_steps=10,
                 memory_size = 10000,
                 batch_size = 32,
                 saving_loading = False
                 ):                                        # 10-60秒
        self.n_act = n_actions
        self.n_fea = n_features
        self.c_lr = c_learning_rate
        self.a_lr = a_learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.decay_update = tua
        self.c_update_s = c_update_steps
        self.a_update_s = a_update_steps
        self.memory_size = memory_size
        self.sess = tf.Session()
        self.road_size = 48

        self.s_p = tf.placeholder(tf.float32, [None, self.road_size, self.road_size, 1], 'state_p')
        self.s_v = tf.placeholder(tf.float32, [None, self.road_size, self.road_size, 1], 'state_v')
        self.s_phase = tf.placeholder(tf.float32, [None, 4], 'light_phase')
        self.Buffer = replay_buffer(self.memory_size)
        self.batch_size = batch_size
        ## critic
        with tf.variable_scope('critic'):
            layer_p1 = tf.layers.conv2d(self.s_p,filters=16, kernel_size=4, strides=2, activation=tf.nn.relu, trainable=True)
            layer_p2 = tf.layers.conv2d(layer_p1, filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, trainable=True)
            layer_v1 = tf.layers.conv2d(self.s_v,filters=16, kernel_size=4, strides=2, activation=tf.nn.relu, trainable=True)
            layer_v2 = tf.layers.conv2d(layer_v1, filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, trainable=True)
            layer_p2_flatten = tf.contrib.layers.flatten(layer_p2)
            layer_v2_flatten = tf.contrib.layers.flatten(layer_v2)
            layer_combine = tf.concat([layer_p2_flatten, layer_v2_flatten, self.s_phase], axis=-1)
            layer1 = tf.layers.dense(layer_combine, 64, tf.nn.relu, trainable=True)
            layer2 = tf.layers.dense(layer1, 32, tf.nn.relu, trainable=True)
            self.v = tf.layers.dense(layer2, 1, tf.nn.relu, trainable=True)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(self.c_lr).minimize(self.closs)
        ## actor
        self.pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]
        self.tfa = tf.placeholder(tf.int32, [None, ], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        a_indices = tf.stack([tf.range(tf.shape(self.tfa)[0], dtype=tf.int32), self.tfa], axis=1)
        pi_prob = tf.gather_nd(params=self.pi, indices=a_indices)   # shape=(None, )
        oldpi_prob = tf.gather_nd(params=oldpi, indices=a_indices)  # shape=(None, )
        ratio = pi_prob/oldpi_prob
        surr = ratio * self.tfadv                       # surrogate loss
        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))
        self.a_train_op = tf.train.AdamOptimizer(self.a_lr).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())
        if saving_loading == True:
            self.save_dir = '../checkpoints/traffic_ppo'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir, exist_ok=True)
            self.save_path = os.path.join(self.save_dir, 'best_validation')
            self.load_model(self.save_dir)
        self.saving_or_loading = saving_loading
    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful
        # update actor
        [self.sess.run(self.a_train_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(self.a_update_s)]
        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(self.c_update_s)]

    def store_transition(self, state, action, reward, state_, phase):
        self.Buffer.add_ppo(state, action, reward, state_, phase)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            layer_p1 = tf.layers.conv2d(self.s_p,filters=16, kernel_size=4, strides=2, activation=tf.nn.relu, trainable=trainable)
            layer_p2 = tf.layers.conv2d(layer_p1, filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, trainable=trainable)
            layer_v1 = tf.layers.conv2d(self.s_v,filters=16, kernel_size=4, strides=2, activation=tf.nn.relu, trainable=trainable)
            layer_v2 = tf.layers.conv2d(layer_v1, filters=32, kernel_size=2, strides=1, activation=tf.nn.relu, trainable=trainable)
            layer_p2_flatten = tf.contrib.layers.flatten(layer_p2)
            layer_v2_flatten = tf.contrib.layers.flatten(layer_v2)
            layer_combine = tf.concat([layer_p2_flatten, layer_v2_flatten, self.s_phase], axis=-1)
            layer1 = tf.layers.dense(layer_combine, 64, tf.nn.relu, trainable=True)
            layer2 = tf.layers.dense(layer1, 32, tf.nn.relu, trainable=True)
            a_prob = tf.layers.dense(layer2, self.n_act, tf.nn.softmax, trainable=trainable)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return a_prob, params

    def choose_action(self, s, phase=0):  # run by a local
        s = np.reshape(s, newshape=[1, self.road_size, self.road_size, 2])
        s1 = np.reshape(s[:, :, :, 0], newshape=[1, self.road_size, self.road_size, 1])
        s2 = np.reshape(s[:, :, :, 1], newshape=[1, self.road_size, self.road_size, 1])
        s_a = np.zeros((1,4))
        s_a[0][4] = 1
        prob_weights = self.sess.run(self.pi, feed_dict={self.s_p: s1, self.s_v: s2, self.s_phase:s_a})
        action = np.random.choice(range(prob_weights.shape[1]),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action
    def load_model(self, path):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(path)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old weights!")
    def get_v(self, s):
        # if s.ndim < 2: s = s[np.newaxis, :]
        v = self.sess.run(self.v, {self.tfs: s})
        return v
    def learn(self):
        bran_batch = self.Buffer.get_Batch(self.batch_size)
        state = [batch[0] for batch in bran_batch]
        action = [batch[1] for batch in bran_batch]
        reward = [batch[2] for batch in bran_batch]
        state_ = [batch[3] for batch in bran_batch]
        q_next = self.get_v(state_)
        q_next = np.array(list(chain.from_iterable(q_next)))
        tem_q = np.array(reward) + self.gamma*q_next
        tem_q = np.reshape(tem_q, newshape=[self.batch_size, 1])
        self.update(state, action, tem_q)



