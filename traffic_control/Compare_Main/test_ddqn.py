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
from tqdm import tqdm
import copy
from tensorflow.contrib.keras.api.keras.models import Sequential
import tensorflow.contrib.keras.api.keras as keras
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Convolution2D, LSTM, TimeDistributed, Input, concatenate
from tensorflow.contrib.keras.api.keras import backend as K
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
## PPO Parameter
EPSILON = 0.2

def test_drqn(max_epLength = 1800, env=None, drqn=None, m_state=True):
    ## P for main road , A for sub road
    # rgy_state_p = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    rgy_state_p = ['rrrrrGGGGgrrrrrGGGGg', 'GGGGgrrrrrGGGGgrrrrr']
    rgy_state_a = ['rrrrrrrrrGrrrrrrrrrG', 'rrrrGrrrrrrrrrGrrrrr']
    # dqna = DeepRQNetWork(env.n_act, saving_loading=True, save_path='../checkpoints/traffic_Ass_drqn', var_scope='Assistant')
    durationp = 6
    durationa = 6
    Record = utils.Record(step_num=max_epLength, vehicle_num=env.vehicle_num)
    env.reset(env.sumoCmd)
    sp = env.start(Record.record)
    if m_state == False:
        sp = env.get_state_q()
    ap_before = np.random.randint(0,2)                               #### 主道路
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
    transition_sp = [sp, ap]
    transition_sp_ = [sp, ap]
    transition_sa = [sa, aa]
    transition_sa_ = [sa, aa]
    # transition_sp = [[sp,ap_before]  for _ in range(drqn.n_frames)]
    # transition_sp_ = [[sp, ap_before] for _ in range(drqn.n_frames)]
    # transition_sa = [[sa, aa_before] for _ in range(drqn.n_frames)]
    # transition_sa_ = [[sa, aa_before] for _ in range(drqn.n_frames)]
    tqdm_e = tqdm(range(max_epLength), desc='reward', leave=True, unit=" episodes")
    for t in tqdm_e:
        if new_time_flag == True:
            new_time_flag = False
            if Principal:
                ap = drqn.choose_action_p(transition_sp)
                time_step_p = 0
                if ap != ap_before:
                    ap_before = ap
                    now_step = 0
                    transition_flag = True
            else:
                aa = drqn.choose_action_a(transition_sa)
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
                        if m_state:
                            sa = env.get_state()
                        else:
                            sa = env.get_state_q()
                    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state_a[aa])
                    time_step_a = 1
        else:
            if Principal:
                # if time_step_p % (durationp/2) == 0 and time_step_p > 0:
                #     transition_sp_ = [[env.get_state_q(), ap]] + transition_sp_[0:(drqn.n_frames - 1)]
                time_step_p += 1
            else:
                # if time_step_a % (durationa/2) == 0 and time_step_a > 0:
                #     transition_sa_ = [[env.get_state_q(), aa]] + transition_sa_[0:(drqn.n_frames - 1)]
                time_step_a += 1
        Record.simulation_step()
        if time_step_p == durationp:
            if m_state == True:
                sp_ = env.get_state()
            else:
                sp_ = env.get_state_q()
            # transition_sp_ = [[sp_, ap]] + transition_sp_[0:(drqn.n_frames - 1)]
            transition_sp_ = [sp_, ap]
            ## for get reward
            # drqn.store_transition_p(sp, ap, r, sp_)
            transition_sp = transition_sp_
            sp = sp_
            new_time_flag = True
        if time_step_a == durationa:
            if m_state == True:
                sa_ = env.get_state()
            else:
                sa_ = env.get_state_q()
            # transition_sa_ = [[sa_, aa]] + transition_sa_[0:(drqn.n_frames - 1)]
            transition_sa_ = [sa_, aa]
            # drqn.store_transition_a(sa, aa + 2, r, sa_)
            transition_sa = transition_sa_
            sa = sa_
            new_time_flag = True
    env.end()
    ### calculate the waiting time
    waiting_time = Record.calc_waiting()
    ### calculate the arrived num
    arrived = Record.arrived_vehicle
    return waiting_time, arrived
    ### 关闭ubuntu系统
    # command = 'shutdown +1'
    # os.system(command)

