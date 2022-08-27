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
# mod_dir = os.path.join(data_dir, args.model_name)

def Compare_DQN(env=None, vehicle_num=1000, max_epLength=1800, dqn=None):
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    # vehicle_num = env.generateTrips_e(max_epLength)
    env.reset(env.sumoCmd)
    Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
    env.start(Record.record)
    s = np.zeros((16, 20, 2))
    a_before = np.random.randint(0,2)
    a = a_before
    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[a_before])
    new_time_flag = False
    transition_flag = False
    now_step = 0
    time_step = 0
    duration = 6
    sp_ = [s, a]
    sp = sp_
    for t in range(max_epLength):
        if new_time_flag == True:
            new_time_flag = False
            a = dqn.choose_action(sp, training=False)
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
            s_ = env.get_state(DRQN=False)
            sp_ = [s_, a]
            sp = sp_
            new_time_flag = True
    env.end()

    ### calculate the waiting time
    waiting_time = Record.calc_waiting()
    arrived = Record.arrived_vehicle
    exists = Record.record['vehicle']['number']
    stop = Record.get_stop_times()
    return waiting_time, arrived, exists, stop
    ### drawing the loss


    ### 关闭ubuntu系统
    # command = 'shutdown +1'
    # os.system(command)






