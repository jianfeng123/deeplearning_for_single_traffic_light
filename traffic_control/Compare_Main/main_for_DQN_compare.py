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

def traffic_train(compare_with_regular = True):
    logger.setup(exp_dir, os.path.join(exp_dir, 'compare_dqn_log.txt'), 'debug')
    regular_waiting_time = 0
    num_episodes = 2000
    max_epLength = 1000
    env = traffic_env()
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    dqn = DQN(env.n_act, saving_loading=True)
    n_waiting_time = []
    n_regular_time = []

    for ep in range(num_episodes):
        # p = 0.1 * (int(ep / 5) + 1)
        vehicle_num = env.generateTrips_e(max_epLength)
        Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
        env.reset(env.sumoCmd)
        s = env.start(Record.record)
        a_before = np.random.randint(0,2)
        a = a_before
        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[a_before])
        new_time_flag = False
        transition_flag = False
        now_step = 0
        time_step = 0
        for t in range(max_epLength):
            if new_time_flag == True:
                new_time_flag = False
                a = dqn.choose_action(s,a, training=False)                              #### 测试
                time_step = 0
                if a != a_before:
                    a_before = a
                    now_step = 0
                    transition_flag = True
            if transition_flag == True:
                now_step = env.transition(now_step, a)
                if now_step == 23:
                    transition_flag = False
                    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[a])
            else:
                time_step += 1
            Record.simulation_step()
            if time_step == 10:
                s = env.get_state()
                new_time_flag = True
        env.end()
        if compare_with_regular:
            regular_waiting_time = calc_normal_light_compare(env, max_epLength, vehicle_num)
            n_regular_time.append(regular_waiting_time)
        if (ep % 400 == 0 or ep == (num_episodes-1)) and ep > 0:
            utils.make_video(env, dqn, max_epLength, 'traffic_ep_' + str(ep) + '.avi')
        ### calculate the waiting time
        waiting_time = Record.calc_waiting()
        n_waiting_time.append(waiting_time)
        logger.debug(' train ' + str(ep) + ' DQN_waitingtime : ' + str(waiting_time) + ' Regular time : ' + str(regular_waiting_time))
        ### drawing the waiting time
        if ep % 50 == 0 and ep > 0:
            utils.plot_dot(n_waiting_time, name='waiting_time_' + str(ep), path=exp_dir, show=True)
            if compare_with_regular:
                utils.plot_dot(n_regular_time, name='regular_time_' + str(ep), path=exp_dir, show=True)
                utils.plot_compare_dot(n_waiting_time, n_regular_time, name='compare_'+ str(ep), path=exp_dir, show=True)
    result = {}
    result['n_waiting_time'] = n_waiting_time
    result['n_fixed_time'] = n_regular_time
    utils.plot_dot(n_regular_time, name='fixed_time_last', path=exp_dir, show=True)
    utils.plot_dot(n_waiting_time, name='waiting_time_last', path=exp_dir, show=True)
    with open(os.path.join(exp_dir, 'dqn_traffic_data.pkl'), "wb") as f:
        pickle.dump(result, f)
    ### 关闭ubuntu系统
    command = 'shutdown +1'
    os.system(command)

## 定时交通灯，进行比较
def calc_normal_light_compare(env=None, max_epLength=1000, vehicle_num=1000):
    record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
    env.reset(env.sumoCmd)
    for t in range(max_epLength):
        record.simulation_step()
    env.end()
    waiting_time = record.calc_waiting()
    return waiting_time


def run_main():
    # calc_normal_light_alone()
    traffic_train()
if __name__ == "__main__":
    run_main()







