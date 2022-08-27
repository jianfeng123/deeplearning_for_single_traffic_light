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
from DRLmodel.PPO import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='traffic_ppo')
parser.add_argument('--model_name', type=str, default='model')
args = parser.parse_args()

data_dir = '../DATA'
exp_dir = os.path.join(data_dir, args.exp_name)
# mod_dir = os.path.join(data_dir, args.model_name)
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir, exist_ok=True)
else:
    os.makedirs(exp_dir, exist_ok=True)

def traffic_train():
    logger.setup(exp_dir, os.path.join(exp_dir, 'log_PPO.txt'), 'debug')
    num_episodes = 2000
    max_epLength = 1000
    env = traffic_env()
    # rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrGrrrrrrrrrGrrrrr', 'rrrrrGGGGgrrrrrGGGGg', 'rrrrrrrrrGrrrrrrrrrG']
    ppo = PPO(saving_loading=True)
    start_training_step = 500
    training_step = 0
    n_loss = []
    total_reward = []
    n_waiting_time = []
    n_reward = []
    n_mean_reward = []
    n_std_reward = []
    vehicle_num = env.generateTrips_e(max_epLength)
    for ep in range(num_episodes):
        Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
        env.reset(env.sumoCmd)
        reward = []
        s = env.start(Record.record)
        a_before = np.random.randint(1,13)
        a = a_before
        tls_phase = 0
        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[tls_phase])
        new_time_flag = False
        transition_flag = False
        now_step = 0
        time_step = 0
        for t in range(max_epLength):
            if new_time_flag == True:
                new_time_flag = False
                a = ppo.choose_action(s,a_before)
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
            if time_step == a*5:
                r = env.get_RewardBaseRecord(Record.record, a*5)
                s_ = env.get_state()
                phase = np.zeros(4)
                phase[tls_phase] = 1
                ppo.store_transition(s, a, r, s_, phase)
                new_time_flag = True
                if ppo.Buffer.count() > start_training_step:
                    training_step += 1
                    ppo.learn()
                    n_loss.append(ppo.cost)
                reward.append(r)
        env.end()

        if (ep % 400 == 0 or ep == (num_episodes-1)) and ep > 0:
            utils.make_video(env, ppo, max_epLength, 'traffic_ep_' + str(ep) + '.avi')

        ### calculate the waiting time
        waiting_time = Record.calc_waiting()
        n_waiting_time.append(waiting_time)
        logger.debug(' train ' + str(ep) + ' totalreward : ' + str(sum(reward)) +
                     ' ReturnAvg : ' + str(np.mean(reward)) + ' ReturnStd : ' + str(np.std(reward)) + ' waitingtime : ' + str(waiting_time))
        total_reward.append(sum(reward))
        ### calculate the reward
        n_reward.append(sum(reward))
        n_mean_reward.append(np.mean(reward))
        n_std_reward.append(np.std(reward))

        if ppo.saving_or_loading == True and ep > 200 and sum(reward) == max(total_reward):
            ppo.saver.save(sess=ppo.sess, save_path=ppo.save_path)
        ### drawing the loss
        if len(n_loss) > 10 and ep % 50 == 0:
            utils.plot_line(n_loss, name='Loss' + str(ep), path=exp_dir, show=True)
        if ep % 50 == 0 and ep > 0:
            utils.plot_dot(n_waiting_time, name='waiting_time_' + str(ep), path=exp_dir, show=True)

    result = {}
    result['n_reward'] = n_reward
    result['n_mean_reward'] = n_mean_reward
    result['n_std_reward'] = n_std_reward
    result['n_waiting_time'] = n_waiting_time
    result['n_loss'] = n_loss
    utils.plot_result(result, path=exp_dir)
    utils.plot_line(n_loss, name='Loss', path=exp_dir)
    utils.plot_line(n_waiting_time, name='Waiting time', path=exp_dir)
    with open(os.path.join(exp_dir, 'dqn_traffic_data.pkl'), "wb") as f:
        pickle.dump(result, f)
    ### 关闭ubuntu系统
    command = 'shutdown +1'
    os.system(command)

def run_main():
    traffic_train()
if __name__ == "__main__":
    run_main()







