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

def traffic_train(compare_with_regular = False):
    logger.setup(exp_dir, os.path.join(exp_dir, 'dqns_log.txt'), 'debug')
    num_episodes = 500
    max_epLength = 1800
    vehicle_num = 1795
    env = traffic_env(vehicle_num)
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrGrrrrrrrrrGrrrrr', 'rrrrrGGGGgrrrrrGGGGg', 'rrrrrrrrrGrrrrrrrrrG']
    dqn = DQN(2, saving_loading=False)
    start_training_step = 50
    training_step = 0
    n_loss = []
    total_reward = []
    n_waiting_time = []
    n_arrived = []
    n_reward = []
    n_mean_reward = []
    n_std_reward = []
    per_waiting_time = []
    duration = 2
    for ep in range(num_episodes):
        # vehicle_num = env.generateTrips_e(max_epLength)
        Record = utils.Record(step_num=max_epLength, vehicle_num=vehicle_num)
        env.reset(env.sumoCmd)
        reward = []
        s = env.start(Record.record)
        s = env.get_state_q()
        a_before = np.random.randint(0,2)
        a = a_before
        phase = 0
        traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[phase])
        new_time_flag = False
        transition_flag = False
        now_step = 0
        time_step = 0
        sp_ = [s, phase]
        sp = sp_
        tqdm_e = tqdm(range(max_epLength), desc='duration', leave=True, unit=" episodes")
        for t in tqdm_e:
            if new_time_flag == True:
                new_time_flag = False
                a = dqn.choose_action(sp)
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
                s_ = env.get_state_q()
                r = env.get_Reward_queue()
                sp_ = [s_, phase]
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
        arrived = Record.arrived_vehicle
        n_arrived.append(arrived)
        logger.debug(' train ' + str(ep) + ' meantime : ' + str(waiting_time) + ' arrived : ' + str(arrived))
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
    # command = 'shutdown +1'
    # os.system(command)

def run_main():
    traffic_train()
if __name__ == "__main__":
    run_main()







