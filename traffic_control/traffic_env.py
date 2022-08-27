import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plt
import scipy.misc
import os
import scipy.stats as ss
import math
from DRLmodel.DQN import DeepQNetWork as DQN
from DRLmodel.PPO import PPO
#matplotlib inline
import os, sys
import datetime
import math
## SUMO控制协议
import traci
import utils as util
from utils import lane_id
import traci.constants as tc
import arrivalGen as ag
# if 'SUMO_HOME' in os.environ:
# The path of SUMO-tools to get the traci library

## 主要是导入一些常用的shell命令
## for linux
os.environ['SUMO_HOME'] = "/home/g/deepLap/sumo-1.1.0"
sys.path.append("/home/g/deepLap/sumo-1.1.0/tools")
## for windows
# os.environ['SUMO_HOME'] = 'E:/sumo-0.1.0'
# sys.path.append(os.path.join('E:/sumo-0.1.0/', 'tools'))
# sys.path.append(os.path.join('E:/sumo-0.1.0/', 'data'))
# sys.path.append('E:/sumo-0.1.0/')

# Environment Model
# sumoBinary = "/usr/local/bin/sumo"
class traffic_env():
    def __init__(self, vehicle_num=0, data_path='../../Env/data/'):
        sumoBinary = "sumo"  ## 控制命令，可以在sumo和sumo-gui之间切换
        self.data_path = data_path
        road_path = data_path + "map1.sumocfg"  # the road path
        sumoCmd = [sumoBinary, "-c", road_path, "--start"]  # The path to the sumo.cfg file
        self.sumoCmd = sumoCmd
        self.sumoCmd_gui = ["sumo-gui", "-c", road_path, "--start"]
        traci.start(["sumo", "-c", road_path, "--start"])
        self.tls = traci.trafficlights.getIDList()
        ID_edge = traci.edge.getIDList()
        self.ID_edge = [v for v in ID_edge if 'edge' in v]
        self.entrance = []
        self.exits = []
        self.unreachable = {}
        for i, iedge in enumerate(self.ID_edge):
            if i <=3 :
                self.exits.append(iedge)
            else:
                self.entrance.append(iedge)
        for i in range(len(self.entrance)):
            self.unreachable[self.entrance[i]] = self.exits[i]
        self.end()
        self.n_act = 2
        ### 出行路口；每个路口的第一个为直行，第二个为左转
        self.n_exit = [['edge-0-2', 'edge-0-4'], ['edge-0-1', 'edge-0-3'], ['edge-0-4', 'edge-0-1'], ['edge-0-3', 'edge-0-2']]
        ### 归一化max变量
        self.max_wait = 1
        self.max_wait_changed  = 1
        self.min_wait_changed = -1
        self.max_len_changed = 1
        self.min_len_changed = -1
        self.max_len = 1
        det = ["laneIC_1", "laneIC_2", "laneIC_3", "laneIC_4"]
        self.dets = []
        det_num = [1,2,3,4]
        for v in det_num:
            for s in det:
                self.dets.append(s.replace("C", str(v)))
        self.vehicles_id = np.zeros((vehicle_num, 2), dtype=int)
        self.lane_vnum = np.zeros((16), dtype=int)
        self.halting_num = np.zeros((2), dtype=int)
        self.vehicle_num = vehicle_num
    def reset(self, cmd):
        try:
            traci.start(cmd)
        except:
            try:
                traci.start(cmd)
            except:
                traci.start(cmd)
    def start(self, record=None):
        if record != None:
            pass
            # self.lwait_time = self.get_waittime()
            wait = record['vehicle']['waittime'][:, (record['step'])]
            wait = sum(wait)
            self.last_wait = wait
            # self.last_qlen = self.get_queue()
    def end(self):
        traci.close(wait=False)
    ### 八个路径的概率都表示出来
    def generateTrips_e(self, whole_step = 1000, p=1):
        whole_step = whole_step
        # flow = np.array(list(reversed([1/5, 1/20, 1/5, 1/20, 1/10, 1/20, 1/10, 1/20]))) * p
        # flow = np.array([2 / 30, 2 / 30, 2 / 30, 2 / 30, 6 / 30, 5 / 30, 6 / 30, 5 / 30]) * p
        flow = np.array([5 / 30, 4 / 30, 5 / 30, 4 / 30, 3 / 30, 3 / 30, 3 / 30, 3 / 30]) * p
        # flow = np.array([6 / 30, 4 / 30, 6 / 30, 4 / 30, 2 / 30, 3 / 30, 2 / 30, 3 / 30]) * p
        # flow = np.array([2 / 30, 2 / 30, 2 / 30, 2 / 30, 2 / 30, 1 / 30, 2 / 30, 1 / 30]) * p  ## 原始
        # flow = np.array([5 / 30, 4 / 30, 5 / 30, 4 / 30, 3 / 30, 3 / 30, 3 / 30, 3 / 30]) * p
        # flow = np.array([1 / 5, 1 / 20, 1 / 5, 1 / 20, 1 / 10, 1 / 20, 1 / 10, 1 / 20]) * p
        tripsList = []
        vehicle_num = 0
        for i in range(whole_step):
            for j in range(len(self.entrance)):
                for k in range(2):
                    inEdge = self.entrance[j]
                    f = flow[j*2 + k]
                    p = random.uniform(0,1)
                    if p > f:
                        continue
                    des = self.n_exit[j][k]
                    tripsList.append([i, inEdge, des])
                    vehicle_num += 1
        trip_path = self.data_path + 'map1.trips.xml'
        ag.writeTripsXml(tripsList, trip_path)
        net_path = self.data_path + 'map1.net.xml'
        rou_path = self.data_path + 'map1.rou.xml'
        os.system('duarouter -n ' + net_path +  ' -t ' + trip_path + ' -o ' + rou_path)
        return vehicle_num
    ### 获得矩阵表示的状态
    def get_state(self, DRQN=True, reward=False):
        if reward == True:
            self.vehicles_id[:, 0] = self.vehicles_id[:, 1]
            self.vehicles_id[:, 1] = 0
        if DRQN:
            state_ = np.zeros((16, 20))
            for i in range(16):
                vehicles_id = traci.lanearea.getLastStepVehicleIDs(self.dets[i])
                if reward == True:
                    if len(vehicles_id) > 0:
                        v_ids = np.array(list(map(int, vehicles_id)))
                        self.vehicles_id[v_ids, 1] = 1
                self.lane_vnum[i] = len(vehicles_id)
                for veh_id in vehicles_id:
                    # traci.vehicle.subscribe(veh_id, tc.VAR_POSITION)
                    # ps = traci.vehicle.getSubscriptionResults(veh_id)
                    j = int(i / 4)
                    # if (338 <= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 498) or (
                    #         532 <= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 692) or \
                    #         (338 <= ps[tc.VAR_POSITION][1] and ps[tc.VAR_POSITION][1] <= 498) or (
                    #         532 <= ps[tc.VAR_POSITION][1] and ps[tc.VAR_POSITION][1] <= 692):
                    ps = traci.vehicle.getPosition(veh_id)
                    # traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_LANE_INDEX))
                    # ps = traci.vehicle.getSubscriptionResults(veh_id)
                    if (338 <= ps[0] and ps[0] <= 498) or (532 <= ps[0] and ps[0] <= 692) or \
                            (338 <= ps[1] and ps[1] <= 498) or (532 <= ps[1] and ps[1] <= 692):
                        if j == 0:
                            # y = 27 - (i % 4)
                            x = i
                            y = np.clip(int((ps[0] - 338 - 4) / 8), 0, 19)
                            state_[x, y] = 1
                            # state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                        elif j == 1:
                            x = i
                            y = np.clip(19 - int((ps[0] - 532 + 4) / 8), 0, 19)
                            # state_[x, y] = 1
                            state_[x, y] = 1
                            # state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                        elif j == 2:
                            x = i
                            y = np.clip(int((ps[1] - 338 - 4) / 8), 0, 19)
                            state_[x, y] = 1
                            # state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                        elif j == 3:
                            x = i
                            y = np.clip(19 - int((ps[1] - 532 + 4) / 8), 0, 19)
                            state_[x, y] = 1
            return state_
        else:
            state_ = np.zeros((16, 20, 2))
            for i in range(16):
                vehicles_id = traci.lanearea.getLastStepVehicleIDs(self.dets[i])
                self.lane_vnum[i] = len(vehicles_id)
                for veh_id in vehicles_id:
                    traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))
                    ps = traci.vehicle.getSubscriptionResults(veh_id)
                    j = int(i / 4)
                    if (338 <= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 498) or (
                            532 <= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 692) or \
                            (338 <= ps[tc.VAR_POSITION][1] and ps[tc.VAR_POSITION][1] <= 498) or (
                            532 <= ps[tc.VAR_POSITION][1] and ps[tc.VAR_POSITION][1] <= 692):
                    # ps = traci.vehicle.getPosition(veh_id)
                    # traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_LANE_INDEX))
                    # ps = traci.vehicle.getSubscriptionResults(veh_id)
                    # if (338 <= ps[0] and ps[0] <= 498) or (532 <= ps[0] and ps[0] <= 692) or \
                    #         (338 <= ps[1] and ps[1] <= 498) or (532 <= ps[1] and ps[1] <= 692):
                        if j == 0:
                            # y = 27 - (i % 4)
                            x = i
                            y = np.clip(int((ps[tc.VAR_POSITION][0] - 338 - 4) / 8), 0, 19)
                            state_[x, y, 0] = 1
                            state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                            # state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                        elif j == 1:
                            x = i
                            y = np.clip(19 - int((ps[tc.VAR_POSITION][0] - 532 + 4) / 8), 0, 19)
                            # state_[x, y] = 1
                            state_[x, y, 0] = 1
                            state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                            # state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                        elif j == 2:
                            x = i
                            y = np.clip(int((ps[tc.VAR_POSITION][1] - 338 - 4) / 8), 0, 19)
                            state_[x, y, 0] = 1
                            state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                            # state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
                        elif j == 3:
                            x = i
                            y = np.clip(19 - int((ps[tc.VAR_POSITION][1] - 532 + 4) / 8), 0, 19)
                            state_[x, y, 0] = 1
                            state_[x, y, 1] = ps[tc.VAR_SPEED] / 19.444
            return state_
    def get_state_q(self, training = True):
        state_ = np.zeros((48))
        self.vehicles_id[:, 0] = self.vehicles_id[:, 1]
        self.vehicles_id[:, 1] = 0
        self.halting_num[0] = self.halting_num[1]
        self.halting_num[1] = 0
        for i in range(16):
            num = traci.lanearea.getLastStepVehicleNumber(self.dets[i])
            num1 = traci.lanearea.getLastStepHaltingNumber(self.dets[i])
            num2 = traci.lanearea.getJamLengthVehicle(self.dets[i])
            state_[i] = num
            state_[i * 3 + 1] = num1
            state_[i * 3 + 2] = num2
            self.lane_vnum[i] = num
            # self.halting_num[1] +=traci.lanearea.getLastStepHaltingNumber(self.dets[i])
            if training:
                v_ids = traci.lanearea.getLastStepVehicleIDs(self.dets[i])
                if len(v_ids) > 0:
                    v_ids = np.array(list(map(int, v_ids)))
                    self.vehicles_id[v_ids, 1] = 1
        return state_
    ### the reward must contain positive value
    def get_RewardBaseRecord(self, record, delay=10):
        wait = record['vehicle']['waittime'][:, (record['step'])] \
                   - record['vehicle']['waittime'][:, (record['step'] - delay)]
        wait = sum(wait)
        wait_c = self.last_wait - wait
        self.last_wait = wait
        reward = wait_c
        return reward
    def get_Reward_queue(self, Principal, a):
        q_index = [[0, 1, 2, 4, 5, 6], [8, 9, 10, 12, 13, 14], [3, 7], [11, 15]]
        roadv_w = np.sum(self.lane_vnum[q_index[0]])
        roadv_n = np.sum(self.lane_vnum[q_index[1]])
        roadv_wl = np.sum(self.lane_vnum[q_index[2]])
        roadv_nl = np.sum(self.lane_vnum[q_index[3]])
        q_output = sum((self.vehicles_id[:, 0] - self.vehicles_id[:, 1]) > 0)  ## 这段期间过停止线的数量
        if Principal:
            if a == 0:
                reward = q_output - roadv_n - 0.2 * (roadv_wl + roadv_nl)
            else:
                reward = q_output - roadv_w - 0.2 * (roadv_wl + roadv_nl)
        else:
            if a == 0:
                reward = q_output - roadv_nl - 0.25 * (roadv_w + roadv_n)
            else:
                reward = q_output - roadv_wl - 0.25 * (roadv_w + roadv_n)
        # reward = self.halting_num[0] - self.halting_num[1]
        # reward = np.linalg.norm(s[q_index[a]]) - np.linalg.norm(s_[q_index[a]])
        return reward
    def get_Reward_queue_c(self, phase):
        q_index = [[0, 1, 2, 4, 5, 6], [8, 9, 10, 12, 13, 14], [3, 7], [11, 15]]
        roadv_w = np.sum(self.lane_vnum[q_index[0]])
        roadv_n = np.sum(self.lane_vnum[q_index[1]])
        roadv_wl = np.sum(self.lane_vnum[q_index[2]])
        roadv_nl = np.sum(self.lane_vnum[q_index[3]])
        q_output = sum((self.vehicles_id[:, 0] - self.vehicles_id[:, 1]) > 0)  ## 这段期间过停止线的数量
        if phase == 0:
            reward = q_output - roadv_n - 0.2 * (roadv_wl + roadv_nl)
        elif phase == 1:
            reward = q_output - roadv_nl - 0.25 * (roadv_w + roadv_n)
        elif phase == 2:
            reward = q_output - roadv_w - 0.2 * (roadv_wl + roadv_nl)
        else:
            reward = q_output - roadv_wl - 0.25 * (roadv_w + roadv_n)
        # reward = self.halting_num[0] - self.halting_num[1]
        # reward = np.linalg.norm(s[q_index[a]]) - np.linalg.norm(s_[q_index[a]])
        return reward

    def get_waittime(self):
        wait_time_map = {}
        for veh_id in traci.vehicle.getIDList():
            wait_time_map[veh_id] = traci.vehicle.getAccumulatedWaitingTime(veh_id)
        wait_temp = dict(wait_time_map)
        wait_sum_time = sum(wait_temp[x] for x in wait_temp)
        return wait_sum_time
    def get_queue(self):
        queue_len = []
        for iedge in self.entrance:
            length = traci.edge.getLastStepVehicleNumber(iedge)
            queue_len.append(length)
        qlen = sum(queue_len)
        if qlen > self.max_len:
            self.max_len = qlen
        return qlen
    def transition(self, step, action):
        if action == 0:
            if step == 0:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrryyyygrrrrryyyyg')
            if step == 6:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrrrrrrGrrrrrrrrrG')
            if step == 26:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrrrrrryrrrrrrrrry')
        if action == 1:
            if step == 0:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'yyyygrrrrryyyygrrrrr')
            if step == 6:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrGrrrrrrrrrGrrrrr')
            if step == 26:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrryrrrrrrrrryrrrrr')
        step += 1
        return step
    def transitionF(self, step, a=0):
        if step == 0:
            if a == 0:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrryyyygrrrrryyyyg')
            elif a == 1:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrrrrrryrrrrrrrrry')
            elif a == 2:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'yyyygrrrrryyyygrrrrr')
            else:
                traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrryrrrrrrrrryrrrrr')
        step += 1
        return step
    def transitionD(self, step, a1=0, a2=0, Principal = None):
        if step == 0:
            if Principal:
                if a1 == 0:
                    traci.trafficlight.setRedYellowGreenState(self.tls[0], 'yyyygrrrrryyyygrrrrr')
                else:
                    traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrryyyygrrrrryyyyg')
            else:
                if a2 == 0:
                    traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrryrrrrrrrrryrrrrr')
                else:
                    traci.trafficlight.setRedYellowGreenState(self.tls[0], 'rrrrrrrrryrrrrrrrrry')
        step += 1
        return step










