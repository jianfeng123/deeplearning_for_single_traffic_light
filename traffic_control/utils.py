import traci
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import cv2
sns.set_style('whitegrid')



lane_id = [{'edge-1-0_0': 0, 'edge-1-0_1': 1, 'edge-1-0_2': 2, 'edge-1-0_3': 3,  'edge-0-1_3': 4, 'edge-0-1_2': 5, 'edge-0-1_1': 6, 'edge-0-1_0': 7},
           {'edge-0-2_0': 0, 'edge-0-2_1': 1, 'edge-0-2_2': 2, 'edge-0-2_3': 3,  'edge-2-0_3': 4, 'edge-2-0_2': 5, 'edge-2-0_1': 6, 'edge-2-0_0': 7},
           {'edge-0-3_0': 0, 'edge-0-3_1': 1, 'edge-0-3_2': 2, 'edge-0-3_3': 3, 'edge-3-0_3': 4, 'edge-3-0_2': 5, 'edge-3-0_1': 6, 'edge-3-0_0': 7},
           {'edge-4-0_0': 0, 'edge-4-0_1': 1, 'edge-4-0_2': 2, 'edge-4-0_3': 3, 'edge-0-4_3': 4, 'edge-0-4_2': 5, 'edge-0-4_1': 6, 'edge-0-4_0': 7}]

def list_of_n_phases(TLIds):
    n_phases = []
    for light in TLIds:
        n_phases.append(int((len(traci.trafficlights.getRedYellowGreenState(light)) ** 0.5) * 2))
    return n_phases
def makemap(TLIds):
    maptlactions = []
    n_phases = list_of_n_phases(TLIds)
    for n_phase in n_phases:
        mapTemp = []
        if len(maptlactions) == 0:
            for i in range(n_phase):
                if i%2 == 0:
                    maptlactions.append([i])
        else:
            for state in maptlactions:
                for i in range(n_phase):
                    if i%2 == 0:
                        mapTemp.append(state+[i])
            maptlactions = mapTemp
    return maptlactions

## 绘制曲线
def plot_line(main_data=None, x_label='Training steps', y_label= None, path=None, name='Loss', color='red', show=False):
    plt.figure(figsize=(10,10))
    f, ax = plt.subplots(1, 1)
    ax.plot(range(len(main_data)), main_data, color=color)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(name)
    f.savefig(os.path.join(path, str(name) + '.png'), dpi=200)
    if show:
        plt.show()
## 绘制曲线
def plot_dot(main_data=None, x_label='Training steps', y_label=None, path=None, name='Reward', show=False,
             color='black'):
    plt.figure(figsize=(10, 10))
    f, ax = plt.subplots(1, 1)
    # ax.plot(range(len(main_data)), main_data, color=color)
    ax.scatter(range(len(main_data)), main_data, color=color, s=5)
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(name)
    f.savefig(os.path.join(path, str(name) + '.png'), dpi=200)
    if show:
        plt.show()
## 绘制比较曲线
def plot_compare_dot(wait_data=None, regular_data=None, path=None, name='Reward', show=False):
    plt.figure(figsize=(10,10))
    f, ax = plt.subplots(1, 1)
    ax.scatter(range(len(wait_data)), wait_data, color='b', label='DQN', marker='*', s=40)
    ax.scatter(range(len(regular_data)), regular_data, color='r', label='Fixed', marker='^', s=40)
    legend = ax.legend(frameon=True, loc='upper left')
    frame = legend.get_frame()                  ### 设置上面的小框框
    frame.set_edgecolor('black')
    frame.set_facecolor('lightcyan')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Waiting time')
    f.savefig(os.path.join(path, str(name) + '.png'), dpi=200)
    if show:
        plt.show()


## 绘制曲线
def plot_result(result=None, path=None, show=False, color='g'):
    ### 绘制reward 曲线
    plt.figure(figsize=(10,10))
    f, ax = plt.subplots(1, 1)
    ax.plot(range(len(result['n_reward'])), result['n_reward'], color=color)
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Return')
    f.savefig(os.path.join(path, 'reward.png'), dpi=200)
    if show:
        plt.show()
    ### 绘制平均奖励变化，增加了fill_between
    color = cm.viridis(0.5)
    plt.figure(figsize=(10, 10))
    f, ax = plt.subplots(1,1)
    ax.plot(range(len(result['n_mean_reward'])), result['n_mean_reward'], color=color)
    r1 = list(map(lambda x: x[0]-x[1], zip(result['n_mean_reward'], result['n_std_reward'])))
    r2 = list(map(lambda x: x[0]+x[1], zip(result['n_mean_reward'], result['n_std_reward'])))
    ax.fill_between(range(len(result['n_mean_reward'])), r1, r2, color=color, alpha=0.3)
    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Return')
    f.savefig(os.path.join(path, 'avgreward' + '.png'), dpi=200)
    if show:
        plt.show()

## 将场景制作为视频
def make_video(env, model=None, max_epLength = 3000, vedio_name='test.avi'):
    video_path = '../Video/'
    env.reset(env.sumoCmd_gui)
    ### 设置主题
    traci.gui.setSchema(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, schemeName='real world')
    ### 将镜头放大两倍
    a = traci.gui.getZoom(viewID=traci._gui.GuiDomain.DEFAULT_VIEW)
    traci.gui.setZoom(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, zoom=4 * a)
    fps = 8
    traci.gui.screenshot(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, filename=video_path + str(0) + ".jpg")
    s = env.start()
    traci.simulationStep()
    img = cv2.imread(video_path + str(0) + ".jpg")
    size = img.shape
    size = tuple(reversed(size[:2]))
    videowriter = cv2.VideoWriter(video_path + vedio_name, cv2.VideoWriter_fourcc('M', 'P', '4', '2'), fps, size,
                                  isColor=True)
    os.remove(video_path + str(0) + ".jpg")
    rgy_state = ['GGGGgrrrrrGGGGgrrrrr', 'rrrrrGGGGgrrrrrGGGGg']
    a_before = np.random.randint(0, 2)
    a = a_before
    traci.trafficlight.setRedYellowGreenState(env.tls[0], rgy_state[a_before])
    new_time_flag = False
    transition_flag = False
    now_step = 0
    time_step = 0
    for t in range(max_epLength):
        if new_time_flag == True:
            new_time_flag = False
            a = model.choose_action(s, a, training=False)
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
        traci.gui.screenshot(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, filename=video_path + str(t) + ".jpg")
        traci.simulationStep()
        img = cv2.imread(video_path + str(t) + ".jpg")
        os.remove(video_path + str(t) + ".jpg")
        videowriter.write(img)
        if time_step == 10:
            s = env.get_state()
            new_time_flag = True
    # env.end()
    videowriter.release()
    ### for windows
    # command = 'taskkill /IM sumo-gui.exe /F'
    # os.system(command)
    # ### for ubuntu
    command = 'skill /IM sumo-gui /F'
    os.system(command)
    # traci.close()

#### 记录道路状态
class Record():
    def __init__(self, step_num = 2000, vehicle_num = 2000):
        self.step_num = step_num
        self.vehicle_num = vehicle_num
        self.record = {}
        self.record['vehicle'] = {}
        ## get vehicle record
        self.record['vehicle']['waittime'] = np.zeros((self.vehicle_num, step_num))
        self.record['vehicle']['waittime1'] = self.getVehicleRecord()
        self.record['vehicle']['arrived_id'] = self.getVehicleRecord()
        self.record['vehicle']['number'] = np.zeros((self.step_num))
        self.record['step'] = -1
        self.step = 0
        self.arrived_vehicle = 0
        det = ["laneOC_1", "laneOC_2", "laneOC_3", "laneOC_4"]
        self.dets = []
        det_num = [1,2,3,4]
        for v in det_num:
            for s in det:
                self.dets.append(s.replace("C", str(v)))
    def getVehicleRecord(self):
        waittime = np.zeros((self.vehicle_num))
        return  waittime
    def addVehicleRecord(self,record, step = 0):
        for v in self.dets:
            for v_id in traci.lanearea.getLastStepVehicleIDs(v):
                wait = traci.vehicle.getAccumulatedWaitingTime(v_id)
                record['vehicle']['waittime1'][int(v_id)] = max(record['vehicle']['waittime1'][int(v_id)], wait)
        count = traci.vehicle.getIDCount()
        record['vehicle']['number'][step] = count
        # if step > 0:
        #     record['vehicle']['waittime'][:, step] = record['vehicle']['waittime'][:, step - 1]
        # for id in traci.vehicle.getIDList():
        #     wait = traci.vehicle.getAccumulatedWaitingTime(id)
        #     if int(id) < self.vehicle_num:
        #         record['vehicle']['waittime'][int(id), step] = max(record['vehicle']['waittime'][int(id), step], wait)
    def calc_waiting(self):
        m_w = np.sum(self.record['vehicle']['waittime1'][np.where(self.record['vehicle']['arrived_id'][:]==1)]) / self.arrived_vehicle
        return m_w
    def addRecord(self, step):
        self.record['step'] = step
        self.addVehicleRecord(self.record, step)
    def get_stop_times(self):
        count = 0
        for i in range(self.vehicle_num):
            if self.record['vehicle']['waittime'][i, -1] != 0:
                wait_set = list(set(self.record['vehicle']['waittime'][i, :]))
                wait_list = list(self.record['vehicle']['waittime'][i, :])
                for j in wait_set:
                    if j > 0:
                        num = wait_list.count(j)
                        if num > 1:
                            count += 1
        return count
    def simulation_step(self):
        self.addRecord(self.step)
        self.step += 1
        traci.simulationStep()
        v_ids = traci.simulation.getArrivedIDList()
        # traci.lanearea.getLastStepVehicleIDs(self.dets[i])
        if len(v_ids) > 0:
            v_ids = np.array(list(map(int, v_ids)))
            self.record['vehicle']['arrived_id'][v_ids] = 1
        self.arrived_vehicle += len(v_ids)

