import traci
import traci.constants as tc
import time
import datetime
import numpy as np
import os
import sys
import cv2
from utils import lane_id
# Environment Model
# sumoBinary = "/usr/local/bin/sumo"
sys.path.append(os.path.join('E:/sumo-0.1.0/', 'tools'))
sys.path.append('E:/sumo-0.1.0/')

sumoBinary = "sumo-gui"
traffic_path = "../data/map1.sumocfg"
sumoCmd = [sumoBinary, "-c", traffic_path, "--start"]  # The path to the sumo.cfg file
traci.start(sumoCmd)

for i in range(1000):

    traci.simulationStep()


# tls = traci.trafficlight.getIDList()

# A = traci.edge.getIDCount()
# D = traci.edge.getIDList()
# EID = [v for v in D if 'edge'in v]
# for iedge in EID:
#     num = traci.edge.getLastStepVehicleNumber(iedge)

        # ps = (traci.vehicle.getSubscriptionResults(veh_id))

        # if (340 <= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 690)
        #
        #     for veh_id in traci.edge.getLastStepVehicleIDs(v):
        #         traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED))

    # p_state = np.zeros((60, 60, 2))
    # for x in p:
    #     ps = p[x][tc.VAR_POSITION]
    #     spd = p[x][tc.VAR_SPEED]
    #     p_state[int(ps[0] / 5), int(ps[1] / 5)] = [1, int(round(spd))]
    #         v_state[int(ps[0]/5), int(ps[1]/5)] = spd


# n_edge = ['edge-1-0', 'edge-2-0', 'edge-3-0', 'edge-4-0']
# for i in range(200):
#     A = traci.edge.getLastStepVehicleIDs('edge-1-0')
#     traci.simulationStep()
#     limit = [[340, 500], [530, 690]]
#     state_1 = np.zeros((48, 48))
#     state_2 = np.zeros((48, 48))
#     D = traci.trafficlight.getPhase("0")
#     E = traci.trafficlight.getRedYellowGreenState("0")
#     for j, v in enumerate(n_edge):
#         for veh_id in traci.edge.getLastStepVehicleIDs(v):
#             traci.vehicle.subscribe(veh_id, (tc.VAR_POSITION, tc.VAR_SPEED, tc.VAR_LANE_ID, tc.VAR_LANE_INDEX))
#             ps = traci.vehicle.getSubscriptionResults(veh_id)
#             if (340<= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 500) or (530<= ps[tc.VAR_POSITION][0] and ps[tc.VAR_POSITION][0] <= 690) or \
#                 (530 <= ps[tc.VAR_POSITION][1] and ps[tc.VAR_POSITION][1] <= 690) or (
#                         340 <= ps[tc.VAR_POSITION][1] and ps[tc.VAR_POSITION][1] <= 500):
#                 if j == 0:
#                     x = int((ps[tc.VAR_POSITION][0] - 340) / 8)
#                     y = 20 + lane_id[j][ps[tc.VAR_LANE_ID]]
#                     state_1[x, y] = 1
#                     state_2[x, y] = ps[tc.VAR_SPEED] / 19.444
#                 elif j == 1:
#                     x = 28 + int((ps[tc.VAR_POSITION][0] - 530) / 8)
#                     y = 20 + lane_id[j][ps[tc.VAR_LANE_ID]]
#                     state_1[x, y] = 1
#                     state_2[x, y] = ps[tc.VAR_SPEED] / 19.444
#                 elif j == 2:
#                     x = 20 + lane_id[j][ps[tc.VAR_LANE_ID]]
#                     y = int((ps[tc.VAR_POSITION][1] - 340) / 8)
#                     state_1[x, y] = 1
#                     state_2[x, y] = ps[tc.VAR_SPEED] / 19.444
#                 elif j == 3:
#                     x = 20 + lane_id[j][ps[tc.VAR_LANE_ID]]
#                     y = 28 + int((ps[tc.VAR_POSITION][1] - 530) / 8)
#                     state_1[x, y] = 1
#                     state_2[x, y] = ps[tc.VAR_SPEED] / 19.444



### 制作视频
# video_path = 'Video/'
# traci.gui.setSchema(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, schemeName='real world')
# a = traci.gui.getZoom(viewID=traci._gui.GuiDomain.DEFAULT_VIEW)
# traci.gui.setZoom(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, zoom=4*a)
# fps = 8
# traci.gui.screenshot(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, filename=video_path + str(0) +".jpg")
# traci.simulationStep()
# img = cv2.imread(video_path + str(0) +".jpg")
# os.remove(video_path + str(0) +".jpg")
# size = img.shape
# size =  tuple(reversed(size[:2]))
# videowriter = cv2.VideoWriter(video_path+"test.avi",cv2.VideoWriter_fourcc('M', 'P', '4', '2'),fps,size, isColor=True)
# for i in range(100):
#     S = traci.trafficlights.getRedYellowGreenState(tls[0])
#     traci.gui.screenshot(viewID=traci._gui.GuiDomain.DEFAULT_VIEW, filename=video_path + str(i) +".jpg")
#     traci.simulationStep()
#     img = cv2.imread(video_path + str(i) +".jpg")
#     os.remove(video_path + str(i) + ".jpg")
#     videowriter.write(img)
    # traci.gui.getZoom(viewID=traci._gui.GuiDomain.DEFAULT_VIEW)
# videowriter.release()
# command = 'taskkill /IM sumo-gui.exe /F'
# os.system(command)
# # traci.close()
# traci.start(sumoCmd)