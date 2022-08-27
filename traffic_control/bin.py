

def get_state(self):
    state_ = []
    for iedge in self.entrance:
        vehicleNum = traci.edge.getLastStepVehicleNumber(iedge)
        haltingNum = traci.edge.getLastStepHaltingNumber(iedge)
        # meanSpeed = traci.edge.getLastStepMeanSpeed(iedge)
        # if meanSpeed == 40:
        #     meanSpeed = 0
        # state_.append(haltingNum)
        state_.append(vehicleNum)
        state_.append(haltingNum)
        # state_.append(meanSpeed)
    return state_


def get_reward(self):
    self.cwait_time = self.get_waittime()
    reward = self.lwait_time - self.cwait_time
    reward -= self.get_queue()
    self.lwait_time = self.cwait_time
    return  reward

### 一般地
def step(self, action):
    lightsPhase = self.actionsMap[action]
    for light, index in zip(self.tls, range(len(self.tls))):
        traci.trafficlights.setPhase(light, lightsPhase[index])
    for i in range(self.interval):
        traci.simulationStep()
    state_ = self.get_state()
    reward = self.get_reward()
    return reward, state_
def generateTrips(self, whole_step = 1000):
    whole_step = whole_step * self.interval
    flow = [0.3, 0.5, 0.3, 0.5]
    tripsList = []
    vehicle_num = 0
    for i in range(whole_step):
        for j in range(len(self.entrance)):
            enEdge = self.entrance[j]
            f = flow[j]
            p = random.uniform(0,1)
            if p > f:
                continue
            p = int(random.uniform(0,len(self.exits)))
            des = self.exits[p]
            while des == self.unreachable[enEdge]:
                p = int(random.uniform(0,len(self.exits)))
                des = self.exits[p]
            tripsList.append([i, enEdge, des])
            vehicle_num += 1
    trip_path = data_path + 'map1.trips.xml'
    ag.writeTripsXml(tripsList, trip_path)
    net_path = data_path + 'map1.net.xml'
    rou_path = data_path + 'map1.rou.xml'
    os.system('duarouter -n ' + net_path +  ' -t ' + trip_path + ' -o ' + rou_path)
    return vehicle_num
