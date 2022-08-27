from collections import deque
import random
class replay_buffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()
        self.mean_reward = 0.0
    def get_Batch(self, batch_size):
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)
    def add(self, state, action, reward, state_):
        experience = (state, action, reward, state_)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        self.num_experiences += 1
    def add_ppo(self, state, action, reward, state_, phase):
        experience = (state, action, reward, state_, phase)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
        else:
            self.buffer.popleft()
            self.buffer.append(experience)
        self.num_experiences += 1

    def count(self):
        return self.num_experiences
    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0
    def remove(self):
        self.buffer.popleft()
        self.num_experiences = self.num_experiences - 1




