import numpy as np
from sensor import *
from utility import *

class Robot():
    def __init__(self, lidar_max_range):
        self.x = 0
        self.y = 0
        self.v = 0
        self.yaw = 0
        self.lidar = lidar_sensor(max_range = lidar_max_range)
        self.gps = gps_sensor()
        self.measurements = []

    def init_state(self, x, y, yaw, v):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v

    def update(self, dt, acc, theta_dot):

        self.v += acc*dt
        self.yaw = wrap_ang(self.yaw + theta_dot*dt)
        self.x += self.v*np.cos(self.yaw)*dt
        self.y += self.v*np.sin(self.yaw)*dt

    def measure(self, poles):

        self.measurements = self.lidar.measure(poles, self.x, self.y, self.yaw)

    def get_state(self):
        s = np.array([self.x, self.y, self.yaw, self.v])
        s.shape = (4, 1)
        return s