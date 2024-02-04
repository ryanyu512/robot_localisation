import numpy as np
from utility import *

class gps_sensor():
    def __init__(self):
        self.x = None
        self.y = None
        self.meas_rate = 1
        self.pos_std = 3

class lidar_sensor():
    def __init__(self, max_range, range_std = 0.03, theta_std = np.deg2rad(1)):
        self.measurments = []
        self.max_range = max_range
        self.range_std = range_std
        self.theta_std  = theta_std
        self.meas_rate = 10

    def measure(self, poles, x, y, theta):

        self.measurments = []

        for p in poles:

            dx = p[0] - x 
            dy = p[1] - y 
            true_dist  = np.sqrt(dx**2 + dy**2)
            meas_dist = np.random.normal(true_dist, self.range_std)

            if true_dist <= self.max_range:

                true_ang = np.arctan2(dy, dx)
                meas_ang  = np.random.normal(true_ang, self.theta_std)

                true_ang -= theta
                true_ang = wrap_ang(true_ang)

                meas_ang -= theta
                meas_ang = wrap_ang(meas_ang)

                self.measurments.append([meas_dist, meas_ang])

        return self.measurments