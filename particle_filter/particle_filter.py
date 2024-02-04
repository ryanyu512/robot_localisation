import numpy as np
import random as r

from robot import *
from sensor import *


def resample_particles(particles, fv, dt, max_scale = 15., min_scale = 0.5):
    
    w = np.array([p.weight + 1e-20 for p in particles])
    w_sum = sum(w)
    p = w/w_sum
    p.shape = (len(particles), )

    scale = len(particles)/(w_sum*2 + 1e-20)

    resample_ind = np.random.choice(np.arange(0, len(particles)), 
                                    p = p,
                                    replace = True, 
                                    size = len(particles))
  
    speed_std = particles[0].speed_std
    yaw_dot_std = particles[0].yaw_dot_std

    resampled_particles = [None]*len(particles)

    scale = max_scale if scale > max_scale else scale
    scale = min_scale if scale < min_scale else scale

    for i, r_ind in enumerate(resample_ind):
        p = particles[r_ind]

        v = np.random.normal(fv, p.speed_std*scale)
        yaw = wrap_ang(p.yaw + np.random.normal(0., p.yaw_dot_std*scale)*dt)

        x = p.x + v*np.cos(yaw)*dt
        y = p.y + v*np.sin(yaw)*dt
        
        resampled_particles[i] = Particle(x, y, yaw, v, p.speed_std, p.yaw_dot_std, p.lidar.max_range)
        
    return resampled_particles

class Particle(Robot):
    def __init__(self, x, y, yaw, v, speed_std, yaw_dot_std, lidar_max_range):
        Robot.__init__(self, lidar_max_range)
        Robot.init_state(self, x, y, yaw, v)

        self.speed_std = speed_std
        self.yaw_dot_std = yaw_dot_std

        self.dist_std = 1
        self.ang_std = np.deg2rad(30)

        self.dis_weight_scale = 1. / (np.math.sqrt(2 * np.math.pi) * self.dist_std)
        self.ang_weight_scale = 1. / (np.math.sqrt(2 * np.math.pi) * self.ang_std)

        self.weight = 0.0
        self.dis_weight = 1.0
        self.ang_weight = 1.0

    def predict(self, fv, meas_yaw_dot, gyro_std, dt):

        self.v   = np.random.normal(fv, self.speed_std)
        yaw_dot_ = np.random.normal(meas_yaw_dot, gyro_std)

        self.yaw = wrap_ang(self.yaw + yaw_dot_*dt)
        self.x += self.v*np.cos(self.yaw)*dt
        self.y += self.v*np.sin(self.yaw)*dt

    def compute_weights(self, mu, std, x):
        p = 1/np.sqrt(2*np.math.pi*std**2)*np.exp(-(x - mu)**2/(2*std**2))
        return p
    
    def update_weight(self, robot_measurements):

        self.weight = 0.

        for i, p_meas in enumerate(self.measurements):
            best_weight = 0.
            for j, r_meas in enumerate(robot_measurements):
                x_dist = p_meas[0] - r_meas[0]

                w_d = self.compute_weights(0, self.dist_std, x_dist)
                w_d /= self.dis_weight_scale
                
                x_ang = abs(p_meas[1] - r_meas[1])
                x_ang = wrap_ang(x_ang)
                    
                w_a = self.compute_weights(0, self.ang_std, x_ang)
                w_a /= self.ang_weight_scale
                
                w = self.dis_weight*w_d*self.ang_weight*w_a 
                
                if w > best_weight:
                    best_weight = w

            self.weight += best_weight
            
        if len(self.measurements) > 0:
            self.weight = self.weight/len(self.measurements)
            N_diff = len(self.measurements)
            if len(robot_measurements) > 0:
                N_diff = abs(len(self.measurements) - len(robot_measurements))  
            else:
                N_diff = len(self.measurements)

            self.weight = self.weight/(N_diff + 1)

        self.weight = self.weight**2