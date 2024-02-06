from sim import *

#initialise simulation setting
sim_params = {
    'dt': 0.01,
    'end_t': 120,
    'max_x':  100,
    'min_x': -100,
    'max_y':  100,
    'min_y': -100,
    'N_point_cloud': 100,
}

#initialise robot setting
max_fv = 1. #max forward velocity
max_av = np.deg2rad(3) #max angular velocity

robot_params = {
    'init_x': 0.,
    'init_y': 0.,
    'init_v': 1.,
    'init_yaw': np.random.uniform(0, np.math.pi*2.),
    'max_fv': max_fv, 
    'max_av': max_av,
}

#initialise particle filter parameters
particle_filter_params = {
                            'N_particles': 100,
                            'speed_std': max_fv*0.5,
                            'yaw_dot_std': max_av*0.5
                        }

#initialise sensor filter parameter
sensor_params = {
    'lidar_max_range': 20,
    'gyro_std': np.deg2rad(0.01),
    'gps_pos_std': 3.
}

sim (sim_params, 
     robot_params,
     particle_filter_params,
     sensor_params, 
     is_animate = True,
     is_save_gif = False)