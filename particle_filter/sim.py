import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

from robot import *
from particle_filter import *

def sim (sim_params, 
        robot_params,
        particle_filter_params,
        sensor_params, 
        is_animate = True,
        is_save_gif = False):

    #define sim setting
    min_x = sim_params['min_x']
    max_x = sim_params['max_x']
    min_y = sim_params['min_y']
    max_y = sim_params['max_y']
    dt = sim_params['dt']
    end_t = sim_params['end_t']
    N_poles = sim_params['N_point_cloud']

    #define robot setting
    init_x = robot_params['init_x']
    init_y = robot_params['init_y']
    init_v = robot_params['init_v']
    init_yaw = robot_params['init_yaw']

    max_fv = robot_params['max_fv']
    max_av = robot_params['max_av']

    #define particle filter setting
    N_particles = particle_filter_params['N_particles']
    speed_std   = particle_filter_params['speed_std']
    yaw_dot_std = particle_filter_params['yaw_dot_std']

    #define sensor setting
    lidar_max_range = sensor_params['lidar_max_range']
    gyro_std    = sensor_params['gyro_std']
    gps_pos_std = sensor_params['gps_pos_std']

    #initialise simulation step
    sim_steps = np.ceil(end_t/dt).astype(int)
    gps_meas_steps = np.ceil(1/gps_sensor().meas_rate/dt).astype(int)
    lidar_meas_steps = np.ceil(1/lidar_sensor(lidar_max_range).meas_rate/dt).astype(int)

    #initialise robot 
    robot = Robot(lidar_max_range = lidar_max_range)
    robot.init_state(x = init_x, y = init_y, yaw = init_yaw, v = init_v)

    #initialise poles 
    poles = [None]*N_poles
    poles_xy = np.random.uniform(low = [min_x, min_y], high = [max_x, max_y], size = (N_poles, 2))
    for i in range(N_poles):
        poles[i] = [poles_xy[i, 0], poles_xy[i, 1]]

    #initialise innovation history
    inno_hist = []

    #initialise estimation error history
    est_err_hist = []

    #initialise kf estimated state history
    est_state_hist = []

    #initialise kf estimated covariance history
    est_cov_hist = []

    #initialise true state history
    true_state_hist = []

    #initialise gps history
    gps_history = []

    #initialise state history
    state_history = []

    #initialise motion type
    motion_type = None
    curve_type = None
    motion_cnt = 0

    particles = []
    particle_pose = [None]*(sim_steps + 1)

    end_sim_ind = None
    for i in range(0, sim_steps + 1):

        #check termination condition
        if max_x < robot.x or max_y < robot.y or robot.x < min_x or robot.y < min_y: 
            end_sim_ind = i    
            break 

        #================== generate motion command ==================#
        if motion_type is None:
            motion_type = np.random.binomial(1, 0.5, 1)[0]
            curve_type  = np.random.binomial(1, 0.5, 1)[0]

        fv = max_fv
        psi_dot = 0
        if  motion_type == 0: #curve motion
            psi_dot = -max_av if curve_type == 0 else max_av

        motion_cnt += 1

        if motion_cnt > int(sim_steps/10):
            motion_cnt = 0
            motion_type = None
            curve_type = None

        #================== update vehicle state ==================#
        #assume the robot perfectly follow the command
        robot.update(dt, 0., psi_dot)
        state = robot.get_state()

        #================== gyro measurement ==================#
        #assume measurement rate of gyroscope is the same as simulation rate
        meas_psi_dot = psi_dot + np.random.normal(loc = 0, scale = gyro_std)

        #================== gps measurement ==================#
        gps_measurement = None
        gps_update = False
        if (i % gps_meas_steps) == 0:
            robot.gps.x = state[0, 0] + np.random.normal(loc = 0, scale = robot.gps.pos_std)
            robot.gps.y = state[1, 0] + np.random.normal(loc = 0, scale = robot.gps.pos_std)

            gps_measurement = [robot.gps.x, robot.gps.y]
            gps_update = True
            if len(particles) == 0:
                #initialise particle filter             
                particles = [None]*N_particles
                p_pose = [None]*N_particles
                for p_ind in range(N_particles):

                    v = np.random.normal(loc = fv, scale = speed_std)
                    yaw = wrap_ang(np.random.uniform(low = 0, high = np.math.pi*2))
                    x = np.random.normal(loc = robot.gps.x, scale = robot.gps.pos_std)
                    y = np.random.normal(loc = robot.gps.y, scale = robot.gps.pos_std)

                    particles[p_ind] = Particle(x, 
                                                y, 
                                                yaw,
                                                v,
                                                speed_std = speed_std, 
                                                yaw_dot_std = yaw_dot_std,
                                                lidar_max_range = lidar_max_range)

        gps_history.append(gps_measurement)

        #================== particle prediction ==================#
        if len(particles) > 0:
            for p_ind, p in enumerate(particles):
                if gps_update: 
                    particles[p_ind].x = np.random.normal(loc = (robot.gps.x + particles[p_ind].x)/2., scale = robot.gps.pos_std)
                    particles[p_ind].y = np.random.normal(loc = (robot.gps.y + particles[p_ind].y)/2., scale = robot.gps.pos_std)

                p.predict(fv, meas_psi_dot, gyro_std, dt)

        #================== lidar measurement and particle update ==================#
        if (i % lidar_meas_steps) == 0:
            robot.measure(poles)

            for j, p in enumerate(particles):
                p.measure(poles)
                p.update_weight(robot.measurements)
                
            print_particle_error(robot, particles)
            particles = resample_particles(particles, fv, dt)

        # #================== record history for display ==================#
        true_state_hist.append(state)
        particle_pose[i] = [None]*N_particles
        
        if len(particles) > 0:
            particle_pose[i] = [[p.x, p.y, p.yaw] for p in particles]
        else:
            particle_pose[i] = [[0, 0, 0] for p in particles]


    #================== animation ==================#
            
    if end_sim_ind is None:
        end_sim_ind = sim_steps

    fig2 = plt.figure()
    ax2  = plt.subplot(1, 1, 1)

    def init_func():
        ax2.clear()
        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    def update_plot(i):

        #clear plot
        ax2.clear()

        #plot true position
        ax2.plot(true_state_hist[i][0,0], true_state_hist[i][1,0], 'og')
        
        #plot average particle pose
        cur_pose = np.array(particle_pose[i])

        x_mean = np.mean(cur_pose[:, 0])
        y_mean = np.mean(cur_pose[:, 1])
        a_mean = np.mean(cur_pose[:, 2])   

        ax2.plot(x_mean, y_mean, 'oy')    

        plt.legend(['true robot', 
                    'averaged particle'])

        #plot gps history
        gps_data = np.array([m for m in gps_history[0:i+1] if m is not None])
        if gps_data is not None and len(gps_data) > 0:
            ax2.plot(gps_data[:, 0], gps_data[:, 1], '+k')

        #plot pole position
        for p in poles:
            dx = (true_state_hist[i][0,0] - p[0])
            dy = (true_state_hist[i][1,0] - p[1])
            if np.sqrt(dx**2 + dy**2) <= lidar_max_range:
                ax2.plot(p[0], p[1], '.b')

                #plot linkage between robot and point cloud
                x = [true_state_hist[i][0, 0], p[0]]
                y = [true_state_hist[i][1, 0], p[1]]
                ax2.plot(x, y, 'r-')

        #plot robot heading
        ax2.plot([true_state_hist[i][0,0], true_state_hist[i][0,0] + 5*np.cos(true_state_hist[i][2,0])], 
                [true_state_hist[i][1,0], true_state_hist[i][1,0] + 5*np.sin(true_state_hist[i][2,0])], '-g')

        ax2.plot([x_mean, x_mean + 5*np.cos(a_mean)], 
                [y_mean, y_mean + 5*np.sin(a_mean)], '-y')

        ax2.set_xlim([min(min_x, true_state_hist[i][0, 0] - 10), max(max_x, true_state_hist[i][0,0] + 10)])
        ax2.set_ylim([min(min_y, true_state_hist[i][1, 0] - 10), max(max_y, true_state_hist[i][1,0] + 10)])

        plt.xlabel('x position (m)')
        plt.ylabel('y position (m)')

    anim = FuncAnimation(fig2, 
                update_plot,
                frames = np.arange(0, end_sim_ind, 60), 
                init_func = init_func,
                interval = 1,
                repeat = False)

    if is_animate:
        plt.show()

    if is_save_gif:
        writer = animation.PillowWriter(fps=15,
                                        metadata=dict(artist='Me'),
                                        bitrate=1800)

        anim.save('demo.gif', writer=writer)