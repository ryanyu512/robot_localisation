import numpy as np

def wrap_ang(ang):
    if ang > 2*np.math.pi:
        ang -= 2*np.math.pi
    elif ang < -2*np.math.pi:
        ang += 2*np.math.pi

    return ang

def print_particle_error(robot, particles):
    weights = []
    for particle in particles:
        weights += [particle.weight]

    #compute average particle
    particle_pose = [[p.x, p.y, p.yaw] for p in particles]
    cur_pose = np.array(particle_pose)
    x_mean = np.mean(cur_pose[:, 0])
    y_mean = np.mean(cur_pose[:, 1])
    a_mean = np.mean(cur_pose[:, 2])   

    diff_x = round(abs(robot.x - x_mean), 1)
    diff_y = round(abs(robot.y - y_mean), 1)
    diff_pos = round(diff_x + diff_y, 2)
    diff_theta = round(abs(robot.yaw - a_mean), 2)
    if diff_theta > np.math.pi:
        diff_theta = round(abs(diff_theta - np.math.pi * 2), 2)

    best_particle = weights.index(max(weights))
    
    print("Error: [" + str(diff_pos) + ", " + str(diff_theta) + "]")
    print("Weight Sum: " + str(round(sum(weights), 2)))
    print("Max Weight: " + str(round(particles[best_particle].weight, 2)))
    if (diff_pos < 3) and (diff_theta < 0.5):
        print("Converged!")