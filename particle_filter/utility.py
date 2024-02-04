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
    best_particle = weights.index(max(weights))
    # print(particles[best_particle].weight)
    diff_x = round(abs(robot.x - particles[best_particle].x), 1)
    diff_y = round(abs(robot.y - particles[best_particle].y), 1)
    diff_pos = round(diff_x + diff_y, 2)
    diff_theta = round(abs(robot.yaw - particles[best_particle].yaw), 2)
    if diff_theta > np.math.pi:
        diff_theta = round(abs(diff_theta - np.math.pi * 2), 2)
    print("Error: [" + str(diff_pos) + ", " + str(diff_theta) + "]")
    print("Weight Sum: " + str(round(sum(weights), 2)))
    print("Max Weight: " + str(round(particles[best_particle].weight, 2)))
    if (diff_pos < 3) and (diff_theta < 0.5):
        print("Converged!")