from numpy import array, linspace, pi, zeros
import matplotlib.pyplot as plt

power = 11 * 0.65
tau = 9.5488 * power / 25

rot_intertia = 2.0 * 0.3**2

ang_accel = tau / rot_intertia

max_rpm = 25
max_rad = max_rpm / 60 * 2 * pi

delta_t = 0.01
T = 2.0

time = linspace(0, T, int(T / delta_t))
vel = []
dist = []

curr_vel = 0
for t in time:

    curr_vel = curr_vel + ang_accel * delta_t
    if (curr_vel > max_rad):
        curr_vel = max_rad
    vel.append(curr_vel)

tot_dist = 0
for v in vel:
    dist.append(tot_dist)
    tot_dist += v * delta_t

vel = array(vel)
dist = array(dist)

plt.figure(1)
plt.plot(time, vel * 60 / (2 * pi))
plt.figure(2)
plt.plot(time, dist * 360 / (2 * pi))
plt.show()

