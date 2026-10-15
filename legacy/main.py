from motor import Motor, MotorGroup
from gearbox import Gearbox
from matplotlib import pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Setup: gearbox and load
# ---------------------------------------------------------------------------
gearbox = Gearbox(ratio=1, efficiency=1.0)

# Wheel/load inertias and drags (same as your script)
gearbox_J_load = 0.5 * 1.0 * 0.05 ** 2
gearbox_k_load = 0.0002
tau_load_const = 0.0

J_reflected   = gearbox.reflect_inertia(gearbox_J_load)
k_reflected   = gearbox.reflect_viscous(gearbox_k_load)
tau_reflected = gearbox.reflect_torque(tau_load_const)

# ---------------------------------------------------------------------------
# Case A: a MotorGroup with 4 identical motors
# ---------------------------------------------------------------------------
N_MOTORS = 4
group = MotorGroup([Motor() for _ in range(N_MOTORS)])

single = Motor()

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
V = 12.0
delta_t = 0.0001       # 0.1 ms -- fast enough for a nice smooth curve
T = 10.0
steps = int(T / delta_t)
times = np.linspace(0, T, steps)

vel_group  = np.empty(steps)
vel_single = np.empty(steps)

for i in range(steps):
    group.update(volts=V, delta_t=delta_t,
                 load_torque=tau_reflected, load_inertia=J_reflected)
    single.update(volts=V, delta_t=delta_t,
                  load_torque=tau_reflected, load_inertia=J_reflected)

    vel_group[i]  = group.motors[0].rot_vel * 60.0 / (2 * np.pi)
    vel_single[i] = single.rot_vel * 60.0 / (2 * np.pi)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

# Top: overlay group vs lumped
ax1.plot(times, vel_group,  lw=2.5, label=f"MotorGroup ({N_MOTORS} × Motor)")
ax1.plot(times, vel_single,  lw=2.5, label=f"Single Motor")
ax1.set_ylabel("Velocity (rpm)")
ax1.set_title("MotorGroup vs Lumped Single Motor")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

