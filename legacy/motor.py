from numpy import load, pi, exp

# https://eprints.polsri.ac.id/4591/14/7%20Motor%20DC%208p.pdf
# Using above datasheet for temporary model parameters, specifically the C23-L33 winding code 20
class Motor:
    
    def __init__(self) -> None:
        self.kv = 0.0301 # 
        self.kt = 0.0300 # 
        self.dampping = 0.000001
        self.R = 1.0 # Winding resistance
        self.rot_vel = 0
        self.prev_vel = 0
        self.rot_inertia = 0.00001554
        self.rot_accel = 0


    def update(self, volts, delta_t, load_torque=0.0, load_inertia=0.0):
        J_total = self.rot_inertia + load_inertia
        damp_coef = self.damping()
        
        # Stall torque (torque at zero speed)
        tau_stall = (self.kt / self.R) * volts
        tau_drive = tau_stall - load_torque
        
        omega_ss = tau_drive / damp_coef
        decay = exp(-damp_coef * delta_t / J_total)
        
        self.rot_vel = omega_ss + (self.rot_vel - omega_ss) * decay

    def torque(self, volts):
        return (self.kt / self.R) * (volts - self.kv * self.rot_vel)

    def damping(self):
        return self.dampping + (self.kt * self.kv) / self.R


class MotorGroup:

    def __init__(self, motors) -> None:
        self.motors: list[Motor] = motors
    def update(self, volts, delta_t, load_torque=0.0, load_inertia=0.0, load_coeff=0.0):
        n = len(self.motors)
        rot_vel = self.motors[0].rot_vel

        tau_stall_total = 0.0
        damp_coef_tot = 0.0
        J_total = 0.0

        for motor in self.motors:
            tau_stall_total += (motor.kt / motor.R) * volts
            damp_coef_tot += motor.damping()
            J_total += motor.rot_inertia

        damp_coef_tot += load_coeff          # viscous load on shared shaft
        J_total += load_inertia

        rot_vel_ss = (tau_stall_total - load_torque) / damp_coef_tot
        decay = exp(-damp_coef_tot * delta_t / J_total)
        new_vel = rot_vel_ss + (rot_vel - rot_vel_ss) * decay

        for motor in self.motors:
            motor.rot_vel = new_vel

