class Gearbox:
    def __init__(self, ratio, efficiency):
        self.N = ratio
        self.eta = efficiency
    
    def reflect_inertia(self, J_load):
        return J_load / (self.N ** 2)
    
    def reflect_viscous(self, k_load):
        # Viscous drag on the load side, transmitted through gearbox
        return k_load / (self.N ** 2 * self.eta)
    
    def reflect_torque(self, tau_load):
        # Constant load torque on the load side
        return tau_load / (self.N * self.eta)
    
    def output_speed(self, motor_omega):
        return motor_omega / self.N
