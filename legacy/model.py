
class DrivePod:

    def __init__(self) -> None:
        self.inertia = 0.5 * 0.5 * (0.05 ** 2)

    def get_accel(self, torque):
        return torque / self.inertia

