from .angle_solver import SMPLAngleSolver
from .ik_solver import IKSolver
from .solver import Solver


class SMPLHybridSolver(Solver):
    def __init__(self, config, robot):
        super().__init__(config, robot)
        self.angle_solver = SMPLAngleSolver(config, robot)
        self.ik_solver = IKSolver(config, robot)

    def __call__(self, target_transforms):
        angle_thetas = self.angle_solver(target_transforms)
        ik_thetas = self.ik_solver(target_transforms)
        angle_thetas[:2] = ik_thetas[:2]
        return angle_thetas

    def update_weights(self, weights):
        self.ik_solver.update_weights(weights)
