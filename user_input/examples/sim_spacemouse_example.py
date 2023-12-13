import pybullet as p
import numpy as np

from cri.robot import SyncRobot
from cri.controller import SimController
from cri.sim.utils.sim_utils import setup_pybullet_env

from cri.transforms import inv_transform_euler
from user_input.spacemouse import SpaceMouse


def main():

    n_eps = 10
    work_frame = (350, 0, 100, -180, 0, 0)  # x->back,  y->left,  z->down, rz->clockwise
    embodiment = setup_pybullet_env()

    spacemouse = SpaceMouse()
    robot = SyncRobot(SimController(embodiment.arm))
    with robot, spacemouse:

        # Set TCP and coordinate frame
        robot.coord_frame = work_frame
        robot.tcp = (0, 0, -85.0, 0, 0, 0)

        for i in range(n_eps):

            # Move to origin of work frame
            robot.move_linear((0, 0, 0, 0, 0, 0))

            d = False
            while not d:

                # get data from space mouse
                sm_state = spacemouse.get_state()
                sm_pose = np.array([
                    sm_state.x,
                    sm_state.y,
                    sm_state.z,
                    sm_state.roll,
                    sm_state.pitch,
                    sm_state.yaw,
                ])

                # get pose of tcp in workframe
                tcp_pose = robot.pose

                # calculate sm pose in workframe
                pose = inv_transform_euler(sm_pose, tcp_pose)

                # move to new pose
                robot.move_linear(pose)

                # check for keyboard interrupts
                q_key = ord("q")
                r_key = ord("r")
                keys = p.getKeyboardEvents()
                if q_key in keys and keys[q_key] & p.KEY_WAS_TRIGGERED:
                    exit()
                elif r_key in keys and keys[r_key] & p.KEY_WAS_TRIGGERED:
                    d = True


if __name__ == "__main__":
    main()
