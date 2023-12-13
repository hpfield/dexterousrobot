import pybullet as p

from cri.robot import SyncRobot
from cri.controller import SimController
from cri.sim.utils.sim_utils import setup_pybullet_env

from cri.transforms import inv_transform_euler
from user_input.keyboard import Keyboard


def main():

    n_eps = 10
    work_frame = (350, 0, 100, -180, 0, 0)  # x->back,  y->left,  z->down, rz->clockwise
    embodiment = setup_pybullet_env()

    keyboard = Keyboard()
    robot = SyncRobot(SimController(embodiment.arm))

    with robot, keyboard:

        # Set TCP and coordinate frame
        robot.coord_frame = work_frame
        robot.tcp = (0, 0, -85.0, 0, 0, 0)

        for i in range(n_eps):

            # Move to origin of work frame
            robot.move_linear((0, 0, 0, 0, 0, 0))

            d = False
            while not d:

                # get data from space mouse
                keyboard_pose = keyboard.get_state()

                # get pose of tcp in workframe
                tcp_pose = robot.pose

                # calculate sm pose in workframe
                pose = inv_transform_euler(keyboard_pose, tcp_pose)

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
