"""Simple test script for SyncRobot and AsyncRobot class using Magician Controller.
Note: Tested in VSCode & spyder
"""

import numpy as np

from cri.robot import SyncRobot, AsyncRobot
from cri.controller import MagicianController as Controller

np.set_printoptions(precision=2, suppress=True)


def main():
    base_frame = (0, 0, 0, 0, 0, 0)
    work_frame = (200, 0, 100, 0, 0, 0)  # base frame: x->front, y->left, z->up, rz->anticlockwise
    
    with AsyncRobot(SyncRobot(Controller())) as robot:       
        # Set TCP, linear speed,  angular speed and coordinate frame
        robot.tcp = (0, 0, 0, 0, 0, 0 )
        robot.linear_speed = 250
        robot.angular_speed = 250
        robot.coord_frame = work_frame
        print('Linear_speed, Angular_speed:', robot.linear_speed, robot.angular_speed)

        # Display robot info
        print("Robot info: {}".format(robot.info))

        # Display initial joint angles
        print("Initial joint angles: {}".format(np.asarray(robot.joint_angles)))

        # Display initial pose in work frame
        print("Initial pose in work frame: {}".format(robot.pose))
        
        # Move to origin of work frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        print("Robot joint angles",robot.joint_angles)
        print("Robot pose: {}".format(robot.pose))

        # Increase and decrease all joint angles
        print("Increasing and decreasing all joint angles ...")
        robot.move_joints(robot.joint_angles + (10,)*4)   
        print("Target joint angles after increase: {}".format(robot.target_joint_angles))
        print("Joint angles after increase: {}".format(robot.joint_angles))
        robot.move_joints(robot.joint_angles - (10,)*4)  
        print("Target joint angles after decrease: {}".format(robot.target_joint_angles))
        print("Joint angles after decrease: {}".format(robot.joint_angles))
        
        # Move backward and forward
        print("Moving backward and forward ...")        
        robot.move_linear((-20, 0, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        # Move right and left
        print("Moving right and left ...")  
        robot.move_linear((0, -20, 0, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        # Move down and up
        print("Moving down and up ...")  
        robot.move_linear((0, 0, -20, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Turn clockwise and anticlockwise around work frame z-axis
        print("Turning clockwise and anticlockwise around work frame z-axis ...")        
        robot.move_linear((0, 0, 0, 0, 0, -20))
        robot.move_linear((0, 0, 0, 0, 0, 0))

        # Move to offset pose then tap down and up in sensor frame
        print("Moving to 20 mm/ 10 deg offset in all pose dimensions ...")         
        robot.move_linear((20, 20, 20, 0, 0, 10))
        print("Target pose after offset move: {}".format(robot.target_pose))
        print("Pose after offset move: {}".format(robot.pose))
        print("Tapping down and up ...")
        robot.coord_frame = base_frame
        robot.coord_frame = robot.target_pose
        robot.move_linear((0, 0, -20, 0, 0, 0))
        robot.move_linear((0, 0, 0, 0, 0, 0))
        robot.coord_frame = work_frame
        print("Moving to origin of work frame ...")
        robot.move_linear((0, 0, 0, 0, 0, 0))
        
        # Pause before commencing asynchronous tests
        print("Repeating test sequence for asynchronous moves ...")

        # # Increase and decrease all joint angles (async)
        # print("Increasing and decreasing all joint angles ...")
        # robot.async_move_joints(robot.joint_angles + (5, 5, 5, 5))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # print("Target joint angles after increase: {}".format(robot.target_joint_angles))
        # print("Joint angles after increase: {}".format(robot.joint_angles))
        # robot.async_move_joints(robot.joint_angles - (5, 5, 5, 5))
        # print("Getting on with something else while command completes ...")      
        # robot.async_result()
        # print("Target joint angles after decrease: {}".format(robot.target_joint_angles))
        # print("Joint angles after decrease: {}".format(robot.joint_angles))

        # Move backward and forward (async)
        print("Moving backward and forward (async) ...")  
        robot.async_move_linear((20, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")
        robot.async_result()
        robot.async_move_linear((0, 0, 0, 0, 0, 0))
        print("Getting on with something else while command completes ...")      
        robot.async_result()
        
        # # Move right and left
        # print("Moving right and left (async) ...")  
        # robot.async_move_linear((0, 20, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()        
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        
        # # Move down and up (async)
        # print("Moving down and up (async) ...")  
        # robot.async_move_linear((0, 0, 20, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        
        # # Turn clockwise and anticlockwise around work frame z-axis (async)
        # print("Turning clockwise and anticlockwise around work frame z-axis (async) ...")   
        # robot.async_move_linear((0, 0, 0, 0, 0, 20))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()

        # # Move to offset pose then tap down and up in sensor frame (async)
        # print("Moving to 20 mm/deg offset in all pose dimensions (async) ...") 
        # robot.async_move_linear((20, 20, 20, 20, 20, 20))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # print("Target pose after offset move: {}".format(robot.target_pose))
        # print("Pose after offset move: {}".format(robot.pose))
        # print("Tapping down and up (async) ...")
        # robot.coord_frame = base_frame
        # robot.coord_frame = robot.target_pose
        # robot.async_move_linear((0, 0, 20, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()
        # robot.coord_frame = work_frame
        # print("Moving to origin of work frame ...")
        # robot.async_move_linear((0, 0, 0, 0, 0, 0))
        # print("Getting on with something else while command completes ...")
        # robot.async_result()

        print("Final target pose in work frame: {}".format(robot.target_pose))
        print("Final pose in work frame: {}".format(robot.pose))


if __name__ == '__main__':
    main()

