"""Robot controller interface/implementations provide a common, low-level
interface to various robot arms.
"""

import warnings
try:
    import pyfranka
except:
    print('To use Franka robots pyfranka needs installing')

from cri.controller import RobotController


class PyfrankaController(RobotController):
    """Pyfranka controller class implements common interface to robot arms.

    Poses and coordinate frames are specified using 3D Cartesian positions
    and quaternion rotations.  This format makes it easy to perform
    coordinate transformations.
    """

    def __init__(self, ip='172.16.0.1'):
        self._ip = ip
        self._client = pyfranka.Robot(ip)
        self._client.recover_from_errors()
        self._client.set_default_behavior()

        self._gripper = None

        self.set_units('millimeters', 'degrees')
        self.tcp = (0, 0, 0, 0, 1, 0, 0)    # Re-align TCP with base frame

        self.rel_velocity = 0.1
        self.rel_accel = 0.1
        self.rel_jerk = 0.1

    @property
    def gripper(self):
        """Returns a gripper object for this robot.
        """
        if self._gripper is None:
            self._gripper = pyfranka.Gripper(self._ip)
        return self._gripper

    @property
    def info(self):
        """Returns a unique robot identifier string.
        """
        return "ip: {}, server version: {}".format(self._ip, self._client.server_version)

    @property
    def tcp(self):
        """Returns the tool center point (TCP) of the robot.
        """
        return self._tcp

    @tcp.setter
    def tcp(self, tcp):
        """Sets the tool center point (TCP) of the robot.
        """
        self._tcp = np.array(tcp, dtype=np.float64).ravel()
        self._client.ee_frame = (self._tcp[:3] * self._scale_linear, self._tcp[3:])
        self._client.stiffness_frame = (self._tcp[:3] * self._scale_linear, self._tcp[3:])

    @property
    def max_trans_velocity(self):
        """Returns the maximum translational velocity of the robot TCP (mm/s).
        """
        return self._client.max_trans_velocity / self._scale_linear

    @max_trans_velocity.setter
    def max_trans_velocity(self, val):
        """Sets the maximum translational velocity of the robot TCP (mm/s).
        """
        self._client.max_trans_velocity = val * self._scale_linear

    @property
    def max_rot_velocity(self):
        """Returns the maximum rotational velocity of the robot TCP (deg/s).
        """
        return self._client.max_rot_velocity / self._scale_angle

    @max_rot_velocity.setter
    def max_rot_velocity(self, val):
        """Sets the maximum rotational velocity of the robot TCP (deg/s).
        """
        self._client.max_trans_velocity = val * self._scale_angle

    @property
    def max_joint_velocity(self):
        """Returns the maximum joint velocity of the robot (deg/s).
        """
        return self._client.max_joint_velocity / self._scale_angle

    @max_joint_velocity.setter
    def max_joint_velocity(self, val):
        """Sets the maximum joint velocity of the robot TCP (deg/s).
        """
        self._client.max_joint_velocity = val * self._scale_angle

    @property
    def max_trans_accel(self):
        """Returns the maximum translational acceleration of the robot TCP (mm/s/s).
        """
        return self._client.max_trans_accel / self._scale_linear

    @max_trans_accel.setter
    def max_trans_accel(self, val):
        """Sets the maximum translational acceleration of the robot TCP (mm/s/s).
        """
        self._client.max_trans_accel = val * self._scale_linear

    @property
    def max_rot_accel(self):
        """Returns the maximum rotational acceleration of the robot TCP (deg/s/s).
        """
        return self._client.max_rot_accel / self._scale_angle

    @max_rot_accel.setter
    def max_rot_accel(self, val):
        """Sets the maximum rotational acceleration of the robot TCP (deg/s/s).
        """
        self._client.max_rot_accel = val * self._scale_angle

    @property
    def max_joint_accel(self):
        """Returns the maximum joint acceleration of the robot (deg/s/s).
        """
        return self._client.max_joint_accel / self._scale_angle

    @max_joint_accel.setter
    def max_joint_accel(self, val):
        """Sets the maximum joint acceleration of the robot TCP (deg/s/s).
        """
        self._client.max_joint_accel = val * self._scale_angle

    @property
    def max_trans_jerk(self):
        """Returns the maximum translational jerk of the robot TCP (mm/s/s/s).
        """
        return self._client.max_trans_jerk / self._scale_linear

    @max_trans_jerk.setter
    def max_trans_jerk(self, val):
        """Sets the maximum translational jerk of the robot TCP (mm/s/s/s).
        """
        self._client.max_trans_jerk = val * self._scale_linear

    @property
    def max_rot_jerk(self):
        """Returns the maximum rotational jerk of the robot TCP (deg/s/s/s).
        """
        return self._client.max_rot_jerk / self._scale_angle

    @max_rot_jerk.setter
    def max_rot_jerk(self, val):
        """Sets the maximum rotational jerk of the robot TCP (deg/s/s/s).
        """
        self._client.max_rot_jerk = val * self._scale_angle

    @property
    def max_joint_jerk(self):
        """Returns the maximum joint jerk of the robot TCP (deg/s/s/s).
        """
        return self._client.max_joint_jerk / self._scale_linear

    @max_joint_jerk.setter
    def max_joint_jerk(self, val):
        """Sets the maximum joint jerk of the robot TCP (deg/s/s/s).
        """
        self._client.max_joint_jerk = val * self._scale_angle

    @property
    def rel_velocity(self):
        """Returns the relative velocity of the robot
        """
        return self._rel_velocity

    @rel_velocity.setter
    def rel_velocity(self, rel_velocity):
        """Sets the velocity to rel_velocity % of the maximum (0 < rel_velocity <= 1)
        """
        self._rel_velocity = rel_velocity
        self._client.rel_velocity = rel_velocity

    @property
    def rel_accel(self):
        """Returns the relative acceleration of the robot
        """
        return self._rel_accel

    @rel_accel.setter
    def rel_accel(self, rel_accel):
        """Sets the acceleraton to rel_accel % of the maximum (0 < rel_accel <= 1)
        """
        self._rel_accel = rel_accel
        self._client.rel_accel = rel_accel

    @property
    def rel_jerk(self):
        """Returns the relative jerk of the robot
        """
        return self._rel_jerk

    @rel_jerk.setter
    def rel_jerk(self, rel_jerk):
        """Sets the jerk to rel_jerk % of the maximum (0 < rel_jerk <= 1)
        """
        self._rel_jerk = rel_jerk
        self._client.rel_jerk = rel_jerk

    @property
    def linear_accel(self):
        """Returns the linear acceleration of the robot TCP (mm/s/s).
        """
        warnings.warn("linear_accel property not implemented in pyfranka")
        return None

    @linear_accel.setter
    def linear_accel(self, accel):
        """Sets the linear acceleration of the robot TCP (mm/s/s).
        """
        warnings.warn("linear_accel property not implemented in pyfranka")

    @property
    def linear_speed(self):
        """Returns the linear speed of the robot TCP (mm/s).
        """
        warnings.warn("linear_speed property not implemented in pyfranka")
        return None

    @linear_speed.setter
    def linear_speed(self, speed):
        """Sets the linear speed of the robot TCP (mm/s).
        """
        warnings.warn("linear_speed property not implemented in pyfranka")

    @property
    def angular_accel(self):
        """Returns the angular acceleration of the robot TCP (deg/s/s).
        """
        warnings.warn("angular_accel property not implemented in pyfranka")
        return None

    @angular_accel.setter
    def angular_accel(self, accel):
        """Sets the angular acceleration of the robot TCP (deg/s/s).
        """
        self._client.set_angular_accel(accel)
        warnings.warn("angular_accel property not implemented in pyfranka")

    @property
    def angular_speed(self):
        """Returns the angular speed of the robot TCP (deg/s).
        """
        warnings.warn("angular_speed property not implemented in pyfranka")
        return None

    @angular_speed.setter
    def angular_speed(self, speed):
        """Sets the angular speed of the robot TCP (deg/s).
        """
        warnings.warn("angular_speed property not implemented in pyfranka")

    @property
    def blend_radius(self):
        """Returns the robot blend radius (mm).
        """
        warnings.warn("blend_radius property not implemented in pyfranka")
        return None

    @blend_radius.setter
    def blend_radius(self, blend_radius):
        """Sets the robot blend radius (mm).
        """
        warnings.warn("blend_radius property not implemented in pyfranka")

    def set_units(self, linear, angular):
        """Sets linear and angular units.
        """
        units_l = {'millimeters' : 0.001,
                   'meters' : 1.0,
                   'inches' : 0.0254,
                   }
        units_a = {'degrees' : 0.0174533,
                   'radians' : 1.0,
                   }
        self._scale_linear = units_l[linear]
        self._scale_angle  = units_a[angular]

    @property
    def joint_angles(self):
        """Returns the current joint angles.
        """
        joint_angles = self._client.current_joints
        joint_angles /= self._scale_angle
        return joint_angles

    @property
    def desired_joint_angles(self):
        """Returns the desired joint angles.
        """
        joint_angles = self._client.desired_joints
        joint_angles /= self._scale_angle
        return joint_angles

    @property
    def commanded_joint_angles(self):
        """Returns the commanded joint angles.
        """
        joint_angles = self._client.commanded_joints
        joint_angles /= self._scale_angle
        return joint_angles

    @property
    def joint_velocities(self):
        """Returns the current joint velocities.
        """
        joint_velocities = self._client.current_joints_velocity
        joint_velocities /= self._scale_angle
        return joint_velocities

    @property
    def desired_joint_velocities(self):
        """Returns the desired joint velocities.
        """
        joint_velocities = self._client.desired_joints_velocity
        joint_velocities /= self._scale_angle
        return joint_velocities

    @property
    def commanded_joint_velocities(self):
        """Returns the commanded joint velocities.
        """
        joint_velocities = self._client.commanded_joints_velocity
        joint_velocities /= self._scale_angle
        return joint_velocities

    @property
    def pose(self):
        """Returns the current base frame pose.
        """
        pose = np.concatenate(self._client.current_pose)
        pose[:3] /= self._scale_linear
        return pose

    @property
    def desired_pose(self):
        """Returns the desired base frame pose.
        """
        pose = np.concatenate(self._client.desired_pose)
        pose[:3] /= self._scale_linear
        return pose

    @property
    def commanded_pose(self):
        """Returns the commanded base frame pose.
        """
        pose = np.concatenate(self._client.commanded_pose)
        pose[:3] /= self._scale_linear
        return pose

    @property
    def linear_velocity(self):
        """Returns the linear velocity.
        """
        linear_velocity = self._client.linear_velocity
        linear_velocity[:3] /= self._scale_linear
        linear_velocity[3:] /= self._scale_angle
        return linear_velocity

    @property
    def desired_linear_velocity(self):
        """Returns the desired linear velocity.
        """
        linear_velocity = self._client.desired_linear_velocity
        linear_velocity[:3] /= self._scale_linear
        linear_velocity[3:] /= self._scale_angle
        return linear_velocity

    @property
    def commanded_linear_velocity(self):
        """Returns the commanded linear velocity.
        """
        linear_velocity = self._client.commanded_linear_velocity
        linear_velocity[:3] /= self._scale_linear
        linear_velocity[3:] /= self._scale_angle
        return linear_velocity

    @property
    def elbow(self):
        """Returns the current elbow angle.
        """
        elbow = self._client.current_elbow
        elbow[0] /= self._scale_angle
        return elbow[0]

    @property
    def desired_elbow(self):
        """Returns the desired elbow angle.
        """
        elbow = self._client.desired_elbow
        elbow[0] /= self._scale_angle
        return elbow[0]

    @property
    def commanded_elbow(self):
        """Returns the commanded elbow angle.
        """
        elbow = self._client.commanded_elbow
        elbow[0] /= self._scale_angle
        return elbow[0]

    def move_joints(self, joint_angles):
        """Executes an immediate move to the specified joint angles.
        """
        joint_angles = np.array(joint_angles, dtype=np.float64).ravel()
        joint_angles *= self._scale_angle

        self._client.move_joints(joint_angles)

    def move_linear(self, pose, elbow=None):
        """Executes a linear/cartesian move from the current base frame pose to
        the specified pose, using the optional target elbow angle.
        """
        pose = np.array(pose, dtype=np.float64).ravel()
        pose[:3] *= self._scale_linear

        if elbow is None:
            self._client.move_linear((pose[:3], pose[3:]))
        else:
            elbow *= self._scale_angle
            target_elbow = self._client.current_elbow
            target_elbow[0] = elbow
            self._client.move_linear((pose[:3], pose[3:]), target_elbow)

    def move_joints_velocity(self, joints_velocity):
        """Executes an immediate move to the specified joint velocities.
        """
        joints_velocity = np.array(joints_velocity, dtype=np.float64).ravel()
        joints_velocity *= self._scale_angle

        self._client.move_joints_velocity(joints_velocity)

    def move_linear_velocity(self, linear_velocity, elbow=None):
        """Executes an immediate move to the specified linear velocity.
        """
        linear_velocity = np.array(linear_velocity, dtype=np.float64).ravel()
        linear_velocity[:3] *= self._scale_linear
        linear_velocity[3:] *= self._scale_angle

        if elbow is None:
            self._client.move_linear_velocity(linear_velocity)
        else:
            elbow *= self._scale_angle
            target_elbow = self._client.current_elbow
            target_elbow[0] = elbow
            self._client.move_linear_velocity(linear_velocity, target_elbow)

    def move_circular(self, via_pose, end_pose, elbow=None):
        """Executes a movement in a circular path from the current base frame
        pose, through via_pose, to end_pose.
        """
        raise NotImplementedError

    def set_collision_behavior(self,
                               lower_torque_thresholds_accel,
                               upper_torque_thresholds_accel,
                               lower_torque_thresholds_nominal,
                               upper_torque_thresholds_nominal,
                               lower_force_thresholds_accel,
                               upper_force_thresholds_accel,
                               lower_force_thresholds_nominal,
                               upper_force_thresholds_nominal):
        self._client.set_collision_behavior(lower_torque_thresholds_accel,
            upper_torque_thresholds_accel,
            lower_torque_thresholds_nominal,
            upper_torque_thresholds_nominal,
            lower_force_thresholds_accel,
            upper_force_thresholds_accel,
            lower_force_thresholds_nominal,
            upper_force_thresholds_nominal)

    def set_joint_impedance(self, impedance):
        """Sets the joint impedances.
        """
        impedance = np.array(impedance, dtype=np.float64).ravel()
        self._client.set_joint_impedance(impedance)

    def set_cartesian_impedance(self, impedance):
        """Sets the cartesian impedances.
        """
        impedance = np.array(impedance, dtype=np.float64).ravel()
        self._client.set_cartesian_impedance(impedance)

    def recover_from_errors(self):
        """Recover from errors and reset Franka controller.
        """
        self._client.recover_from_errors()

    def close(self):
        """Recover from errors and reset Franka controller.
        """
        self._client.recover_from_errors()
