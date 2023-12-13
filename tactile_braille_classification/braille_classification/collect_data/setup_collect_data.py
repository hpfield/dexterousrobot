import os
import numpy as np
import itertools as it

from tactile_image_processing.utils import save_json_obj

KEY_LABEL_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 
    'U', 'V', 'W', 'X', 'Y', 'Z', 'UP', 'DOWN', 'LEFT', 'RIGHT', 
    'NONE', 'SPACE'
]

KEYS_ALPHABET = [
    'Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', 
    'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 
    'Z', 'X', 'C', 'V', 'B', 'N', 'M', 'SPACE', 'NONE'
]

KEYS_ARROWS = [
    'UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE'
]


def setup_sensor_image_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160,    240-160+25, 320+160,    240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if 'sim' in robot:
        sensor_image_params = {
            "type": "standard_tactip",
            "image_size": (256, 256),
            "show_tactile": True
        }
    else:
        sensor_image_params = {
            'type': sensor_type,
            'source': 0,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    if save_dir:
        save_json_obj(sensor_image_params, os.path.join(save_dir, 'sensor_image_params'))

    return sensor_image_params


def setup_collect_params(robot, task, save_dir=None):

    pose_lims_dict = {
        'alphabet': [(-2.5, -2.5, 3, 0, 0, -10), (2.5, 2.5, 5, 0, 0, 10)],
        'arrows':   [(-2.5, -2.5, 3, 0, 0, -10), (2.5, 2.5, 5, 0, 0, 10)],
    }

    # WARNING: urdf does not follow this pattern exactly due to auto placement of STLs.
    # This can introduce some bias in the data due to a slight offset in key placement.

    object_poses = {
        KEY_LABEL_NAMES[10*i+j]: (-17.5*i, 17.5*j, 0, 0, 0, 0)
        for i, j in np.ndindex(3, 10)
    }
    object_poses[KEY_LABEL_NAMES[-2]] = (-17.5*3, 17.5*8, -10, 0, 0, 0)
    object_poses[KEY_LABEL_NAMES[-1]] = (-17.5*3, 17.5*3, 0, 0, 0, 0)
    
    object_poses_dict = {
        'alphabet': {key: object_poses[key] for key in KEYS_ALPHABET},
        'arrows':   {key: object_poses[key] for key in KEYS_ARROWS}
    }

    collect_params = {
        'pose_llims': pose_lims_dict[task][0],
        'pose_ulims': pose_lims_dict[task][1],
        'object_poses': object_poses_dict[task],
        'sample_disk': True,
        'sort': True,
        'seed': 0
    }

    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, save_dir=None):

    if robot.split('_')[0] == 'sim':
        robot = 'sim'

    work_frame_dict = {
        'sim':   (593, -7, 25, -180, 0, 0),
    }

    tcp_pose_dict = {
        'sim':   (0, 0, -85, 0, 0, 90),
    }  # SHOULD BE ROBOT + SENSOR

    env_params = {
        'robot': robot,
        'stim_name': 'static_keyboard',
        'work_frame': work_frame_dict[robot],
        'tcp_pose': tcp_pose_dict[robot],
        'show_gui': True
    }

    if 'sim' in robot:
        env_params['speed'] = float('inf')
        env_params['stim_pose'] = (600, 0, 0, 0, 0, 0)

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, task, save_dir=None):
    collect_params = setup_collect_params(robot, task, save_dir)
    sensor_image_params = setup_sensor_image_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, save_dir)
    
    return collect_params, env_params, sensor_image_params


if __name__ == '__main__':
    pass
