import os
import json
import pickle
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPatz

from tactile_data_shear.tactile_servo_control import BASE_DATA_PATH


def load_pkl_obj(name):
    with open(name + ".pkl", "rb") as f:
        return pickle.load(f)


def save_json_obj(obj, name):
    with open(name + ".json", "w") as fp:
        json.dump(obj, fp)


if __name__ == '__main__':

    file_path = os.path.join(BASE_DATA_PATH, 'franka_tactip_2', 'surface_3d', 'val', 'meta')

    meta_dict = load_pkl_obj(file_path)

    del meta_dict['root_dir'], meta_dict['data_dir'], meta_dict['image_target_file'], meta_dict['image_dir']
    print(meta_dict)

    save_json_obj(meta_dict, file_path)
