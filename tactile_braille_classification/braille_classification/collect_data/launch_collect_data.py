"""
python launch_collect_data.py -r sim -s tactip -t arrows
"""
import os

from tactile_data.braille_classification import BASE_DATA_PATH
from tactile_image_processing.collect_data.collect_data import collect_data
from tactile_image_processing.collect_data.setup_targets import setup_targets
from tactile_image_processing.process_data.process_image_data import process_image_data, partition_data
from tactile_image_processing.utils import make_dir

from braille_classification.collect_data.setup_collect_data import setup_collect_data
from braille_classification.utils.parse_args import parse_args
from braille_classification.utils.setup_embodiment import setup_embodiment


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        for args.data_dir, args.sample_num in zip(args.data_dirs, args.sample_nums):

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, args.data_dir)
            image_dir = os.path.join(save_dir, "sensor_images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_image_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_image_params
            )

            # setup targets to collect
            target_df = setup_targets(
                collect_params,
                args.sample_num,
                save_dir
            )

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process(args, image_params, split=None):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        path = os.path.join(BASE_DATA_PATH, output_dir, args.task)
        data_dirs = partition_data(path, args.data_dirs, split)
        process_image_data(path, data_dirs, image_params)


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['alphabet'],
        data_dirs=['train', 'test'],
        sample_nums=[100, 50] # per key
    )

    process_params = {
        "bbox": (12, 12, 240, 240)  # sim (12, 12, 240, 240)
    }

    launch(args)
    process(args, process_params)#, split=0.8)
