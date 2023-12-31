import os
import itertools as it

from tactile_data.braille_classification import BASE_DATA_PATH
from tactile_learning.supervised.image_generator import demo_image_generation

from braille_classification.learning.setup_training import setup_learning, setup_model_image
from braille_classification.learning.setup_training import csv_row_to_label
from braille_classification.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        data_dirs=['train_temp', 'val_temp']
    )

    output_dir = '_'.join([args.robot, args.sensor])

    data_dirs = [
        os.path.join(BASE_DATA_PATH, output_dir, *i) for i in it.product(args.tasks, args.data_dirs)
    ]

    learning_params = setup_learning()
    image_params = setup_model_image()

    demo_image_generation(
        data_dirs,
        csv_row_to_label['all'],
        learning_params,
        image_params['image_processing'],
        image_params['augmentation'],
    )
