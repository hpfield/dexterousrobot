"""
python launch_training.py -i ur_tactip -o sim_tactip -t edge_2d -m pix2pix_128 -v tap
"""
import os
import itertools as it

from tactile_data.braille_classification import BASE_DATA_PATH as INPUT_DATA_PATH
from tactile_data.tactile_sim2real import BASE_DATA_PATH as TARGET_DATA_PATH
from tactile_data.tactile_sim2real import BASE_MODEL_PATH
from tactile_image_processing.utils import make_dir
from tactile_learning.pix2pix.image_generator import Pix2PixImageGenerator
from tactile_learning.pix2pix.models import create_model
from tactile_learning.pix2pix.train_pix2pix import train_pix2pix
from tactile_learning.utils.utils_learning import seed_everything

from tactile_sim2real.learning.setup_training import setup_training
from tactile_sim2real.utils.parse_args import parse_args


def launch(args):

    input_train_data_dirs = [
        os.path.join(INPUT_DATA_PATH, *i) for i in it.product(args.inputs, args.tasks, args.train_dirs)
    ]
    target_train_data_dirs = [
        os.path.join(TARGET_DATA_PATH, *i) for i in it.product(args.targets, args.tasks, args.train_dirs)
    ]
    input_val_data_dirs = [
        os.path.join(INPUT_DATA_PATH, *i) for i in it.product(args.inputs, args.tasks, args.val_dirs)
    ]
    target_val_data_dirs = [
        os.path.join(TARGET_DATA_PATH, *i) for i in it.product(args.targets, args.tasks, args.val_dirs)
    ]

    for args.model in args.models:

        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))
        output_dir = "_to_".join([*args.inputs, *args.targets])
        task_dir = "_".join(args.tasks)

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, task_dir, model_dir_name)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, image_params = setup_training(
            args.model,
            input_train_data_dirs,
            save_dir
        )

        # configure dataloaders
        train_generator = Pix2PixImageGenerator(
            input_train_data_dirs,
            target_train_data_dirs,
            **{**image_params['image_processing'], **image_params['augmentation']}
        )
        val_generator = Pix2PixImageGenerator(
            input_val_data_dirs,
            target_val_data_dirs,
            **image_params['image_processing']
        )

        # create the model
        seed_everything(learning_params['seed'])
        generator, discriminator = create_model(
            image_params['image_processing']['dims'],
            model_params,
            device=args.device
        )

        # run training
        train_pix2pix(
            generator,
            discriminator,
            train_generator,
            val_generator,
            learning_params,
            image_params['image_processing'],
            save_dir,
            device=args.device
        )


if __name__ == "__main__":

    args = parse_args(
        inputs=['ur_tactip_small'],
        targets=['sim_ur_tactip'],
        tasks=['alphabet'],
        train_dirs=['train'],
        val_dirs=['val'],
        models=['pix2pix_128'],
        # model_version=['']
    )

    launch(args)
