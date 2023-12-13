"""
python evaluate_model.py -m simple_cnn -t arrows
"""
import os
import itertools as it
import pandas as pd
from torch.autograd import Variable
import torch

from tactile_data.braille_classification import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_image_processing.utils import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.utils.utils_plots import ClassificationPlotter

from braille_classification.learning.setup_training import csv_row_to_label
from braille_classification.utils.label_encoder import LabelEncoder
from braille_classification.utils.parse_args import parse_args


def evaluate_model(
    model,
    label_encoder,
    generator,
    learning_params,
    error_plotter,
    device='cpu'
):

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    pred_df = pd.DataFrame()
    targ_df = pd.DataFrame()

    for batch in loader:

        # get inputs
        inputs, labels_dict = batch['inputs'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)

        # decode predictions into label
        predictions_dict = label_encoder.decode_label(outputs)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
        batch_targ_df = pd.DataFrame.from_dict(labels_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)
    metrics = label_encoder.calc_metrics(pred_df, targ_df)

    # plot full error graph
    error_plotter.final_plot(
        pred_df, targ_df, metrics
    )


def evaluation(args):

    # test the trained networks
    for args.task, args.model in it.product(args.tasks, args.models):

        output_dir = '_'.join([args.robot, args.sensor])
        model_dir_name = '_'.join(filter(None, [args.model, *args.model_version]))

        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, dir) for dir in args.val_dirs
        ]

        # set save dir
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)

        # setup parameters
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        label_params = load_json_obj(os.path.join(model_dir, 'model_label_params'))
        image_params = load_json_obj(os.path.join(model_dir, 'model_image_params'))

        # configure dataloader
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label[args.task],
            **image_params['image_processing']
        )

        # create the label encoder/decoder and error plotter
        label_encoder = LabelEncoder(label_params['label_names'], args.device)
        error_plotter = ClassificationPlotter(label_params['label_names'], model_dir, name='error_plot_best.png')

        # create the model
        model = create_model(
            in_dim=image_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=args.device
        )
        model.eval()

        evaluate_model(
            model,
            label_encoder,
            val_generator,
            learning_params,
            error_plotter,
            device=args.device
        )


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        val_dirs=['val_temp'],
        models=['simple_cnn'],
        model_version=['temp'],
        device='cuda'
    )

    evaluation(args)
