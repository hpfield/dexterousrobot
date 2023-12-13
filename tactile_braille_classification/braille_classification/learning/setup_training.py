import os
import shutil

from tactile_image_processing.utils import save_json_obj
from braille_classification.collect_data.setup_collect_data import KEY_LABEL_NAMES


def csv_row_to_label_arrows(row):
    return {
        'id': KEY_LABEL_NAMES[:5].index(row['object_label']),
        'label': row['object_label'],
    }


def csv_row_to_label_alphabet(row):
    return {
        'id': KEY_LABEL_NAMES[5:].index(row['object_label']),
        'label': row['object_label'],
    }


def csv_row_to_label_all(row):
    return {
        'id': KEY_LABEL_NAMES.index(row['object_label']),
        'label': row['object_label'],
    }


csv_row_to_label = {
    'alphabet': csv_row_to_label_alphabet,
    'arrows': csv_row_to_label_arrows,
    'all': csv_row_to_label_all,
}


def setup_learning(save_dir=None):

    learning_params = {
        'seed': 42,
        'batch_size': 64,
        'epochs': 20,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'adam_b1': 0.9,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 1,
        'n_train_batches_per_epoch': None,
        'n_val_batches_per_epoch': None,
    }

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))

    return learning_params


def setup_model_image(save_dir=None):

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    model_image_params = {
        'image_processing': image_processing_params,
        'augmentation': augmentation_params
    }

    if save_dir:
        save_json_obj(model_image_params, os.path.join(save_dir, 'model_image_params'))

    return model_image_params


def setup_model(model_type, save_dir):

    model_params = {
        'model_type': model_type
    }

    if model_type == 'simple_cnn':
        model_params['model_kwargs'] = {
            'conv_layers': [32, 32, 32, 32],
            'conv_kernel_sizes': [11, 9, 7, 5],
            'fc_layers': [512, 512],
            'activation': 'relu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }

    elif model_type == 'posenet_cnn':
        model_params['model_kwargs'] = {
            'conv_layers': [256, 256, 256, 256, 256],
            'conv_kernel_sizes': [3, 3, 3, 3, 3],
            'fc_layers': [64],
            'activation': 'elu',
            'dropout': 0.0,
            'apply_batchnorm': True,
        }

    elif model_type == 'nature_cnn':
        model_params['model_kwargs'] = {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        }        

    elif model_type == 'resnet':
        model_params['model_kwargs'] = {
            'layers': [2, 2, 2, 2]
        }

    elif model_type == 'vit':
        model_params['model_kwargs'] = {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
            'pool': 'cls',  # for classification
        }    

    else:
        raise ValueError(f'Incorrect model_type specified: {model_type}')

    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_model_labels(task_name, save_dir=None):
    """
    Returns task specific details.
    """

    label_names_dict = {
        'arrows': KEY_LABEL_NAMES[:5],
        'alphabet': KEY_LABEL_NAMES[5:]
    }

    model_label_params = {
        'label_names': label_names_dict[task_name],
        'out_dim': len(label_names_dict[task_name]),
    }

    # save parameters
    if save_dir:
        save_json_obj(model_label_params, os.path.join(save_dir, 'model_label_params'))

    return model_label_params


def setup_training(model_type, task, data_dirs, save_dir=None):
    learning_params = setup_learning(save_dir)
    model_params = setup_model(model_type, save_dir)
    model_label_params = setup_model_labels(task, save_dir)
    model_image_params = setup_model_image(save_dir)

    is_processed = os.path.isdir(os.path.join(data_dirs[0], 'processed_images'))

    # retain data parameters
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'env_params.json'), save_dir)
        if is_processed:
            shutil.copy(os.path.join(data_dirs[0], 'processed_image_params.json'), save_dir)
        else:
            shutil.copy(os.path.join(data_dirs[0], 'sensor_image_params.json'), save_dir)

    return learning_params, model_params, model_label_params, model_image_params
