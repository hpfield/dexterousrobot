# import time
import os
import cv2
import imageio
import numpy as np
import itertools as it
import torch
from torch.autograd import Variable

from tactile_data.tactile_sim2real import BASE_MODEL_PATH
from tactile_image_processing.utils import load_json_obj
from tactile_image_processing.image_transforms import process_image
from tactile_image_processing.simple_sensors import RealSensor
from tactile_learning.pix2pix.models import create_model
from tactile_learning.utils.utils_learning import seed_everything

from tactile_sim2real.utils.parse_args import parse_args


class GeneratorModel:
    def __init__(
        self,
        generator,
        image_processing_params,
        device='cuda'
    ):
        self.generator = generator
        self.image_processing_params = image_processing_params
        self.device = device

    def process(self, input_image):

        processed_input_image = process_image(
            input_image,
            gray=False,
            **self.image_processing_params
        )

        # channel first for pytorch; add batch dim
        processed_input_image = np.rollaxis(processed_input_image, 2, 0)
        processed_input_image = processed_input_image[np.newaxis, ...]

        # perform inference with the trained model
        model_input = Variable(torch.from_numpy(processed_input_image)).float().to(self.device)
        generated_output_image = self.generator(model_input).detach().cpu().numpy()

        # remove batch (and channel dimension for grayscale)
        processed_input_image = processed_input_image.squeeze()
        generated_output_image = generated_output_image.squeeze()

        return generated_output_image, processed_input_image


def run_live_generation(
    camera,
    generator_model,
    record_video=False,
):

    if record_video:
        video_frames = []

    cv2.namedWindow("GAN_display")
    while True:

        # pull raw frames from camera
        image = camera.process()

        # process with gan here (proccessing applied in GAN class)
        generated_output_image, processed_input_image = generator_model.process(image)

        # create an overlay of the two images
        overlay_image = cv2.addWeighted(
            processed_input_image, 0.25, generated_output_image, 0.9, 0
        )

        # concat all images to display at once
        disp_image = np.concatenate(
            [processed_input_image, generated_output_image, overlay_image], axis=1
        )

        if record_video:
            video_frames.append(disp_image)

        # show image
        cv2.imshow("GAN_display", disp_image)
        k = cv2.waitKey(10)
        if k == 27:  # Esc key to stop
            if record_video:
                video_file = os.path.join("../../readme_images", "tactile_video.mp4")
                imageio.mimwrite(video_file, np.stack(video_frames), fps=20)
            break


if __name__ == "__main__":

    args = parse_args(
        inputs=['ur_tactip'],
        targets=['sim_ur_tactip'],
        tasks=['surface_3d'],
        train_dirs=['train_shear'],
        val_dirs=['val_shear'],
        models=['pix2pix_256'],
        model_version=[''],
    )

    for args.task, args.model in it.product(args.tasks, args.models):

        # get model dir
        output_dir = "_to_".join([*args.inputs, *args.targets])
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, args.model)

        # setup parameters
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(model_dir, 'preproc_params'))
        sensor_params = load_json_obj(os.path.join(model_dir, 'sensor_params'))

        # create the model
        seed_everything(learning_params['seed'])
        generator, _ = create_model(
            preproc_params['image_processing']['dims'],
            model_params,
            saved_model_dir=model_dir,
            device=args.device,
        )
        generator.eval()
        generator_model = GeneratorModel(generator, preproc_params['image_processing'], args.device)

        # overwrite source to current camera
        sensor_params['source'] = 8
        camera = RealSensor(sensor_params)

        run_live_generation(
            camera,
            generator_model,
            record_video=False
        )
