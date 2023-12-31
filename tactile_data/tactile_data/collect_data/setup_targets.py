import os
import numpy as np
import pandas as pd

POSE_LABEL_NAMES = ["pose_x", "pose_y", "pose_z", "pose_Rx", "pose_Ry", "pose_Rz"]
SHEAR_LABEL_NAMES = ["shear_x", "shear_y", "shear_z", "shear_Rx", "shear_Ry", "shear_Rz"]
OBJECT_POSE_LABEL_NAMES = ["object_x", "object_y", "object_z", "object_Rx", "object_Ry", "object_Rz"]


def setup_targets(
    collect_params,
    num_poses=100,
    save_dir=None,
):
    """
    Generates a dataframe with target poses used for data collection.
    """
    pose_label_names = collect_params.get('pose_label_names', POSE_LABEL_NAMES)
    pose_llims = collect_params.get('pose_llims', [0]*6)
    pose_ulims = collect_params.get('pose_ulims', [0]*6)

    shear_label_names = collect_params.get('shear_label_names', SHEAR_LABEL_NAMES)
    shear_llims = collect_params.get('shear_llims', [0]*6)
    shear_ulims = collect_params.get('shear_ulims', [0]*6)

    sample_disk = collect_params.get('sample_disk', False)
    sort = collect_params.get('sort', False)

    object_pose_label_names = collect_params.get('object_pose_label_names', OBJECT_POSE_LABEL_NAMES)
    object_poses_dict = collect_params.get('object_poses', {'1': [0]*6})

    # initialize target df
    target_df = pd.DataFrame(
        columns=[
            "sensor_image",
            "object_label",
            *pose_label_names,
            *shear_label_names,
            *object_pose_label_names
        ]
    )

    # make geneated data predictable
    np.random.seed(collect_params.get('seed', None))

    # save target data
    ind = 0
    for obj_label, obj_pose in object_poses_dict.items():

        # generate random poses and shears
        poses = sample_poses(pose_llims, pose_ulims, num_poses, sample_disk)
        shears = sample_poses(shear_llims, shear_ulims, num_poses, sample_disk)

        # sort parameters by label
        if sort:
            i_sort = -1 if type(sort) is bool else pose_label_names.index(sort)
            poses = poses[poses[:, i_sort].argsort()]

        # populate dataframe
        for i in range(num_poses):
            image_name = f"image_{ind+1}.png"
            pose = poses[i, :]
            shear = shears[i, :]
            target_df.loc[ind] = np.hstack([image_name, obj_label, pose, shear, obj_pose])
            ind += 1

    # save to file
    if save_dir:
        target_file = os.path.join(save_dir, "targets.csv")
        target_df.to_csv(target_file, index=False)

    return target_df


def random_spherical(num_samples, phi_max):
    """Return uniform random sample over a spherical cap bounded by polar angle."""
    phi_max = np.radians(phi_max)                                 # maximum value of polar angle
    theta = 2*np.pi * np.random.rand(num_samples)                 # azimuthal angle samples
    kappa = 0.5 * (1 - np.cos(phi_max))                           # value of cumulative dist function at phi_max
    phi = np.arccos(1 - 2 * kappa * np.random.rand(num_samples))  # polar angle samples

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Compute Rx, Ry component samples for extrinsic-xyz Euler parameterization
    Rx = -np.arcsin(y)        # Rotation around x needed to move (0, 0, 1) to (*, y, *)
    Ry = -np.arctan2(x, z)    # Rotation around y needed to move (*, y, *) to (x, y, z) (r = 1)

    return np.degrees(Rx), np.degrees(Ry)   # degrees


def random_disk(num_samples, r_max):
    """Return uniform random sample over a 2D circular disk of radius r_max."""
    theta = 2*np.pi * np.random.rand(num_samples)
    r = r_max * np.sqrt(np.random.rand(num_samples))
    x, y = r * (np.cos(theta), np.sin(theta))
    theta = np.degrees(theta)
    return x, y


def random_linear(num_samples, x_max):
    """Return uniform random sample over a 1D segment [-x_max, x_max]."""
    return -x_max + 2 * x_max * np.random.rand(num_samples)


def sample_poses(llims, ulims, num_samples, sample_disk):
    poses_mid = (np.array(ulims) + llims) / 2
    poses_max = ulims - poses_mid

    # default linear sampling on all components
    samples = [random_linear(num_samples, x_max) for x_max in poses_max]

    # resample components if circular sampling
    if sample_disk:
        inds_pos = [i for i, v in enumerate(poses_max[:2]) if v > 0]     # only x, y
        inds_rot = [3+i for i, v in enumerate(poses_max[3:5]) if v > 0]  # only Rx, Ry

        if len(inds_pos) == 2:
            r_max = max(poses_max[inds_pos])
            samples_pos = random_disk(num_samples, r_max)

            scales = poses_max[inds_pos] / r_max  # for limits not equal
            samples_pos *= scales[np.newaxis, :2].T

            samples[inds_pos[0]], samples[inds_pos[1]] = samples_pos

        if len(inds_rot) == 2:
            phi_max = max(poses_max[inds_rot])
            samples_rot = random_spherical(num_samples, phi_max)

            scales = poses_max[inds_rot[:2]] / phi_max  # for limits not equal
            samples_rot *= scales[np.newaxis, :].T

            samples[inds_rot[0]], samples[inds_rot[1]] = samples_rot

    poses = np.array(samples).T
    poses += poses_mid

    return poses
