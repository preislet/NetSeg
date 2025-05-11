import os
import random
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import skimage.segmentation
import skimage.morphology
from skimage import img_as_ubyte
from tqdm import tqdm

import utils.project_dir_manipulation_tools as dirtools
import utils.augmentation as augmentation
from config import config_vars

# turn off warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skimage")


def create_split_files():
    file_list = os.listdir(config_vars["raw_images_dir"])
    train, test, val = dirtools.create_image_lists(
        config_vars["raw_images_dir"],
        config_vars["training_fraction"],
        config_vars["validation_fraction"]
    )
    dirtools.write_path_files(config_vars["path_files_training"], train)
    dirtools.write_path_files(config_vars["path_files_test"], test)
    dirtools.write_path_files(config_vars["path_files_validation"], val)


def normalize_images():
    os.makedirs(config_vars["normalized_images_dir"], exist_ok=True)
    filelist = sorted(os.listdir(config_vars["raw_images_dir"]))

    for filename in tqdm(filelist, desc="Normalizing images", unit="file"):
        orig_img = skimage.io.imread(os.path.join(config_vars["raw_images_dir"], filename))

        # Normalize to [0,1]
        percentile = 99.9
        high = np.percentile(orig_img, percentile)
        low = np.percentile(orig_img, 100 - percentile)
        img = np.clip(orig_img, low, high)
        img = (img - low) / (high - low)
        img = skimage.img_as_ubyte(img)

        out_path = os.path.join(config_vars["normalized_images_dir"], filename.rsplit('.', 1)[0] + '.png')
        skimage.io.imsave(out_path, img)


def create_boundary_labels():
    os.makedirs(config_vars["boundary_labels_dir"], exist_ok=True)
    filelist = sorted(os.listdir(config_vars["raw_annotations_dir"]))
    total_objects = 0

    for filename in tqdm(filelist, desc="Creating boundary labels"):
        path = os.path.join(config_vars["raw_annotations_dir"], filename)
        annot = skimage.io.imread(path)

        if annot.ndim == 3:
            annot = annot[:, :, 0]

        annot = skimage.morphology.label(annot)
        total_objects += len(np.unique(annot)) - 1
        annot = skimage.morphology.remove_small_objects(annot, min_size=config_vars["min_nucleus_size"])
        boundaries = skimage.segmentation.find_boundaries(annot)

        for k in range(2, config_vars["boundary_size"], 2):
            boundaries = skimage.morphology.binary_dilation(boundaries)

        label_binary = np.zeros((*annot.shape, 3), dtype=np.uint8)
        label_binary[(annot == 0) & (boundaries == 0), 0] = 255
        label_binary[(annot != 0) & (boundaries == 0), 1] = 255
        label_binary[boundaries == 1, 2] = 255

        out_path = os.path.join(config_vars["boundary_labels_dir"], filename)
        skimage.io.imsave(out_path, label_binary)

    print("Total objects:", total_objects)


def generate_augmented_examples(filelist, n_augmentations, n_points, distort, dir_labels, dir_images):
    updated_filelist = []

    for filename in tqdm(filelist, desc="Augmenting images"):
        label_path = os.path.join(dir_labels, filename)
        if not pathlib.Path(label_path).is_file():
            continue

        x = skimage.io.imread(os.path.join(dir_images, filename))
        y = skimage.io.imread(label_path)

        for n in range(1, n_augmentations):
            x_aug, y_aug = augmentation.deform(x, y, points=n_points, distort=distort)

            filename_augmented = os.path.splitext(filename)[0] + f'_aug_{n:03d}' + os.path.splitext(filename)[1]

            # Ensure data type is suitable for PNG
            x_aug = img_as_ubyte(x_aug)
            y_aug = img_as_ubyte(y_aug)

            skimage.io.imsave(os.path.join(dir_images, filename_augmented), x_aug)
            skimage.io.imsave(os.path.join(dir_labels, filename_augmented), y_aug)

            updated_filelist.append(filename_augmented)

    return filelist + updated_filelist


def main():
    if config_vars["create_split_files"]:
        create_split_files()

    data_partitions = dirtools.read_data_partitions(config_vars, load_augmented=False)

    if config_vars["transform_images_to_PNG"]:
        normalize_images()
    else:
        config_vars["normalized_images_dir"] = config_vars["raw_images_dir"]

    create_boundary_labels()

    if config_vars["augment_images"]:
        tmp_value = config_vars["max_training_images"]
        config_vars["max_training_images"] = 0

        tmp_partitions = dirtools.read_data_partitions(config_vars, load_augmented=False)
        _ = generate_augmented_examples(
            tmp_partitions["training"],
            config_vars["elastic_augmentations"],
            config_vars["elastic_points"],
            config_vars["elastic_distortion"],
            config_vars["boundary_labels_dir"],
            config_vars["normalized_images_dir"]
        )
        print(tmp_value)
        print(tmp_partitions)

        config_vars["max_training_images"] = tmp_value


if __name__ == '__main__':
    main()
