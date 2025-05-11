import os
import glob
import shutil
import zipfile
import requests
from config import config_vars


def download_file(url, dest_path):
    response = requests.get(url)
    assert response.ok, f"Failed to download {url}"
    with open(dest_path, 'wb') as f:
        f.write(response.content)
    return dest_path


def extract_and_flatten(zip_path, extract_root, subfolder_name):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file in zip_ref.namelist():
            if file.startswith(f"{subfolder_name}/"):
                zip_ref.extract(file, extract_root)

    os.remove(zip_path)

    extracted_path = os.path.join(extract_root, subfolder_name)
    for file in glob.glob(os.path.join(extracted_path, '*')):
        shutil.move(file, extract_root)
    shutil.rmtree(extracted_path)


def prepare_images():
    os.makedirs(config_vars['raw_images_dir'], exist_ok=True)
    zip_path = os.path.join(config_vars['raw_images_dir'], 'images.zip')
    download_file('https://data.broadinstitute.org/bbbc/BBBC039/images.zip', zip_path)
    extract_and_flatten(zip_path, config_vars['raw_images_dir'], 'images')


def prepare_masks():
    os.makedirs(config_vars['raw_annotations_dir'], exist_ok=True)
    zip_path = os.path.join(config_vars['raw_annotations_dir'], 'masks.zip')
    download_file('https://data.broadinstitute.org/bbbc/BBBC039/masks.zip', zip_path)
    extract_and_flatten(zip_path, config_vars['raw_annotations_dir'], 'masks')


def prepare_metadata():
    zip_path = os.path.join(config_vars['root_directory'], 'metadata.zip')
    download_file('https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip', zip_path)
    extract_and_flatten(zip_path, config_vars['root_directory'], 'metadata')


def main():
    prepare_images()
    prepare_masks()
    prepare_metadata()


if __name__ == '__main__':
    main()
