import os

import glob
import logging
import shutil

import numpy as np

from bing_image_downloader.downloader import download
from PIL import Image

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def load_images(path: str, list_actors: list, limit_load: int = 15) -> None:
    """
    Downloading actors and actress pictures
    :param path: path to dataset
    :param list_actors: dictionary with names of actor/actress
    :param limit_load: limitation on the number of downloaded images
    :return: None
    """
    logging.info('Clean the folder')
    # delete folder with dataset
    try:
        shutil.rmtree(path)
    except OSError as e:
        print("Error: %s : %s" % (path, e.strerror))

    logging.info('Download images of actors by Bing')
    # get every names
    for face in list_actors:
        str_face = f'face {face}'
        # download 15 photos for current name
        download(str_face,
                 limit=limit_load,
                 output_dir=path,
                 adult_filter_off=True,
                 force_replace=False,
                 timeout=60,
                 verbose=False)
        # rename directory
        os.rename(path + '/' + str_face, path + '/' + face)
    logging.info('Completing the loading of actor images')


def resize_images(image: Image, size_new: int) -> np.array:
    """
    Changing the size of images
    :rtype: object
    :param image: image
    :param size_new: image size on one side
    :return: image
    """
    # get size of image
    size = image.size
    # get the coefficient by which to reduce/increase it
    # the image on one side
    coef = size_new / size[0]
    # resize the image
    resized_image = image.resize(
        (int(size[0] * coef), int(size[1] * coef)))
    resized_image = resized_image.convert('RGB')
    return resized_image


def format_images(path: str, list_actors: list, size_new: int) -> None:
    """
    Formatting the size of images
    :param size_new: the size of the image on one side
    :param path: path to the folder with the dataset
    :param list_actors: dictionary of actor/actress names
    :return: None
    """
    logging.info('Formatting the image of actors')
    # get every names
    for face in list_actors:
        # unload all file names from the folder
        files = glob.glob(f'{path}/{face}/*')
        # let's go through the list of files in a loop
        for file in files:
            try:
                file_img = Image.open(file)
                resized_image = resize_images(file_img, size_new)
                resized_image.save(file)
            except Exception as ex:
                logging.info(f'Remove image {file} of {face}\nmessage: {ex}')
                os.remove(file)
