import json
from typing import Tuple, Union

import pickle
import logging

import numpy as np
from PIL import Image
import face_recognition

import pandas as pd

import yaml
import src


config = yaml.safe_load(open('config/params.yaml'))
path_model = config['train']['path_model']
SIZE = config['predict']['SIZE']
path_load = config['predict']['path_load']

with open('data/processed/dict_labels.json', 'r') as openfile:
    dict_labels = json.load(openfile)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def predict_actor(image: np.array,
                    model: pickle,
                    dict_labels: dict) -> Union[None, Tuple[str, float, pd.DataFrame]]:
    """
    Predicting an actor's/actress' name by photo
    :param image: test image
    :param model: model
    :param dict_labels: dictionary with the names of actors/actresses
    :return: name, probability, frame with general results
    """
    logging.info('Search for a face in a photo')
    face_bounding_boxes = face_recognition.face_locations(image)

    if len(face_bounding_boxes) != 1:
        print('Problem with finding a face')
    else:
        logging.info('Create bbox for a test image')
        # convert photo with face to vector, get embedding
        face_enc = face_recognition.face_encodings(image)[0]
        # predict actress/actor
        predict = model.predict([face_enc])
        predict_name = list(dict_labels.keys())[list(dict_labels.values()).index(predict)]
        predict_proba = model.predict_proba([face_enc])[0][predict][0]

        frame_proba = pd.DataFrame()
        frame_proba['actress'] = list(dict_labels.keys())
        frame_proba['score'] = model.predict_proba([face_enc])[0]

        return predict_name, predict_proba, frame_proba.sort_values(by='score')[::-1]
    return None


if __name__ == "__main__":
    file_img = Image.open(path_load)
    image_resize = np.array(src.resize_images(file_img, size_new=SIZE))

    with open(path_model, 'rb') as f:
        clf = pickle.load(f)
    predict_labels, predict_value, frame_proba = predict_actor(image=image_resize,
                                                                 model=clf,
                                                                 dict_labels=dict_labels)
    print(predict_labels, round(predict_value, 2), '\n')
    print(frame_proba[:5])
    print('Hello!')
