import os
import glob
import logging

import json

from typing import Tuple

import pickle

import face_recognition

import numpy as np
from tqdm import tqdm


class GetEmbedings:
    """
    Searching for faces in images and getting embeddings of those faces
    """
    def __init__(self, list_actors: list, path_load: str,  path_write: str = True):
        self.list_actors = list_actors
        self.path_data = path_load
        self.path_write = path_write

    def get_labels(self) -> dict:
        """
        Creating a vocabulary with labels for each actor
        """
        dict_labels = dict()
        for i, key in enumerate(self.list_actors):
            dict_labels[key] = i
        return dict_labels

    def get_embedings(self) -> Tuple[np.array, list]:
        """
        Obtaining embeddings with recognized faces
        """
        embedings = np.empty(128)
        target = []

        dict_labels = self.get_labels()
        for person in tqdm(self.list_actors):
            files = len(glob.glob(f'{self.path_data}/{person}/*'))
            if files < 2:
                print(f'Remove from dataset: {person}')
            else:
                # get a list of images inside the folder
                images = os.listdir(f"{self.path_data}/{person}")
                logging.info(f'Create embedding for {person}')
                for i, person_img in enumerate(images):
                    try:
                        face = face_recognition.load_image_file(f"{self.path_data}/"
                                                                f"{person}/"
                                                                f"{person_img}")
                        face_bounding_boxes = face_recognition.face_locations(face)
                        # If there is more than one face in the photo, or there is no face in the photo, the pass
                        if len(face_bounding_boxes) == 1:
                            try:
                                face_enc = face_recognition.face_encodings(face)[0]
                                embedings = np.vstack((embedings, face_enc))
                                # Add targeting by the current index
                                target.append(dict_labels[person])
                            except Exception as ex:
                                print(f'Error message {ex}')
                    except Exception as ex:
                        print(f'Error message {ex}')
        return embedings[1:], target

    def get_save_embedding(self) -> None:
        """
        Obtaining Targeting and Embeddings followed by Recording
        """
        embedings, target = self.get_embedings()
        logging.info('Writing embeddings & labels & dict actor ')

        with open(f"{self.path_write}/embeddings.pkl", 'wb') as f:
            pickle.dump(embedings, f)

        with open(f'{self.path_write}/labels.pkl', 'wb') as f:
            pickle.dump(target, f)

        json_object = json.dumps(self.get_labels(), indent=4)
        with open(f'{self.path_write}/dict_labels.json', 'w') as fp:
            fp.write(json_object)
