import logging
import pickle

from typing import Tuple
from collections import Counter

import yaml
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import src

config = yaml.safe_load(open('config/params.yaml'))
path_load = config['load']['path']['load']
path_write = config['load']['path']['write']
actor = config['load']['images']['actor']
SIZE = config['load']['images']['SIZE']
limit = config['load']['images']['limit_load']

key_load_img = config['train']['key_load_img']
RAND = config['train']['random_state']
test_size = config['train']['test_size']
path_model = config['train']['path_model']

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


def check_count_images(target: list) -> Tuple[int, str]:
    """
    Checking the number of persons found
    :param target: target tags
    :return: minimum number of photos with faces, name of actor/actress
    """
    check = Counter(target)
    min_item = np.inf
    name_check = ''
    for key in check.keys():
        if check[key] < min_item:
            min_item = check[key]
            name_check = key
    return min_item, name_check


def fit(random_state: int,
        test_size: int,
        embedings: np.array,
        target: list,
        path_model: str) -> None:
    """
    Trainig model
    :param embedings: embeddings with recognized faces
    :param test_size: test sample size
    :return: None
    """
    X_train, X_test, y_train, y_test = train_test_split(
        embedings, target, test_size=test_size, stratify=target, random_state=random_state)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    with open(path_model, 'wb') as f:
        pickle.dump(clf, f)
    f1_metric = f1_score(y_test, clf.predict(X_test), average='micro')
    print(f'F1 score = {f1_metric}')


def load_files(path_to: str) -> Tuple[np.array, list]:
    """
    Loading embeddings and targets for training
    :param path_to:
    :return:
    """
    logging.info('Loading embedings & labels')
    with open(f'{path_to}/embedings.pkl', 'rb') as f:
        embedings = pickle.load(f)

    with open(f'{path_to}/labels.pkl', 'rb') as f:
        targets = pickle.load(f)
    return embedings, targets


if __name__ == "__main__":
    # whether it is necessary to download images from the Internet
    if key_load_img is True:
        # downloading images from the internet
        src.load_images(path_load, actor, limit_load=limit)
        # image resizing
        src.format_images(path_load, actor, SIZE)
        # getting embeddings and writing to a folder
        emb = src.GetEmbedings(list_actors=actor, path_load=path_load,  path_write=path_write)
        emb.get_save_embedding()

    # open the saved embeddings and dictionary with actors
    embedings, target_list = load_files(path_write)
    min_item, name_check = check_count_images(target_list)

    if min_item > 1:
        logging.info('Fitting the model')
        fit(random_state=RAND,
            test_size=test_size,
            embedings=embedings,
            target=target_list,
            path_model=path_model)
    else:
        logging.info(f'Problem with size dataset {name_check}')
