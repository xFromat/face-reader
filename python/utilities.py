import pickle
from tensorflow import keras
from keras import models
from network_model import compile_model
import random

def make_labels_dict(classes_names: list) -> dict:
    classes_dict = {}
    for i in range(len(classes_names)):
        classes_dict[classes_names[i]] = i
    return classes_dict

def shuffle_data(data_imgs: list, data_labels: list):
    temp = list(zip(data_imgs, data_labels))
    random.shuffle(temp)
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return res1, res2

def get_paths(main_path, folders_names):
    import os
    def get_foldername(foldername, folders):
        return list(filter(lambda x: foldername in x, folders))[0]
    tmp_train = get_foldername('train', folders_names)
    tmp_val = get_foldername('val', folders_names)
    tmp_test = get_foldername('test', folders_names)
    path_train = os.path.join(main_path, tmp_train)
    path_val = os.path.join(main_path, tmp_val)
    path_test = os.path.join(main_path, tmp_test)

    return path_train, path_val, path_test