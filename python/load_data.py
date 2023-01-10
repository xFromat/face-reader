import numpy as np
import os
from PIL import Image
import random

def load_images(path) -> list:
    content_list = os.listdir(path)
    files_list = [os.path.join(path, contet_thing) for contet_thing in content_list if os.path.isfile(os.path.join(path, contet_thing))]
    imgs_list = []
    for file_path in files_list:
        temp_img = Image.open(file_path)
        numpy_image = np.asarray(temp_img)
        numpy_image = normalize_data(numpy_image)
        imgs_list.append(numpy_image)
    return imgs_list

def get_classes_names(path) -> list:
    content_list = os.listdir(path)
    dir_list = [contet_thing for contet_thing in content_list if not os.path.isfile(contet_thing)]
    return dir_list

def normalize_data(data) -> np.array:
    if np.max(data) == 0:
        return data
    normalized_data = ((data-np.min(data))/(np.max(data)-np.min(data)))*255
    return normalized_data
