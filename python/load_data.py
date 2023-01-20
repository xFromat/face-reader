import numpy as np
import os
from PIL import Image
from utilities import shuffle_data
from keras.utils import to_categorical
import pickle
from keras import models
from network_model import compile_model

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
    normalized_data = data/255.
    return normalized_data

def save_to_arr(classes: list, path, classes_numbers: dict) -> np.array:
    imgs = []
    labels = []
    starting_ind = 0
    for class_name in classes:
        # train images
        temp_path = os.path.join(path, class_name)
        temp = load_images(temp_path)
        imgs.extend(temp)
        for i in range(starting_ind, starting_ind+len(temp)):
            labels.append(classes_numbers[class_name])
        starting_ind+=len(temp)

    imgs, labels = shuffle_data(imgs, labels)
    imgs = np.asarray(imgs)
    labels = np.asarray(labels)
    return imgs, labels

DEF_MODEL_PATH = "../../Model/"

def serialize_model(model, file_name, history_model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(DEF_MODEL_PATH+file_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(DEF_MODEL_PATH+file_name+".h5")
    with open(DEF_MODEL_PATH+file_name, 'wb') as file_pi:
        pickle.dump(history_model, file_pi)
    print("Saved model to disk")

def desrialize_model(file_name):
    # load json and create model
    json_file = open(DEF_MODEL_PATH+file_name+".json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(DEF_MODEL_PATH+file_name+".h5")
    print("Loaded model from disk")
    loaded_model = compile_model(loaded_model)
    # history
    with open(DEF_MODEL_PATH+file_name, "rb") as file_pi:
        history = pickle.load(file_pi)
    return loaded_model, history