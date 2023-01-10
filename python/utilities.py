from tensorflow import keras
from keras import models
from network_model import compile_model
import random

def make_labels_dict(classes_names: list) -> dict:
    classes_dict = {}
    for i in range(len(classes_names)):
        classes_dict[classes_names[i]] = i
    return classes_dict


DEF_MODEL_PATH = "../../Model/"

def serialize_model(model, file_name):
    # serialize model to JSON
    model_json = model.to_json()
    with open(DEF_MODEL_PATH+file_name+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(DEF_MODEL_PATH+file_name+".h5")
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
    return loaded_model

def shuffle_data(data_imgs: list, data_labels: list):
    temp = list(zip(data_imgs, data_labels))
    random.shuffle(temp)
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    # res1 and res2 come out as tuples, and so must be converted to lists.
    res1, res2 = list(res1), list(res2)
    return res1, res2