from tensorflow import keras
from keras import models
from network_model import compile_model
import random
import cv2
import sys

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

def take_picture(camera_stream, face_cascade):
    if camera_stream is None or not camera_stream.isOpened():
        sys.exit("CAMERA NOT FOUND")

    # t_msec = 1000*(0*60 + 10)
    # camera_stream.set(cv2.CAP_PROP_POS_MSEC, t_msec)
    ret, frame = camera_stream.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale_factor = 1.3
    min_neighbors=5
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # Draw rectangle in the face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 53, 18), 2)  # Rect for the face

    cv2.imshow('Video frame',frame)