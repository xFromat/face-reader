from tensorflow import keras
import numpy as np
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

def take_face_picture(camera_stream, face_cascade):
    if camera_stream is None or not camera_stream.isOpened():
        sys.exit("CAMERA NOT FOUND")

    ret, frame = camera_stream.read()

    faces = detect_faces(frame, face_cascade)
    if len(faces) < 1:
        return
    faces_size = faces.shape
    if faces_size[0] < 1 or faces_size[1] != 4:
        return
    main_face = get_the_biggest(faces)

    extracted_face = extract_faces(frame, main_face)

    return extracted_face
    # Draw a rectangle around the faces
    # for (x, y, w, h) in main_face:
    #     # Draw rectangle in the face
    #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Rect for the face
    #     extract_faces(frame, x, y, w, h)
    # cv2.imshow('Video frame',frame)

def detect_faces(frame, classifier):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scale_factor = 1.3
    min_neighbors=3
    faces = classifier.detectMultiScale(gray, scale_factor, min_neighbors, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def extract_faces(image, coordinates):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face = gray[coordinates[1]:coordinates[1] + coordinates[3], coordinates[0]:coordinates[0] + coordinates[2]]
    # cv2.imwrite(str(coordinates[2]) + str(coordinates[3]) + '_faces.jpg', face)
    return face

def get_the_biggest(coordinates_list):
    max_size = 0
    max_array = []
    for (x, y, w, h) in coordinates_list:
        current_size = w*h
        if current_size > max_size:
            max_size = current_size
            max_array = [x, y, w, h]
    return max_array