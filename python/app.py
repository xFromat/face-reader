import cv2
import sys
import config
import subprocess
import utilities as tools
import load_data as ld
import os
import time
import network_model as nt


# cascPath = sys.argv[1]

face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascade_frontalface_default.xml'))

if face_cascade.empty():
    print("not loaded")

model, history_model = ld.desrialize_model("network_trained_model_3", ".\\model")

# classes = ld.get_classes_names(path_train)

cam_access = cv2.VideoCapture(config.camera_index)
print("App started")
while True:
    face = tools.take_face_picture(cam_access, face_cascade)
    if cv2.waitKey(1) == ord('q'):
        # Print feedback
        print('Camera Off')
        break
    detected_class = nt.test_photo(model, face)
    tools.preform_action(config.action_performs[1], is_windows = config.IS_WINDOWS)
    time.sleep(1/config.fs)

cam_access.release()