import cv2
import config
import utilities as tools
import load_data as ld
import os
import time
import network_model as nt
import sys

face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascade_frontalface_default.xml'))

if face_cascade.empty():
    print("not loaded")

try:
    model, history_model = ld.desrialize_model("network_trained_model_3", ".\\model")
except:
    print("404: Classificator not found")
    sys.exit()

# classes = ld.get_classes_names(path_train)

cam_access = cv2.VideoCapture(config.camera_index)
if cam_access is None or not cam_access.isOpened():
        sys.exit("CAMERA NOT FOUND")
print("App started")
while True:
    face = tools.take_face_picture(cam_access, face_cascade)
    if cv2.waitKey(1) == ord('q'):
        # Print feedback
        print('Camera Off')
        break
    detected_class = nt.test_photo(model, face)
    print("Detected class number: "+str(detected_class))
    if detected_class is not None:
        tools.preform_action(config.action_performs[detected_class], is_windows = config.IS_WINDOWS)
    time.sleep(1/config.fs)

cam_access.release()