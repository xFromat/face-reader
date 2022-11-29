import cv2
from matplotlib import pyplot as plt

cam_access = cv2.VideoCapture(0)

# ret, frame = cam_access.read()
# plt.imshow(frame)

cam_access.release()
