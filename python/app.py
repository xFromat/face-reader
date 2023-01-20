import cv2
import sys
import config
import subprocess


cam_access = cv2.VideoCapture(config.camera_index)

if cam_access is None or not cam_access.isOpened():
    sys.exit("CAMERA NOT FOUND")

t_msec = 1000*(0*60 + 10)
cam_access.set(cv2.CAP_PROP_POS_MSEC, t_msec)
ret, frame = cam_access.read()
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cv2.imshow('frame',frame);cv2.waitKey(0)

print(config.action_performs[0].path)

# subprocess.Popen(config.action_performs[0].path)

cam_access.release()