import cv2
import sys
import config
import subprocess
import utilities as tools
import os
import time


# cascPath = sys.argv[1]

face_cascade = cv2.CascadeClassifier(os.path.abspath('haarcascade_frontalface_default.xml'))

if face_cascade.empty():
    print("not loaded")

cam_access = cv2.VideoCapture(config.camera_index)
print("App started")
while True:
    tools.take_picture(cam_access, face_cascade)
    if cv2.waitKey(1) == ord('q'):
        # Print feedback
        print('Camera Off')
        break
        # break
    time.sleep(1/config.fs)

# print(config.action_performs[0].path)

# subprocess.Popen(config.action_performs[0].path)

cam_access.release()



# import win32com.client
 
# wmi = win32com.client.GetObject ("winmgmts:")
# for usb in wmi.InstancesOf ("Win32_USBHub"):
#     print(usb.DeviceID)

#     USB\VID_0C45&PID_636B\SN0001 - this is krux camera through all the hubs