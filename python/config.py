import TargetElement as my_classes

camera_index = 0

fs = 0.25 #Hz

el = my_classes.TargetElement("FILE", "C:\\Program Files\\Mozilla Firefox\\firefox.exe")

action_performs: dict = {
    0: el.__dict__
}