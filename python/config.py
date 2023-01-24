import TargetElement as my_classes

camera_index = 0

fs = 0.25 #Hz

# PRE-CONFIGURED FOR WINDOWS, Unix systems => change flag to false na maybe predefined 
# the paths/commands
IS_WINDOWS = True
action_performs: list = [
    my_classes.TargetElement("FILE", "C:\\Program Files\\Mozilla Firefox\\firefox.exe"),
    my_classes.TargetElement("FILE", "C:\\Users\\peter\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe"),
    my_classes.TargetElement("FILE", "C:\\Program Files\\Mozilla Firefox\\firefox.exe"),
    my_classes.TargetElement("FILE", "C:\\Users\\peter\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe"),
    my_classes.TargetElement("FILE", "C:\\Program Files\\Mozilla Firefox\\firefox.exe"),
    my_classes.TargetElement("FILE", "C:\\Users\\peter\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe"),
    my_classes.TargetElement("ACTION", command = "shutdown -t 0 -r")
]