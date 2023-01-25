import TargetElement as my_classes

camera_index = 0

fs = .5 #Hz

# PRE-CONFIGURED FOR WINDOWS, Unix systems => change flag to false and maybe predefined 
# paths/commands
IS_WINDOWS = True
action_performs: list = [
    my_classes.TargetElement("FILE", path="C:\\Program Files\\Mozilla Firefox\\firefox.exe"),
    my_classes.TargetElement("FILE", path="C:\\Users\\peter\\AppData\\Local\\Discord\\Update.exe --processStart Discord.exe"),
    my_classes.TargetElement("ACTION", command="notepad"),
    my_classes.TargetElement("ACTION", path="C:\\Program Files (x86)\\Microsoft\Edge\\Application\\msedge.exe"),
    my_classes.TargetElement("FILE", path="C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"),
    my_classes.TargetElement("ACTION", command = "ls"),
    my_classes.TargetElement("ACTION", command = "shutdown -t 0 -r") #it won't go, 6 classes
]