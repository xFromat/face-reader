# import os
ActionType = {"FILE" , "ACTION"}
class TargetElement:
    def __init__(self, action_type: ActionType, path: str = "", command: str = ""):
        self.action_type = action_type
        if action_type != "ACTION" and len(path) > 0:
            self.path = path
        elif action_type == "ACTION":
            self.command = command