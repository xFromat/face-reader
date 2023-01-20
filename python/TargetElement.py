# import os
class TargetElement:
    ActionType = {"FILE" , "ACTION"}
    def __init__(self, action_type: ActionType, path: str = ""):
        self.action_type = action_type
        if action_type != "ACTION" and len(path) > 0:
            self.path = path