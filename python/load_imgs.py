import numpy as np
import os

def load_images(path) -> np.array:
    content_list = os.listdir(path)
    dir_list = [contet_thing for contet_thing in content_list if not os.path.isFile(contet_thing)]
    print(dir_list)
    return []
