from typing import Dict
import yaml, os


def load_yaml(path) -> Dict:
    #path = os.path.dirname(__file__)
    #path = os.path.join(path, file_path)
    with open(path) as json_file:
        data = yaml.load(json_file, Loader=yaml.FullLoader)
    return data