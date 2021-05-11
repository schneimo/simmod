"""
Copyright (c) 2021, Moritz Schneider.
"""
from typing import Dict, AnyStr
import yaml


def load_yaml(path: AnyStr) -> Dict:
    with open(path) as json_file:
        data = yaml.load(json_file, Loader=yaml.FullLoader)
    return data