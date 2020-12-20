"""
Copyright (c) 2020, Moritz Schneider
@Author: Moritz Schneider
"""
from typing import Dict, AnyStr
import yaml


def load_yaml(path: AnyStr) -> Dict:
    with open(path) as json_file:
        data = yaml.load(json_file, Loader=yaml.FullLoader)
    return data