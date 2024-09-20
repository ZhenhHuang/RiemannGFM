import json
import os


class DotDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = DotDict(value)
            self[key] = value


def load_config(config_dict, json_file: str):
    with open(json_file, 'rt') as f:
        config_dict.update(json.load(f))
    configs = DotDict(config_dict)
    f.close()
    return configs


def save_config(config_dict, save_path: str):
    with open(save_path, 'wt') as f:
        json.dump(config_dict, f, indent=4)
    f.close()
    return


def list2str(l):
    s = ""
    for e in l[:-1]:
        s += f"{e}_"
    s += f"{l[-1]}"
    return s