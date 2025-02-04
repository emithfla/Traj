import json
import torch

def _load_config(config_name,base_dir='config'):
    confs = ['default.json', config_name]
    # confs = config_name
    config = {}
    for conf in confs:
        config_ = json.load(open(f'{base_dir}/{conf}'))
        for k in config_:
            config[k] = config_[k]
    print(config)
    for k in config:
        globals()[k] = config[k]

_load_config('cnn-gat-sgan.json')

# device = torch.device('cuda:' +str(cuda) if torch.cuda.is_available() else "cpu")

datasets = []
device_ids = [0]