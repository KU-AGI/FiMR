import os
from omegaconf import OmegaConf
import json

def save_config(save_path, config):
    config_save_path = os.path.join(save_path, 'config.yaml') 
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
        
    with open(config_save_path, "w") as f:
        if isinstance(config, dict):
            json.dump(config, f, indent=4)
        else:
            json.dump(OmegaConf.to_container(config), f, indent=4)