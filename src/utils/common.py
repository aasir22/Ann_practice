import yaml

def get_config(config_path):
    with open(config_path) as config:
        config_content = yaml.safe_load(config)
        return config_content