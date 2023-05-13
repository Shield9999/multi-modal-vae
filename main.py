import argparse
import yaml

import numpy as np
import torch

from models.modelutils import get_model

def load_config(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-modal VAE')
    parser.add_argument(
        '--config', type=str, default='config.yaml', help='Path to the config file.')
    
    return parser.parse_args()


def main(config):
    # Set random seed
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = get_model(config)


if __name__ == '__main__':
    args = parse_args()
    config = load_config(args)
    print(config)
    main()