from argparse import ArgumentParser
from utils import MODEL_ARCHIVE_MAP, seed_everything
from config.config import WsdConfig

if __name__=='__main__':
    
    parser = ArgumentParser("Wsd zone")
    parser.add_argument('--model_name', type=str, default='vipubmed-deberta-base', help=f'Please choose the right model name at: {MODEL_ARCHIVE_MAP.keys()}')
    
    

    config = WsdConfig()
    
    