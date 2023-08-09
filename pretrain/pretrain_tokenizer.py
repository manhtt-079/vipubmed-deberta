from tokenizer.tokenizer import Tokenizer
from config.config import TokenizerConf

if __name__=='__main__':
    t_conf = TokenizerConf()
    t = Tokenizer(t_conf)
    t.train()