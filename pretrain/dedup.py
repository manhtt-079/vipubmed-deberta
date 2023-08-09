from dataset.dataset import DatasetPipeLine
from config.config import DatasetConf



if __name__=='__main__':
    conf = DatasetConf()
    d = DatasetPipeLine(conf=conf)
    d.run()
    d.create_dataset()
    d.pre_tokenizer()