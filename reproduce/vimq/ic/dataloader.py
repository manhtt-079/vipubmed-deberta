import os
import json
import copy
from loguru import logger
from pyvi import ViTokenizer
from datasets import Dataset
from transformers import AutoTokenizer
from utils.utils import path_exists
from config import VIMqConfig

class InputExample(object):
    def __init__(
        self,
        guid: int|str = None,
        sentence: str = None,
        label: int = None
    ) -> None:
        super().__init__()
        
        self.guid = guid
        self.sentence = sentence
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Processor(object):

    def __init__(self, config: VIMqConfig) -> None:
        super().__init__()
        self.config = config

    @path_exists
    def read_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            raise RuntimeError(f'Error when reading: {path}')
    
    def re_tokenize(self, sentence: str):
        return ViTokenizer.tokenize(sentence.replace('_', ' '))
    
    def to_examples(self, json_data: list[dict[str, str|list[tuple[int, int, str]]]]) -> list[InputExample]:
        
        if self.config.re_tokenize:
            examples = [InputExample(sentence=self.re_tokenize(e['sentence']), label=e['sent_label']) for e in json_data]
        else:
            examples = [InputExample(sentence=e['sentence'], label=e['sent_label']) for e in json_data]
            
        return examples


    def get_examples(self, mode: str):

        data_path = os.path.join(self.config.data_dir, mode, self.config.file_name)
        data = self.read_json(data_path)
        
        return self.to_examples(json_data=data)
    
class VIMqDataset:
    def __init__(self, config: VIMqConfig, tokenizer: AutoTokenizer) -> None:
        self.config = config
        self.processor = Processor(config=config)
        self.tokenizer = tokenizer
    
    def get_dataset(self, mode: str):
        cached_features_file = os.path.join(
            self.config.data_dir,
            'cached_{}_{}_{}'.format(
                mode,
                list(filter(None, self.config.pretrained_model_name_or_path.split("/"))).pop(),
                self.config.max_seq_len
            ))
        
        if os.path.exists(cached_features_file):
            logger.info(f'Load cached features from: {cached_features_file}')
            return Dataset.load_from_disk(cached_features_file)
        else:

            examples: list[dict[str, str]] = [e.to_dict() for e in self.processor.get_examples(mode=mode)]
            dataset = Dataset.from_list(examples)
            
            dataset =  dataset.map(
                function=self.preprocess_function,
                remove_columns=['sentence'],
                batched=True,
                num_proc=self.config.num_proc
            )
            dataset.save_to_disk(cached_features_file)
            
            return dataset
        
    def preprocess_function(self, examples):
        inputs = self.tokenizer(examples['sentence'], self.config.max_length, truncation=True, padding=True)
        inputs['label'] = [self.config.label2id(e) for e in examples['label']]
        
        return inputs
