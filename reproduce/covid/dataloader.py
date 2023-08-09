import os
import json
import copy
from loguru import logger
from pyvi import ViTokenizer
from datasets import Dataset
from transformers import AutoTokenizer
from utils.utils import path_exists
from config import CovIdConfig


class InputExample(object):
    def __init__(
        self,
        guid: int|str = None,
        words: list[str] = None,
        tags: list[int] = None
    ) -> None:
        super().__init__()
        
        self.guid = guid
        self.words = words
        self.tags = tags

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

    def __init__(self, config: CovIdConfig) -> None:
        super().__init__()
        self.config = config

    @path_exists
    def read_json(self, path: str):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            raise RuntimeError(f'Error when reading: {path}')
    
    def re_align(e: dict[list[str], list[str]]):
        pass
    
    def to_examples(self, json_data: list[dict[str, str|list[tuple[int, int, str]]]]) -> list[InputExample]:
        
        if self.config.re_tokenize:
            # todo
            examples = [InputExample(sentence=self.re_tokenize(e['sentence']), label=e['sent_label']) for e in json_data]
        else:
            examples = [InputExample(words=e['words'], tags=e['tags']) for e in json_data]
            
        return examples


    def get_examples(self, mode: str):

        data_path = os.path.join(self.config.data_dir, mode, self.config.file_name)
        data = self.read_json(data_path)
        
        return self.to_examples(json_data=data)
    
class CovIdDataset:
    def __init__(self, config: CovIdConfig, tokenizer: AutoTokenizer) -> None:
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
    
    # def prepare_dataset():
    #     def bio2id(example):
    #         example['ner_tags'] = [label2id[i] for i in example['ner_tags']]
    #         return example
        
    #     train = read_json(config.train_path)
    #     dev = read_json(config.dev_path)
    #     test = read_json(config.test_path)

    #     train_set = Dataset.from_list(train).map(lambda x: bio2id(x))
    #     dev_set = Dataset.from_list(dev).map(lambda x: bio2id(x))
    #     test_set = Dataset.from_list(test).map(lambda x: bio2id(x))
        
    #     return train_set, dev_set, test_set

    # train_set, dev_set, test_set = prepare_dataset()


    # def tokenize_and_align_labels(examples):
        
    #     ignore_idx = -100
    #     features = {
    #         'input_ids': [],
    #         'attention_mask': [],
    #         'token_type_ids': [],
    #         'labels': []
    #     }
        
    #     for words, tags in zip(examples['words'], examples['ner_tags']):
    #         tokens = []
    #         slot_label_ids = []
            
    #         for word, tag in zip(words, tags):
    #             word_tokens = tokenizer.tokenize(word)
    #             if word_tokens:
    #                 tokens.extend(word_tokens)
    #             else:
    #                 tokens.extend(tokenizer.unk_token)
    #             slot_label_ids.extend([tag] + [ignore_idx]*(len(word_tokens)-1))
            
    #         tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    #         input_ids = tokenizer.convert_tokens_to_ids(tokens)
            
    #         features['input_ids'].append(input_ids)
    #         features['attention_mask'].append([1]*len(input_ids))
    #         features['token_type_ids'].append([0]*len(input_ids))
    #         features['labels'].append([ignore_idx] + slot_label_ids + [ignore_idx])
            
        
    #     return features
