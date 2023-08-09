import os
import datasets
import argparse
import multiprocessing
from itertools import chain
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from config.config import TrainerConf, TokenizerConf, get_logger


logger = get_logger()

def setup(conf: TrainerConf):
    def make_dir(path: str):
        if not os.path.exists(path=path):
            os.system(f'mkdir -p {path}')
            os.system(f'chmod 777 {path}')
    for d in [conf.output_dir, conf.logging_dir]:
        make_dir(d)

def main(model_type: str):    
    t_conf = TokenizerConf()
    conf = TrainerConf(sec=model_type)
    
    logger.info('Setting up')
    setup(conf=conf)
    
    logger.info('Loading dataset')
    data = datasets.load_from_disk(t_conf.dataset_dir)
    train_data = data['train']
    eval_data = data['val']
    # test_data = data['test']
    
    logger.info(f'Loading tokenizer: {conf.pretrained_model_name}')
    tokenizer = AutoTokenizer.from_pretrained(conf.pretrained_model_name)
    tokenizer.model_max_length = 512
    num_proc = multiprocessing.cpu_count()
    logger.info(f"The max length for the tokenizer is: {tokenizer.model_max_length}")
    
    def tokenize_text(examples): return tokenizer(
        examples[t_conf.column],
        return_special_tokens_mask=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= tokenizer.model_max_length:
            total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    
    train_tokenized_datasets = train_data.map(
        tokenize_text,
        batched=True,
        remove_columns=[t_conf.column]
    )
    
    train_tokenized_datasets = train_tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
    train_dataset = train_tokenized_datasets.shuffle(seed=conf.data_seed)
    # train_dataset.save_to_disk("/home/int2-user/vipubmedbert/data/tokenized_data/train")
    logger.info(f"The dataset contains in total {len(train_tokenized_datasets)*tokenizer.model_max_length} tokens")

    eval_tokenized_datasets = eval_data.map(
        tokenize_text,
        batch_size=True,
        remove_columns=[t_conf.column]
    )
    eval_dataset = eval_tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
    # eval_dataset.save_to_disk('/home/int2-user/vipubmedbert/data/tokenized_data/val')
    
    # test_tokenized_datasets = test_data.map(
    #     tokenize_text,
    #     batched=True,
    #     remove_columns=[t_conf.column]
    # )
    # test_dataset = test_tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc)
    # test_dataset.save_to_disk('/home/int2-user/vipubmedbert/data/tokenized_data/test')
    
    logger.info(f'Load model checkpoint: {conf.pretrained_model_name}')
    model = AutoModelForMaskedLM.from_pretrained('/home/int2-user/vipubmedbert/checkpoint/vipubmed-deberta/base/checkpoint-10000')
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=conf.mlm_probability,
        pad_to_multiple_of=4
    )

    training_args = TrainingArguments(
        run_name=conf.run_name,
        data_seed=conf.data_seed,
        dataloader_num_workers=conf.dataloader_num_workers,
        do_train=conf.do_train,
        do_eval=conf.do_eval,
        evaluation_strategy=conf.evaluation_strategy,
        eval_steps=conf.eval_steps,
        fp16=conf.fp16,
        gradient_accumulation_steps=conf.gradient_accumulation_steps,
        logging_dir=conf.logging_dir,
        learning_rate=conf.learning_rate,
        log_level=conf.log_level,
        logging_strategy=conf.logging_strategy,
        logging_steps=conf.logging_steps,
        lr_scheduler_type=conf.lr_scheduler_type,
        output_dir=conf.output_dir,
        max_steps=conf.max_steps,
        per_device_train_batch_size=conf.per_device_train_batch_size,
        per_device_eval_batch_size=conf.per_device_eval_batch_size,
        prediction_loss_only=conf.prediction_loss_only,
        report_to=conf.report_to,
        save_strategy=conf.save_strategy,
        save_steps=conf.save_steps,
        save_total_limit=conf.save_total_limit,
        seed=conf.seed,
        warmup_ratio=conf.warmup_ratio,
        warmup_steps=conf.warmup_steps,
        weight_decay=conf.weight_decay
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    logger.info('Training')
    trainer.train()
    
if __name__=='__main__':
    import wandb
    wandb.login(key='6620c8cd235027492af4387f5784e279e0303bd3')
    
    parser = argparse.ArgumentParser('ViPubMed-DeBERTa trainer')
    parser.add_argument('--model_type', type=str, default='xsmall')
    args = parser.parse_args()
    
    main(args.model_type)