from datasets import load_dataset, DatasetDict
from glob import glob
import random
from transformers import BertTokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def tokenize(element):
    # return_overflowing_tokens 设置为 True， 可以将 超出 context_length的语句编为另一句
    outputs = tokenizer(element["content"],truncation=True,max_length=context_length,return_overflowing_tokens=True,return_length=True,)
    input_batch = []

    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

if __name__ == "__main__":
    random.seed(1010)

    all_file_list = glob(pathname="gpt2_data/wiki/**")[:500000]
    test_file_list = random.sample(all_file_list, 10)
    train_file_list = [i for i in all_file_list if i not in test_file_list]

    raw_datasets = load_dataset("csv", data_files={'train': train_file_list, 'valid': test_file_list},cache_dir="cache_data")

    context_length = 128
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    tokenizer.add_special_tokens({'bos_token': '[begin]','eos_token': '[end]','pad_token': '[PAD]'})
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenized_datasets = raw_datasets.map(tokenize, batched=True, remove_columns=raw_datasets["train"].column_names)

    config = GPT2Config.from_pretrained("gpt2",vocab_size=len(tokenizer),n_ctx=context_length,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
        pad_token = tokenizer.pad_token
    )

    model = GPT2LMHeadModel(config)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT-2 size: {model_size / 1000 ** 2:.1f}M parameters")

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
    for key in out:
        print(f"{key} shape: {out[key].shape}")

    args = TrainingArguments(
        output_dir="chinese_gpt2_big",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        eval_steps=2000,
        logging_steps=2000,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=300,
        fp16=True,
        push_to_hub=False,
    )
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )
    trainer.train()
