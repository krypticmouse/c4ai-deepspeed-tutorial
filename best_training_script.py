import evaluate

import numpy as np
from loguru import logger
from utils import find_quantiles
from datasets import load_dataset

from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer


rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) 
            for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def train(
    checkpoint: str,
    enable_deepseed: bool = False,
    fairscale_sharded_ddp_strategy: str = ''
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    
    logger.info("Starting...")
    dataset = load_dataset("samsum")

    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    logger.info("Finding quantiles")

    dialogue_quantiles = find_quantiles(train_dataset['dialogue'], tokenizer)
    summary_quantiles = find_quantiles(train_dataset['summary'], tokenizer)

    logger.debug(f"Dialogue quantiles: {dialogue_quantiles}")
    logger.debug(f"Summary quantiles: {summary_quantiles}")

    logger.info("Loading Model...")
    
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    logger.info("Model Loaded Successfully...")

    training_args = Seq2SeqTrainingArguments(
        output_dir="best_summarization_model",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        logging_dir="logs",
        logging_strategy="steps",
        logging_steps=500,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=4,
        deepspeed=enable_deepseed,
        sharded_ddp=fairscale_sharded_ddp_strategy,
        predict_with_generate=True,
        fp16=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.debug("Starting Training Job...")

    trainer.train()
    tokenizer.save_pretrained("best_summarization_model")

    logger.debug("Model Trained Successfully.")

if __name__ == "__main__":
    import fire

    fire.Fire(train)