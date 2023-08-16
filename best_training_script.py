import rouge
import numpy as np
from loguru import logger
from utils import find_quantiles, preprocess

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

checkpoint = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


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


def train():
    dataset = load_dataset("samsum")

    train_dataset = dataset['train']
    val_dataset = dataset['validation']

    logger.info("Finding quantiles")

    dialogue_quantiles = find_quantiles(train_dataset['dialogue'])
    summary_quantiles = find_quantiles(train_dataset['summary'])

    logger.debug(f"Dialogue quantiles: {dialogue_quantiles}")
    logger.debug(f"Summary quantiles: {summary_quantiles}")

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

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

    trainer.train()