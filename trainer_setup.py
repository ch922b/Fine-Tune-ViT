from transformers import TrainingArguments, Trainer
from evaluate import load
import numpy as np
import config

def get_training_args():
    return TrainingArguments(
        output_dir=config.OUTPUT_DIR,
        per_device_train_batch_size=config.BATCH_SIZE,
        eval_strategy="steps",
        num_train_epochs=config.NUM_EPOCHS,
        fp16=config.USE_FP16,
        save_steps=config.SAVE_STEPS,
        eval_steps=config.EVAL_STEPS,
        logging_steps=config.LOGGING_STEPS,
        learning_rate=config.LEARNING_RATE,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        remove_unused_columns=config.REMOVE_UNUSED_COLUMNS,
        push_to_hub=config.PUSH_TO_HUB,
        report_to=config.REPORT_TO,
        load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
    )

def get_metrics():
    metric = load("accuracy")
    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1),
            references=p.label_ids
        )
    return compute_metrics

def get_trainer(model, args, collate_fn, compute_metrics, processor, train_ds, eval_ds):
    return Trainer(
        model=model,
        args=args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=processor,
    )
