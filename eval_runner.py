def run_evaluation(trainer, eval_dataset):
    metrics = trainer.evaluate(eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)