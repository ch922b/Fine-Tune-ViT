import config
from data import get_dataset, get_processor, transform_fn, apply_transform, collate_fn
from model import get_model
from trainer_setup import get_training_args, get_metrics, get_trainer
from train_runner import run_training
from eval_runner import run_evaluation

def main():
    # 데이터 및 전처리
    dataset = get_dataset()
    processor = get_processor(config.MODEL_NAME)
    transform = transform_fn(processor)
    prepared_ds = apply_transform(dataset, transform)

    # 모델 생성
    label_names = dataset['train'].features['labels'].names
    model = get_model(config.MODEL_NAME, label_names)

    # Trainer 구성
    training_args = get_training_args()
    compute_metrics = get_metrics()
    trainer = get_trainer(
        model=model,
        args=training_args,
        collate_fn=collate_fn,
        compute_metrics=compute_metrics,
        processor=processor,
        train_ds=prepared_ds['train'],
        eval_ds=prepared_ds['validation'],
    )

    # 훈련 및 평가
    run_training(trainer)
    run_evaluation(trainer, prepared_ds['validation'])

if __name__ == "__main__":
    main()