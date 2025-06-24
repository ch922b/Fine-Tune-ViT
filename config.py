# config.py

# 모델 및 데이터 경로
MODEL_NAME = 'google/vit-base-patch16-224-in21k'
OUTPUT_DIR = './vit-base-beans'

# 학습 설정
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 2e-4

# 평가/저장 설정
EVAL_STEPS = 100
SAVE_STEPS = 100
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 2

# 기타 설정
USE_FP16 = True
REMOVE_UNUSED_COLUMNS = False
PUSH_TO_HUB = False
REPORT_TO = 'tensorboard'
LOAD_BEST_MODEL_AT_END = True
