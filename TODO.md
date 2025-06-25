# TODO

## 새로 구현해야 할 것
- Inference 구현: 하나의 이미지를 입력받아 예측되는 label를 출력
- Inference 의 성능 테스트: 기존 데이터셋의 test 부분을 활용 혹은 인터넷에서 (이미지, 라벨) 쌍 수집
- Gradio를 통한 웹 UI 구현 -> 이미지 업로드, 예측 결과 확인


## 개선해야 할 기능
- 모듈의 Trainer_setup과 Train_runner 모듈의 통합 -> Train.py


## 발견된 문제와 해결방향
### inference 구현문제
현재 정확도 평가에 사용되는 Trainer.evaluate은 metrics에 평가에 들어간 전체 데이터셋 결과의 요약만을 표시함. (전체 Loss, Accuracy 등)
따라서 Logit을 반환해주는 Trainer.predict가 더 적합할 것으로 생각됨. 단, predict가 입력으로 Dataset만 받는 문제가 있음.
이 문제는 다음 2가지 경우를 시도할 수 있음.
- process 된 이미지 텐서를 입력으로 받을 수 있는 다른 함수 탐색 혹은 구현
#### 
- 기존의 이미지를 단 한개의 항목을 가지는 데이터셋으로 변환하여 기존에 predict에 입력
