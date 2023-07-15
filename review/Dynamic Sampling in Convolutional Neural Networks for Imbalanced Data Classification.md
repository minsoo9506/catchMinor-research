![figure2](./figures/Dynamic%20Sampling%20in%20Convolutional%20Neural%20Networks%20for%20Imbalanced%20Data%20Classification%20Figure2.png)
- 3가지 module로 이루어져 있음
    - real-time data augmentation
    - transfer learning
    - dynamic sampling

### real-time data augmentation

### transfer learning in cnn
- imagenet task를 network camera classification으로 transfer learning
- top layer만 학습 나머지는 freeze
- early layer는 generic feature를 만든다는 것을 이용하기 위해

### dynamic sampling in cnn
- reference dataset을 이용하여 performance metric으로 training dataset의 분포를 바꿔간다.
- F1 score 사용 (one vs all)
- iteration 마다 reference dataset으로 각 class의 F1 score를 계산하고 낮은 f1 score class를 더 많이 model에 train dataset으로 넣어준다.