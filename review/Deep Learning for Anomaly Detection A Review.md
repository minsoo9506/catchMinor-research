# Problem complexities and challenges
- Major problem complexities
  - Unknownness
  - Heterogeneous anomaly classes
  - Rarity and class imbalance
  - Diverse types of anomaly
    - point anomalies, conditional anomalies, group anomalies
- Main challenges tackled by deep anomaly detection
  - low anomaly detection recall rate
  - anomaly detection in high-dimensional and/or not-independent data
  - Data-efficient learning of normality/abnormality
    - unsupervised, semi-supervised, weakly-supervised
  - Noise-resilient anomaly detection
  - Detection of complex anomalies
  - Anomaly explanation

# Addressing the challenges with deep anomaly detection
- Categoriztion of deep anomaly detection
  - Deep learning for feature extraction
  - Learning feature represetations of normality
    - Generic normality feature learning
      - Autoencoder
      - Gan
      - Predictability modeling
      - Self-supervised classification
    - Anomaly measure-dependent feature learning
      - Distance-based classification
      - One-class classification measure
      - Clustering-based measure
  - End-to-end anomaly score learning
    - Ranking model
    - Prior-driven model
    - Softmax likelihood model
    - End-to-end one-class classification

# Deep learning for feature extraction
- extract low-dim feature representations from high-dim and/or non-linearly separable data
- feature extraction과 anomaly scoring은 독립적으로 이루어진다.
- deep learning은 feature extraction에서만 사용, anomaly scoring에서의 알고리즘은 아무거나 상관없다.
- 가정
  -  extracted feature는 normal과 abnormal을 구분할 수 있는 정보 보존
- 종류
  - pre-trained model 이용하여 fine-tuning
  - autoencoder같이 feature extraction하며 train하는 모델
- 장점
  - pre-trained model 사용가능
  - 일반적인 linear method보다 차원축소에 효과적
  - 쉽다
- 단점
  - feature extraction과 anomaly scoring은 독립적으로 이루어져서 suboptimal가능
  - pre-trained model이 해당하는 영역에서만 사용가능
- challenges
  - 축소한다고 해서 무조건 정보를 잘 보존 한다는 보장이 없다

# Learning feature representation of normality
## Generic normality feature learning
- learns the representations of data instances by optimizing a generic feature learning objective function that is not primariliy designed for anomaly detection but empower the anomaly detection
- 종류
  - Autoencoders
    - sparse AE, Denoising AE, Contractive AE, Variational AE, Replicator neural network, RandNet, RDA, graph data, sequence data....
    - 장점
      - 다양한 data에 사용가능
    - 단점
      - feature representation의 이 완벽한게 아님
    - challenges
      - overfitting
  - GAN
    - 종류
      - AnoGAN, EBGAN, BiGAN, ALAD, GANomaly
    - 장점
      - 활용가능성이 높은 편
      - image쪽에서의 강점
    - 단점
      - train의 어려움
      - true dist가 복잡하거나 outlier가 있는 경우 취약한 편
  - Predictability modeling
    - 이전 시점의 data representation을 이용하여 현재 시점의 data 예측
    - 가정 : normal instance는 anomaly에 비해 temporally more predictable
    - sequential, spatial한 곳에서 많이 사용
    - U-net이용, AR 이용
    - 장점
      - 다양한 sequence learning 방법론 응용
      - temporal and spatial dependency를 배울 수 있다
    - 단점
      - sequence data에서만 사용
      - sequential prediction이라 computationally expensive
      - 원래 anomaly가 아닌 sequential prediction 모델들을 이용하는거라 suboptimal가능
  - self-supervised classification
    - learns representations of normality by building self-supervised classification models and identifies instances that are inconsistent to the classification models as anomalies.
## Anomaly measure-dependent feature learning
- anomaly measure를 optimize하여 learning feature representation
- 종류
  - Distance-based measure
    - 장점
      - deep learning 이전의 work들이 많다
      - low-dim에서 work하기에 distance를 잘 이용가능
    - 단점
      - computationally expensive
      - distance기반 방법론들의 단점
  - One-class classfication-based measure
    - 가정 : 하나의 class에서 만들어진 normal instance들은 compact model로 설명할 수 있다
    - one-class SVM에 deep 추가
    - 장점
      - deep learning 이전 연구가 많다
      - kernel을 고를 필요가 없다
      - representation learning과 one-class classification model이 잘 어우러진다
    - 단점
      - normal class에서 complex dist를 갖고 있으면 어렵다
      - measure에 따라 performance가 달라진다
    - Clustering-based measure
## End-to-end anomaly score learning
- incorporate order or discriminative information into the anomaly scoring network
- 종류
  - Ranking model
  - Prior-driven model
  - Softmax likelihood model
  - End-to-End one-class classification