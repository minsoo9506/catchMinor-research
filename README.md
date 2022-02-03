# catchMinor

- Research and Code Implementation
    - Imbalance Learning
    - Anomaly Detection, Outlier Analysis
- To do
    - `Basemodel.py` 만들어서 기준 만들기
    - 기존 레포 합치기
    - git action
        - test 실행

# Index
1. [Code Implementation](#code-implementation)
2. [Applied Project](#applied-project)
3. [Paper Read](#paper-read)

## Code Implementation

## Applied Project
- Dacon 신용카드 사용자 연체 예측 AI 경진대회
  - tabular, multiple classes classification(3 classes), imbalance, logloss
  - practice
    - OVO + Oversampling
    - Predict Probability Calibration
    - MetaCost
- Kaggle Credit Card Fraud Detection
  - tabular, binary classification, imbalance
  - practice
    - SMOTE
    - Unsupervised PCA based algorithm
- 네트워크임베딩 대학원수업 기말 프로젝트
  - Anomaly Detection with Graph Embedding Ensemble
    - (small size) tabular data
    - Node2Vec, PCA, Mahalanobis, LOF, Random Forest

## Paper Read
### Imbalanced Learning
#### Survey
- Learning From Imbalanced Data: open challenges and future directions (survey article 2016)
  - [`Paper Link`](https://link.springer.com/article/10.1007/s13748-016-0094-0) | `My Summary` | `My Code`

#### Perfomance Measure
- The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets
  - [`Paper Link`](https://pubmed.ncbi.nlm.nih.gov/25738806/) | `My Summary` | `My Code`
- The Relationship Between Precision-Recall and ROC Curves
  - [`Paper Link`](https://www.biostat.wisc.edu/~page/rocpr.pdf) | `My Summary` | `My Code`
- Predicting Good Probabilities With Supervised Learning
  - [`Paper Link`](https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf) | `My Summary` | `My Code`
- Properties and benefits of calibrated classifiers
  - [`Paper Link`](http://www.ifp.illinois.edu/~iracohen/publications/CalibrationECML2004.pdf) | `My Summary` | `My Code`
- The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets
  - [`Paper Link`](https://www.researchgate.net/publication/273155496_The_Precision-Recall_Plot_Is_More_Informative_than_the_ROC_Plot_When_Evaluating_Binary_Classifiers_on_Imbalanced_Datasets) | `My Summary` | `My Code`
- The relationship between precision-recall and ROC curves
  - [`Paper Link`](https://www.biostat.wisc.edu/~page/rocpr.pdf) | `My Summary` | `My Code`
  
#### Cost-sensitive
- An optimized cost-sensitive SVM for imbalanced data learning
  - [`Paper Link`](https://webdocs.cs.ualberta.ca/~zaiane/postscript/pakdd13-1.pdf) | `My Summary` | `My Code`
- Metacost : a general method for making classifiers cost-sensitive (KDD 99)
  - [`Paper Link`](https://homes.cs.washington.edu/~pedrod/papers/kdd99.pdf) | `My Summary` | `My Code`
- The influence of class imbalance on cost-sensitive learning (IEEE 2006)
  - [`Paper Link`](https://ieeexplore.ieee.org/document/4053137) | `My Summary` | `My Code`

#### Over, Under Sampling
- SMOTE (2002)
  - [`Paper Link`](https://arxiv.org/pdf/1106.1813.pdf) | `My Summary` | `My Code`
- SMOTE for learning from imbalanced data : progress and challenges (2018)
  - [`Paper Link`](https://www.jair.org/index.php/jair/article/view/11192) | `My Summary` | `My Code`
- Influence of minority class instance types on SMOTE imbalanced data oversampling
  - [`Paper Link`](https://www.researchgate.net/publication/320625181_Influence_of_minority_class_instance_types_on_SMOTE_imbalanced_data_oversampling) | `My Summary` | `My Code`

#### Ensemble Learning
- Self-paced Ensemble for Highly Imbalanced Massive Data Classification (2020)
  - [`Paper Link`](https://arxiv.org/abs/1909.03500) | `My Summary` | `My Code`

#### Imbalanced Classification with Multiple Classes
- Imbalanced Classification with Multiple Classes
  - Decomposition-Based Approaches
  - Ad-hoc Approaches

### Anomaly Detection, Outlier Analysis
#### Outlier Analysis (2017) - Charu C. Aggarwal
- Chapter02 Probabilistic and Statistical Models for Outlier Detection
- Chapter03 Linear Models for Outlier Detection
    - Linear Regression, PCA, OCSVM
- Chapter04 Proximity-Based Outlier Detection
    - Distance-Based
    - Density-Based (LOF, LOCI, Histogram, Kernel Density)
- Chapter05 High-Dimensional Outlier Detection
    - Axis-Parallel subsapce
    - Generalized subspace
- Chapter06 Outlier Ensembles
    - Variance reduction
    - Bias reduction
- Chapter07 Supervised Outlier Detection
    - Cost-Sentitive (MetaCost, Weighting Method)
    - Adaptive Re-sampling (SMOTE)
    - Boosting
    - Semi-Supervision
    - Supervised Models for Unsupervised Outlier Detection
- Chapter08 Outlier Detection in Categorical, Text, and Mixed Attributed Data
- Chapter09 Time Series and Streaming Outlier Detection
    - Prediction-based Anomaly Detection
        - Univariate aase (ARIMA)
        - Multiple Time Series
        - selection method
        - PCA method
- ...

#### Categoriztion of Deep Anomaly Detection
- Deep learning for feature extraction
- Learning feature representations of normality
  - Generic normality feature learning
    - AutoEncoder, GAN, Predictability Modeling, Self-Supervised classification
  - Anomaly measure-dependent feature learning
    - Distance-based classification, One-class classification measure, Clustering-based measure
- End-to-end anomaly score learning
  - Ranking model, Prior-driven model, Softmax likelihood model, End-to-end one-class classification

#### Survey
- Deep Learning for Anomaly Detection A Review (2020)
  - [`Paper Link`](https://arxiv.org/pdf/2007.02500.pdf) | [`My Summary`](./My%20summary) | `My Code`
- Autoencoders (2020)
  - [`Paper Link`](https://arxiv.org/pdf/2003.05991.pdf) | `My Summary` | `My Code`

#### Learning feature representations of normality
- Outlier Detection with AutoEncoder Ensemble (2017)
  - [`Paper Link`](https://saketsathe.net/downloads/autoencoder.pdf) | `My Summary` | `My Code`
- Auto-Encoding Variational Bayes (2014)
  - [`Paper Link`](https://arxiv.org/abs/1312.6114) | [`My Summary`](https://minsoo9506.github.io/blog/VAE/) | [`My Code`](./My%20code)
- Deep Variational Information Bottleneck (ICLR 2017)
  - [`Paper Link`](https://arxiv.org/abs/1612.00410) | [`My Summary`](https://minsoo9506.github.io/blog/IB/) | `My Code`
- Extracting and Composing Robust Features with Denoising Autoencoders (2008)
  - [`Paper Link`](https://www.cs.toronto.edu/~larocheh/publications/icml-2008-denoising-autoencoders.pdf) | `My Summary` | `My Code`
- Generatice Adversarial Nets (NIPS 2014)
  - [`Paper Link`](https://papers.nips.cc/paper/2014/hash/5ca3e9b122f61f8f06494c97b1afccf3-Abstract.html) | [`My Summary`](https://minsoo9506.github.io/blog/GAN/) | [`My Code`](./My%20code) 
- Least Squares Generative Adversarial Networks (2016)
  - [`Paper Link`](https://arxiv.org/abs/1611.04076) | [`My Summary`](https://minsoo9506.github.io/blog/LSGAN/) | [`My Code`](./My%20code) 
- Adversarial Autoencoders (2016)
  - [`Paper Link`](https://arxiv.org/abs/1511.05644) | [`My Summary`](./My%20summary) | `My Code`
- Generative Probabilistic Novelty Detection with Adversarial Autoencoders (NIPS 2018)
  - [`Paper Link`](https://papers.nips.cc/paper/2018/file/5421e013565f7f1afa0cfe8ad87a99ab-Paper.pdf) | `My Summary`| `My Code`
- Deep Autoencoding Gaussian Mixture Model For Unsupervised Anomaly Detection (ICLR 2018)
  - [`Paper Link`](https://sites.cs.ucsb.edu/~bzong/doc/iclr18-dagmm.pdf) | [`My Summary`](./My%20summary) | `My Code`
- Anomaly Detection with Robust Deep Autoencoders (KDD 2017)
  - [`Paper Link`](https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p665.pdf) | `My Summary` | `My Code`

#### Time Series and Streaming Anomaly Detection
- Anomaly Detection In Univariate Time-Series : A Survey on the state-of-the-art
  - [`Paper Link`](https://arxiv.org/abs/2004.00433) | `My Summary` | `My Code`
- USAD : UnSupervised Anomaly Detection on multivariate time series (KDD2020)
  - [`Paper Link`](https://dl.acm.org/doi/10.1145/3394486.3403392) | [`My Summary`](./My%20summary) | `My Code`
- Variational Attention for Sequence-to-Sequence Models (2017)
  - [`Paper Link`](https://arxiv.org/abs/1712.08207) | `My Summary` | `My Code`
- A Multimodal Anomaly Detector for Robot-Assisted Feeding Using an LSTM-based Variational Autoencoder (2017)
  - [`Paper Link`](https://arxiv.org/abs/1711.00614) | `My Summary` | `My Code`
- Outlier Detection for Time Series with Recurrent Autoencoder Ensembles (2019)
  - [`Paper Link`](https://www.ijcai.org/proceedings/2019/0378.pdf) | `My Summary` | `My Code`
- Robust Anomaly Detection for Multivariate time series through Stochastic Recurrent Neural Network (KKD 2019)
  - [`Paper Link`](https://github.com/NetManAIOps/OmniAnomaly) | `My Summary` | `My Code`
- Time Series Anomaly Detection with Multiresolution Ensemble Decoding (AAAI 2021)
  - [`Paper Link`](https://ojs.aaai.org/index.php/AAAI/article/view/17152) | `My Summary` | `My Code`
- An Improved Arima-Based Traffic Anomaly Detection Algorithm for Wireless Sensor Networks (2016)
  - [`Paper Link`](https://journals.sagepub.com/doi/pdf/10.1155/2016/9653230) | `My Summary` | `My Code`
- Time-Series Anomaly Detection Service at Microsoft (2019)
  - [`Paper Link`](https://arxiv.org/abs/1906.03821) | `My Summary` | `My Code`

## Resource

## Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── test           <- unit test
    │   │   └── test.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
