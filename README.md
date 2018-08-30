# awesome-anomaly-detection[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
A curated list of awesome anomaly detection resources. Inspired by [`awesome-architecture-search`](https://github.com/sdukshis/awesome-ml) and [`awesome-automl`](https://github.com/hibayesian/awesome-automl-papers).  
*Last updated: 2018/8/29*

## What is anomaly detection?

<p align="center">
  <img width="600" src="/assets/anomaly_detection_example1.PNG" "Example of anomaly detection.">
</p>

Anomaly detection is a technique used to identify unusual patterns that do not conform to expected behavior, called outliers. Typically, this is treated as an unsupervised learning problem where the anomalous samples are not known a priori and it is assumed that the majority of the training dataset consists of “normal” data (here and elsewhere the term “normal” means *not anomalous* and is unrelated to the Gaussian distribution). [Lukas Ruff et al., 2018; Deep One-Class Classification]

In general, Anomaly detection is also called `Novelty Detection` or `Outlier Detection`, `Forgery Detection` and `Out-of-distribution Detection`.   

Each term has slightly different meanings. Mostly, on the assumption that you do not have unusual data, this problem is especially called `One Class Classification`, `One Class Segmentation`.  

Also, there are two types of target data. (`time-series data`, and `image data`)  
In time-series data, it is aimed to detect a abnormal sections or frames in input data. (ex, videos, signal, etc)  
In image data, it is aimed to classify abnormal images or to segment abnormal regions, for example, defect in some manufacturing data.  



## Table of Contents
- [Time-series anomaly detection](#time-series-anomaly-detection)
- [Image-level anomaly detection](#image-level-anomaly-detection)
  - [Classification target](#classification-target)
  - [Segmenatation target](#segmenatation-target)

## Time-series anomaly detection
- Anomaly Detection of Time Series | Deepthi Cheboli | **[Thesis' 10] ** |[`[pdf]`](https://conservancy.umn.edu/bitstream/handle/11299/92985/Cheboli_Deepthi_May2010.pdf?sequence=1)
- Time Series Anomaly Detection; Detection of anomalous drops with limited features and sparse examples in noisy highly periodic data | Dominique T. Shipmon, et al. | **[Arxiv' 17]** |   [`[pdf]`](https://arxiv.org/ftp/arxiv/papers/1708/1708.03665.pdf)
- Abnormal Event Detection in Videos using Spatiotemporal Autoencoder | Yong Shean Chong, et al. | **[ISNN' 17]** | [`[pdf]`](https://arxiv.org/pdf/1701.01546.pdf)
- Real-world Anomaly Detection in Surveillance Videos | Waqas Sultani, et al. | **[Arxiv' 18]** | [`[pdf]`](http://crcv.ucf.edu/cchen/anomaly_detection.pdf) [`[project page]`](http://crcv.ucf.edu/cchen/)

## Image-level anomaly detection

### Classification target
- Estimating the Support of a High- Dimensional Distribution [**OC-SVM**] | Bernhard Schölkopf, et al. | **[Journal of Neural Computation' 01]** | [`[pdf]`](http://users.cecs.anu.edu.au/~williams/papers/P132.pdf)
- A Survey of Recent Trends in One Class Classification | Shehroz S, et al. | **[AICS' 09]** |  [`[pdf]`](https://aran.library.nuigalway.ie/xmlui/bitstream/handle/10379/1472/camera_ready_occ_lnai.pdf?sequence=1)
- Anomaly detection using au- toencoders with nonlinear dimensionality reduction | Mayu Sakurada, et al. | **[MLSDA Workshop' 14]** | [`[link]`](https://dl.acm.org/citation.cfm?id=2689747)
- A review of novelty detection | Marco A. F Pimentel et al. | **[Signal Processing' 14]** |  [`[link]`](https://www.sciencedirect.com/science/article/pii/S016516841300515X)
- High-dimensional and large-scale anomaly detection using a linear one-class SVM with deep learning | Sarah M. Erfani, et al. | **[Pattern Recognition' 16]** | [`[link]`](https://dl.acm.org/citation.cfm?id=2952200)
- Transfer Representation-Learning for Anomaly Detection | Jerone T. A. Andrews, et al. | **[ICML' 16]** | [`[pdf]`](https://pdfs.semanticscholar.org/c533/52a4239568cc915ad968aff51c49924a3072.pdf)
- Outlier Detection with Autoencoder Ensembles | Haowen Xu, et al. | **[SDM' 17]** | [`[pdf]`](https://arxiv.org/pdf/1802.03903.pdf)
- Provable self-representation based outlier detection in a union of subspaces | Chong You, et al. | **[CVPR' 17]** | [`[pdf]`](https://arxiv.org/pdf/1704.03925.pdf)
- Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks [**ODIN**] | Shiyu Liang, et al. | **[ICLR' 18]** | [`[pdf]`](https://arxiv.org/pdf/1706.02690.pdf)
- Learning Confidence for Out-of-Distribution Detection in Neural Networks | Terrance DeVries, et al. | **[Arxiv' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1802.04865.pdf)
- Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples | Kimin Lee, et al. | **[ICLR' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1711.09325.pdf)
- Adversarially Learned One-Class Classifier for Novelty Detection [**ALOCC**] | Mohammad Sabokrou, et al. | **[CVPR' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1802.09088.pdf) [`[code]`](https://github.com/khalooei/ALOCC-CVPR2018)
- Learning Deep Features for One-Class Classification | Pramuditha Perera, et al. | **[Arxiv' 18]** |   [`[pdf]`](https://arxiv.org/pdf/1801.05365.pdf) [`[code]`](https://github.com/PramuPerera/DeepOneClass)
- Efficient GAN-Based Anomaly Detection | Houssam Zenati, et al. | **[ICLR Workshop' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1802.06222.pdf)
- Hierarchical Novelty Detection for Visual Object Recognition | Kibok Lee, et al. | **[CVPR' 18]** | [`[pdf]`](https://arxiv.org/pdf/1804.00722.pdf)
- Deep One-Class Classification | Lukas Ruff, el al. | **[ICML' 18]** | [`[pdf]`](http://data.bit.uni-bonn.de/publications/ICML2018.pdf)

### Segmenatation target
- Anomaly Detection and Localization in Crowded Scenes | Weixin Li, et al. | **[TPAMI' 14]** | [`[pdf]`](http://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf)
- Real-Time Anomaly Detection and Localization in Crowded Scenes | Mohammad Sabokrou, et al. | **[CVPR Workshop' 15]** | [`[pdf]`](https://arxiv.org/pdf/1511.06936.pdf)
- Learning Deep Representations of Appearance and Motion for Anomalous Event Detection | Dan Xu, et al. | **[BMVC' 15]** | [`[pdf]`](https://arxiv.org/pdf/1510.01553.pdf)
- Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery  [**AnoGAN**] | Thomas Schlegl, et al. | **[IPMI' 17]** | [`[pdf]`](https://arxiv.org/pdf/1703.05921.pdf) 
- Deep-Anomaly: Fully Convolutional Neural Network for Fast Anomaly Detection in Crowded Scenes | Mohammad Sabokrou, et al. | **[Journal of Computer Vision and Image Understanding' 17]** | [`[pdf]`](https://arxiv.org/pdf/1609.00866.pdf)
- Anomaly Detection using a Convolutional Winner-Take-All Autoencoder | Hanh T. M. Tran, et al. | **[BMVC' 17]** |  [`[pdf]`](http://eprints.whiterose.ac.uk/121891/1/BMVC2017.pdf)
- Anomaly Detection in Nanofibrous Materials by CNN-Based Self-Similarity | Paolo Napoletano , et al. | **[Sensors' 17]** | [`[pdf]`](http://www.mdpi.com/1424-8220/18/1/209/pdf)
- Abnormal event detection in videos using generative adversarial nets | Mahdyar Ravanbakhsh, et al. |  **[ICIP' 17]** | [`[link]`](https://ieeexplore.ieee.org/document/8296547/)
- An overview of deep learning based methods for unsupervised and semi-supervised anomaly detection in videos |  B Ravi Kiran, et al. | **[Arxiv' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1801.03149.pdf)
- Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders | Paul Bergmann, et al. | **[Arxiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1807.02011.pdf)
- Satellite Image Forgery Detection and Localization Using GAN and One-Class Classifier | Sri Kalyan Yarlagadda, et al. | **[IS&T EI' 18]** | [`[pdf]`](https://arxiv.org/pdf/1802.04881.pdf)
- Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images | Christoph Baur, et al. | **[Arxiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1804.04488.pdf)
