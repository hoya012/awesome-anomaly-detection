# awesome anomaly detection
A curated list of awesome anomaly detection resources. Inspired by [`awesome-architecture-search`](https://github.com/sdukshis/awesome-ml) and [`awesome-automl`](https://github.com/hibayesian/awesome-automl-papers).  

*Last updated: 2020/05/07*

## What is anomaly detection?

<p align="center">
  <img width="600" src="/assets/anomaly_detection_example1.PNG" "Example of anomaly detection.">
</p>

Anomaly detection is a technique used to identify unusual patterns that do not conform to expected behavior, called outliers. Typically, this is treated as an unsupervised learning problem where the anomalous samples are not known a priori and it is assumed that the majority of the training dataset consists of “normal” data (here and elsewhere the term “normal” means *not anomalous* and is unrelated to the Gaussian distribution). [Lukas Ruff et al., 2018; Deep One-Class Classification]

In general, Anomaly detection is also called `Novelty Detection` or `Outlier Detection`, `Forgery Detection` and `Out-of-distribution Detection`.   

Each term has slightly different meanings. Mostly, on the assumption that you do not have unusual data, this problem is especially called `One Class Classification`, `One Class Segmentation`.  

<p align="center">
  <img width="600" src="/assets/anomaly_detection_types.png" "Example of anomaly detection.">
</p>

and `Novelty Detection` and `Outlier Detection` have slightly different meanings. Figure below shows the differences of two terms.

Also, there are two types of target data. (`time-series data`, and `image data`)  
In time-series data, it is aimed to detect a abnormal sections or frames in input data. (ex, videos, signal, etc)  
In image data, it is aimed to classify abnormal images or to segment abnormal regions, for example, defect in some manufacturing data.  

## Survey Paper
- Deep Learning for Anomaly Detection: A Survey  | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.03407.pdf)
- Anomalous Instance Detection in Deep Learning: A Survey | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.06979.pdf)

## Table of Contents
- [Time-series anomaly detection](#time-series-anomaly-detection)
- [Image-level anomaly detection](#image-level-anomaly-detection)
  - [Anomaly Classification target](#anomaly-classification-target)
  - [Out-Of-Distribution(OOD) Detction target](#out-of-distributionood-detction-target)
  - [Anomaly Segmentation target](#anomaly-segmentation-target)

## Time-series anomaly detection **(need to survey more..)**
- Anomaly Detection of Time Series  | **[Thesis' 10]** |[`[pdf]`](https://conservancy.umn.edu/bitstream/handle/11299/92985/Cheboli_Deepthi_May2010.pdf?sequence=1)
- Long short term memory networks for anomaly detection in time series | **[ESANN' 15]** |[`[pdf]`](https://www.researchgate.net/publication/304782562_Long_Short_Term_Memory_Networks_for_Anomaly_Detection_in_Time_Series)
 - LSTM-Based System-Call Language Modeling and Robust Ensemble Method for Designing Host-Based Intrusion Detection Systems | **[arXiv' 16]** |   [`[pdf]`](https://arxiv.org/pdf/1611.01726.pdf)
- Time Series Anomaly Detection; Detection of anomalous drops with limited features and sparse examples in noisy highly periodic data | **[arXiv' 17]** |   [`[pdf]`](https://arxiv.org/ftp/arxiv/papers/1708/1708.03665.pdf)
- Abnormal Event Detection in Videos using Spatiotemporal Autoencoder | **[ISNN' 17]** | [`[pdf]`](https://arxiv.org/pdf/1701.01546.pdf)
- Anomaly Detection in Multivariate Non-stationary Time Series for Automatic DBMS Diagnosis | **[ICMLA' 17]** | [`[pdf]`](https://arxiv.org/abs/1708.02635)
- Real-world Anomaly Detection in Surveillance Videos | **[arXiv' 18]** | [`[pdf]`](http://crcv.ucf.edu/cchen/anomaly_detection.pdf) [`[project page]`](http://crcv.ucf.edu/cchen/)
- Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling | **[CVPR Workshop' 18]** | [`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Wei_Unsupervised_Anomaly_Detection_CVPR_2018_paper.pdf)
- Dual-Mode Vehicle Motion Pattern Learning for High Performance Road Traffic Anomaly Detection  | **[CVPR Workshop' 18]** | [`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Xu_Dual-Mode_Vehicle_Motion_CVPR_2018_paper.pdf)
- Truth Will Out: Departure-Based Process-Level Detection of Stealthy Attacks on Control Systems  | **[ACM CCS '18]** | [`[pdf]`](https://research.chalmers.se/publication/507989/file/507989_Fulltext.pdf)
- Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos  | **[CVPR' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1903.03295.pdf)
- Challenges in Time-Stamp Aware Anomaly Detection in Traffic Videos  | **[CVPRW' 19]** |  [`[pdf]`](https://arxiv.org/ftp/arxiv/papers/1906/1906.04574.pdf)
- Motion-Aware Feature for Improved Video Anomaly Detection | **[BMVC' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1907.10211v1.pdf)
- Time-Series Anomaly Detection Service at Microsoft  | **[KDD' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1906.03821v1.pdf)
- Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network  | **[KDD' 19]** |  [`[pdf]`](https://dl.acm.org/doi/10.1145/3292500.3330672)
- A Systematic Evaluation of Deep Anomaly Detection Methods for Time Series | **Under Review** | [`[code]`](https://github.com/KDD-OpenSource/DeepADoTS)
- BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time | **[IJCAI 19]** | [`[pdf]`](https://www.ijcai.org/Proceedings/2019/0616.pdf)

- MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams  | **[AAAI' 20]** |  [`[pdf]`](https://www.comp.nus.edu.sg/~sbhatia/assets/pdf/midas.pdf)  [`[code]`](https://github.com/bhatiasiddharth/MIDAS)

## Image-level anomaly detection

### One Class (Anomaly) Classification target
- Estimating the Support of a High- Dimensional Distribution [**OC-SVM**]  | **[Journal of Neural Computation' 01]** | [`[pdf]`](http://users.cecs.anu.edu.au/~williams/papers/P132.pdf)
- A Survey of Recent Trends in One Class Classification  | **[AICS' 09]** |  [`[pdf]`](https://aran.library.nuigalway.ie/xmlui/bitstream/handle/10379/1472/camera_ready_occ_lnai.pdf?sequence=1)
- Anomaly detection using autoencoders with nonlinear dimensionality reduction  | **[MLSDA Workshop' 14]** | [`[link]`](https://dl.acm.org/citation.cfm?id=2689747)
- A review of novelty detection | **[Signal Processing' 14]** |  [`[link]`](https://www.sciencedirect.com/science/article/pii/S016516841300515X)
- Variational Autoencoder based Anomaly Detection using Reconstruction Probability |  **[SNU DMC Tech' 15]** | [`[pdf]`](http://dm.snu.ac.kr/static/docs/TR/SNUDM-TR-2015-03.pdf)
- High-dimensional and large-scale anomaly detection using a linear one-class SVM with deep learning | **[Pattern Recognition' 16]** | [`[link]`](https://dl.acm.org/citation.cfm?id=2952200)
- Transfer Representation-Learning for Anomaly Detection | **[ICML' 16]** | [`[pdf]`](https://pdfs.semanticscholar.org/c533/52a4239568cc915ad968aff51c49924a3072.pdf)
- Outlier Detection with Autoencoder Ensembles  | **[SDM' 17]** | [`[pdf]`](http://saketsathe.net/downloads/autoencode.pdf)
- Provable self-representation based outlier detection in a union of subspaces | **[CVPR' 17]** | [`[pdf]`](https://arxiv.org/pdf/1704.03925.pdf)
- [**ALOCC**]Adversarially Learned One-Class Classifier for Novelty Detection  | **[CVPR' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1802.09088.pdf) [`[code]`](https://github.com/khalooei/ALOCC-CVPR2018)
- Learning Deep Features for One-Class Classification | **[arXiv' 18]** |   [`[pdf]`](https://arxiv.org/pdf/1801.05365.pdf) [`[code]`](https://github.com/PramuPerera/DeepOneClass)
- Efficient GAN-Based Anomaly Detection  | **[arXiv' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1802.06222.pdf)
- Hierarchical Novelty Detection for Visual Object Recognition  | **[CVPR' 18]** | [`[pdf]`](https://arxiv.org/pdf/1804.00722.pdf)
- Deep One-Class Classification | **[ICML' 18]** | [`[pdf]`](http://data.bit.uni-bonn.de/publications/ICML2018.pdf)
- Reliably Decoding Autoencoders’ Latent Spaces for One-Class Learning Image Inspection Scenarios | **[OAGM Workshop' 18]** | [`[pdf]`](https://workshops.aapr.at/wp-content/uploads/Proceedings/2018/OAGM_2018_paper_19.pdf)
- q-Space Novelty Detection with Variational Autoencoders  | **[arXiv' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1806.02997.pdf)
- GANomaly: Semi-Supervised Anomaly Detection via Adversarial Training | **[ACCV' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1805.06725.pdf)
- Deep Anomaly Detection Using Geometric Transformations  | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/8183-deep-anomaly-detection-using-geometric-transformations.pdf)
- Generative Probabilistic Novelty Detection with Adversarial Autoencoders | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/7915-generative-probabilistic-novelty-detection-with-adversarial-autoencoders.pdf)
- A loss framework for calibrated anomaly detection | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/7422-a-loss-framework-for-calibrated-anomaly-detection.pdf)
- A Practical Algorithm for Distributed Clustering and Outlier Detection | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/7493-a-practical-algorithm-for-distributed-clustering-and-outlier-detection.pdf)
- Efficient Anomaly Detection via Matrix Sketching  | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/8030-efficient-anomaly-detection-via-matrix-sketching.pdf)
- Adversarially Learned Anomaly Detection  | **[IEEE ICDM' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1812.02288.pdf)
- Anomaly Detection With Multiple-Hypotheses Predictions  | **[ICML' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1810.13292v5.pdf)
- Exploring Deep Anomaly Detection Methods Based on Capsule Net  | **[ICMLW' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1907.06312v1.pdf)
- Latent Space Autoregression for Novelty Detection | **[CVPR' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1807.01653.pdf)
- OCGAN: One-Class Novelty Detection Using GANs With Constrained Latent Representations | **[CVPR' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1903.08550.pdf)
- Unsupervised Learning of Anomaly Detection from Contaminated Image Data using Simultaneous Encoder Training | **[arXiv' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1905.11034.pdf)
- Using Self-Supervised Learning Can Improve Model Robustness and Uncertainty | **[NeurIPS' 19]** |  [`[pdf]`](https://arxiv.org/abs/1906.12340) [`[code]`](https://github.com/hendrycks/ss-ood)
- Classification-Based Anomaly Detection for General Data | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=H1lK_lBtvS)
- Robust Subspace Recovery Layer for Unsupervised Anomaly Detection   | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=rylb3eBtwr)
- RaPP: Novelty Detection with Reconstruction along Projection Pathway   | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=HkgeGeBYDB)
- Novelty Detection Via Blurring  | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=ByeNra4FDB)
- Deep Semi-Supervised Anomaly Detection   | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=HkgH0TEYwH)
- Robust anomaly detection and backdoor attack detection via differential privacy | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=SJx0q1rtvS)
- Classification-Based Anomaly Detection for General Data | **[ICLR' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2005.02359v1.pdf)

### Out-of-Distribution(OOD) Detction target
- A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks  | **[ICLR' 17]** | [`[pdf]`](https://arxiv.org/pdf/1610.02136.pdf)
- [**ODIN**] Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks  | **[ICLR' 18]** | [`[pdf]`](https://arxiv.org/pdf/1706.02690.pdf)
- Training Confidence-calibrated Classifiers for Detecting Out-of-Distribution Samples | **[ICLR' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1711.09325.pdf)
- Learning Confidence for Out-of-Distribution Detection in Neural Networks | **[arXiv' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1802.04865.pdf)
- Out-of-Distribution Detection using Multiple Semantic Label Representations | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/7967-out-of-distribution-detection-using-multiple-semantic-label-representations.pdf)
- A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks  | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/7947-a-simple-unified-framework-for-detecting-out-of-distribution-samples-and-adversarial-attacks.pdf)
- Deep Anomaly Detection with Outlier Exposure  | **[ICLR' 19]** |  [`[pdf]`](https://openreview.net/pdf?id=HyxCxhRcY7)
- Why ReLU networks yield high-confidence predictions far away from the training data and how to mitigate the problem  | **[CVPR' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1812.05720.pdf)
- Outlier Exposure with Confidence Control for Out-of-Distribution Detection | **[arXiv' 19]** |  [`[pdf]`](https://arxiv.org/abs/1906.03509v2) [`[code]`](https://github.com/nazim1021/OOD-detection-using-OECC)
- Likelihood Ratios for Out-of-Distribution Detection | **[NeurIPS' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1906.02845.pdf)
- Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=SyxIWpVYvr)


### One Class (Anomaly) Segmentation target
- Anomaly Detection and Localization in Crowded Scenes  | **[TPAMI' 14]** | [`[pdf]`](http://www.svcl.ucsd.edu/publications/journal/2013/pami.anomaly/pami_anomaly.pdf)
- Novelty detection in images by sparse representations  | **[IEEE Symposium on IES' 14]** | [`[link]`](https://ieeexplore.ieee.org/document/7008985/)
- Detecting anomalous structures by convolutional sparse models | **[IJCNN' 15]** | [`[pdf]`](http://www.cs.tut.fi/~foi/papers/IJCNN2015-Carrera-Detecting_Anomalous_Structures.pdf)
- Real-Time Anomaly Detection and Localization in Crowded Scenes | **[CVPR Workshop' 15]** | [`[pdf]`](https://arxiv.org/pdf/1511.06936.pdf)
- Learning Deep Representations of Appearance and Motion for Anomalous Event Detection  | **[BMVC' 15]** | [`[pdf]`](https://arxiv.org/pdf/1510.01553.pdf)
- Scale-invariant anomaly detection with multiscale group-sparse models | **[IEEE ICIP' 16]** | [`[link]`](https://ieeexplore.ieee.org/document/7533089/)
- [**AnoGAN**] Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery  | **[IPMI' 17]** | [`[pdf]`](https://arxiv.org/pdf/1703.05921.pdf) 
- Deep-Anomaly: Fully Convolutional Neural Network for Fast Anomaly Detection in Crowded Scenes | **[Journal of Computer Vision and Image Understanding' 17]** | [`[pdf]`](https://arxiv.org/pdf/1609.00866.pdf)
- Anomaly Detection using a Convolutional Winner-Take-All Autoencoder | **[BMVC' 17]** |  [`[pdf]`](http://eprints.whiterose.ac.uk/121891/1/BMVC2017.pdf)
- Anomaly Detection in Nanofibrous Materials by CNN-Based Self-Similarity  | **[Sensors' 17]** | [`[pdf]`](http://www.mdpi.com/1424-8220/18/1/209/pdf)
- Defect Detection in SEM Images of Nanofibrous Materials | **[IEEE Trans. on Industrial Informatics' 17]** | [`[pdf]`](http://home.deib.polimi.it/boracchi/docs/2017_Anomaly_Detection_SEM.pdf)
- Abnormal event detection in videos using generative adversarial nets  |  **[ICIP' 17]** | [`[link]`](https://ieeexplore.ieee.org/document/8296547/)
- An overview of deep learning based methods for unsupervised and semi-supervised anomaly detection in videos  | **[arXiv' 18]** |  [`[pdf]`](https://arxiv.org/pdf/1801.03149.pdf)
- Improving Unsupervised Defect Segmentation by Applying Structural Similarity to Autoencoders  | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1807.02011.pdf)
- Satellite Image Forgery Detection and Localization Using GAN and One-Class Classifier  | **[IS&T EI' 18]** | [`[pdf]`](https://arxiv.org/pdf/1802.04881.pdf)
- Deep Autoencoding Models for Unsupervised Anomaly Segmentation in Brain MR Images  | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1804.04488.pdf)
- AVID: Adversarial Visual Irregularity Detection  | **[arXiv' 18]** |[`[pdf]`](https://arxiv.org/pdf/1805.09521.pdf)
- MVTec AD -- A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection | **[CVPR' 19]** |  [`[pdf]`](https://www.mvtec.com/fileadmin/Redaktion/mvtec.com/company/research/mvtec_ad.pdf)
- Exploiting Epistemic Uncertainty of Anatomy Segmentation for Anomaly Detection in Retinal OCT | **[IEEE TMI' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1905.12806v1.pdf)
- Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings | **[arXiv' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1911.02357.pdf)
- Attention Guided Anomaly Detection and Localization in Images | **[arXiv' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1911.08616v1.pdf)



## Contact & Feedback
If you have any suggenstions about papers, feel free to mail me :)
- [e-mail](mailto:Hoseong.Lee@cognex.com)
- [blog](https://hoya012.github.io/)
- [pull request](https://github.com/hoya012/awesome-anomaly-detection/pulls)
