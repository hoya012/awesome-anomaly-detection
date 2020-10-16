# awesome anomaly detection
A curated list of awesome anomaly detection resources. Inspired by [`awesome-architecture-search`](https://github.com/sdukshis/awesome-ml) and [`awesome-automl`](https://github.com/hibayesian/awesome-automl-papers).  

*Last updated: 2020/10/16*

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

Also, typically there are three types of target data. (`time-series data`, and `image data`, `video data`)  
In time-series data, it is aimed to detect a abnormal sections. 
In image, video data, it is aimed to classify abnormal images or to segment abnormal regions, for example, defect in some manufacturing data.  

## Survey Paper
- Deep Learning for Anomaly Detection: A Survey  | **[arXiv' 19]** |[`[pdf]`](https://arxiv.org/pdf/1901.03407.pdf)
- Anomalous Instance Detection in Deep Learning: A Survey | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2003.06979.pdf)
- Deep Learning for Anomaly Detection: A Review | **[arXiv' 20]** |[`[pdf]`](https://arxiv.org/pdf/2007.02500.pdf)

## Table of Contents
- [Time-series anomaly detection](#time-series-anomaly-detection)
- [Video-level anomaly detection](#video-level-anomaly-detection)
- [Image-level anomaly detection](#image-level-anomaly-detection)
  - [Anomaly Classification target](#anomaly-classification-target)
  - [Out-Of-Distribution(OOD) Detection target](#out-of-distributionood-detection-target)
  - [Anomaly Segmentation target](#anomaly-segmentation-target)

## Time-series anomaly detection **(need to survey more..)**
- Anomaly Detection of Time Series  | **[Thesis' 10]** |[`[pdf]`](https://conservancy.umn.edu/bitstream/handle/11299/92985/Cheboli_Deepthi_May2010.pdf?sequence=1)
- Long short term memory networks for anomaly detection in time series | **[ESANN' 15]** |[`[pdf]`](https://www.researchgate.net/publication/304782562_Long_Short_Term_Memory_Networks_for_Anomaly_Detection_in_Time_Series)
 - LSTM-Based System-Call Language Modeling and Robust Ensemble Method for Designing Host-Based Intrusion Detection Systems | **[arXiv' 16]** |   [`[pdf]`](https://arxiv.org/pdf/1611.01726.pdf)
- Time Series Anomaly Detection; Detection of anomalous drops with limited features and sparse examples in noisy highly periodic data | **[arXiv' 17]** |   [`[pdf]`](https://arxiv.org/ftp/arxiv/papers/1708/1708.03665.pdf)
- Anomaly Detection in Multivariate Non-stationary Time Series for Automatic DBMS Diagnosis | **[ICMLA' 17]** | [`[pdf]`](https://arxiv.org/abs/1708.02635)
- Truth Will Out: Departure-Based Process-Level Detection of Stealthy Attacks on Control Systems  | **[ACM CCS '18]** | [`[pdf]`](https://research.chalmers.se/publication/507989/file/507989_Fulltext.pdf)
- Time-Series Anomaly Detection Service at Microsoft  | **[KDD' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1906.03821v1.pdf)
- Robust Anomaly Detection for Multivariate Time Series through Stochastic Recurrent Neural Network  | **[KDD' 19]** |  [`[pdf]`](https://dl.acm.org/doi/10.1145/3292500.3330672)
- A Systematic Evaluation of Deep Anomaly Detection Methods for Time Series | **Under Review** | [`[code]`](https://github.com/KDD-OpenSource/DeepADoTS)
- BeatGAN: Anomalous Rhythm Detection using Adversarially Generated Time | **[IJCAI 19]** | [`[pdf]`](https://www.ijcai.org/Proceedings/2019/0616.pdf)
- MIDAS: Microcluster-Based Detector of Anomalies in Edge Streams  | **[AAAI' 20]** |  [`[pdf]`](https://www.comp.nus.edu.sg/~sbhatia/assets/pdf/midas.pdf) | [`[code]`](https://github.com/bhatiasiddharth/MIDAS)
- Timeseries Anomaly Detection using Temporal Hierarchical One-Class Network | **[NeurIPS' 20]** 

## Video-level anomaly detection
- Abnormal Event Detection in Videos using Spatiotemporal Autoencoder | **[ISNN' 17]** | [`[pdf]`](https://arxiv.org/pdf/1701.01546.pdf)
- Real-world Anomaly Detection in Surveillance Videos | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/abs/1801.04264) [`[project page]`](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/)
- Unsupervised Anomaly Detection for Traffic Surveillance Based on Background Modeling | **[CVPR Workshop' 18]** | [`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Wei_Unsupervised_Anomaly_Detection_CVPR_2018_paper.pdf)
- Dual-Mode Vehicle Motion Pattern Learning for High Performance Road Traffic Anomaly Detection  | **[CVPR Workshop' 18]** | [`[pdf]`](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Xu_Dual-Mode_Vehicle_Motion_CVPR_2018_paper.pdf)
- Detecting Abnormality without Knowing Normality: A Two-stage Approach for Unsupervised Video Abnormal Event Detection | **[ACMMM' 18]** | [`[link]`](https://dl.acm.org/doi/10.1145/3240508.3240615)
- Motion-Aware Feature for Improved Video Anomaly Detection | **[BMVC' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1907.10211v1.pdf)
- Challenges in Time-Stamp Aware Anomaly Detection in Traffic Videos  | **[CVPRW' 19]** |  [`[pdf]`](https://arxiv.org/ftp/arxiv/papers/1906/1906.04574.pdf)
- Learning Regularity in Skeleton Trajectories for Anomaly Detection in Videos  | **[CVPR' 19]** |  [`[pdf]`](https://arxiv.org/pdf/1903.03295.pdf)
- Graph Convolutional Label Noise Cleaner: Train a Plug-and-play Action Classifier for Anomaly Detection | [CVPR'19] | [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhong_Graph_Convolutional_Label_Noise_Cleaner_Train_a_Plug-And-Play_Action_Classifier_CVPR_2019_paper.pdf)
- Graph Embedded Pose Clustering for Anomaly Detection | **[CVPR' 20]** |  [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Markovitz_Graph_Embedded_Pose_Clustering_for_Anomaly_Detection_CVPR_2020_paper.pdf)
- Self-Trained Deep Ordinal Regression for End-to-End Video Anomaly Detection | **[CVPR' 20]** |  [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Pang_Self-Trained_Deep_Ordinal_Regression_for_End-to-End_Video_Anomaly_Detection_CVPR_2020_paper.pdf)
- Learning Memory-Guided Normality for Anomaly Detection | **[CVPR' 20]** | [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)
- Clustering-driven Deep Autoencoder for Video Anomaly Detection | **[ECCV' 20]** |[`[pdf]`](https://cse.buffalo.edu/~jsyuan/papers/2020/ECCV2020-2341-CameraReady.pdf)
- CLAWS: Clustering Assisted Weakly Supervised Learning with Normalcy Suppression for Anomalous Event Detection | **[ECCV' 20]** |[`[pdf]`](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123670358.pdf)
- Cloze Test Helps: Effective Video Anomaly Detection via Learning to Complete Video Events | **[ACM MM' 20]** | [`[pdf]`](https://arxiv.org/pdf/2008.11988.pdf) | [`[code]`](https://github.com/yuguangnudt/VEC_VAD)
- A Self-Reasoning Framework for Anomaly Detection Using Video-Level Labels | **[IEEE SPL' 20]** | [`[pdf]`](https://arxiv.org/pdf/2008.11887.pdf)
- Few-Shot Scene-Adaptive Anomaly Detection	 | **[ECCV' 20]** 

## Image-level anomaly detection

### One Class (Anomaly) Classification target
- Estimating the Support of a High- Dimensional Distribution [**OC-SVM**]  | **[Journal of Neural Computation' 01]** | [`[pdf]`](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.86.5955&rep=rep1&type=pdf)
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
- Generative Probabilistic Novelty Detection with Adversarial Autoencoders | **[NIPS' 18]** |  [`[pdf]`](http://papers.nips.cc/paper/7915-generative-probabilistic-novelty-detection-with-adversarial-autoencoders.pdf) [`[code]`](https://github.com/podgorskiy/GPND)
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
- Effective End-to-end Unsupervised Outlier Detection via Inlier Priority of Discriminative Network | **[NeurIPS' 19]** | [`[pdf]`](http://papers.nips.cc/paper/8830-effective-end-to-end-unsupervised-outlier-detection-via-inlier-priority-of-discriminative-network.pdf) [`[code]`](https://github.com/demonzyj56/E3Outlier)
- Classification-Based Anomaly Detection for General Data | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=H1lK_lBtvS)
- Robust Subspace Recovery Layer for Unsupervised Anomaly Detection   | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=rylb3eBtwr)
- RaPP: Novelty Detection with Reconstruction along Projection Pathway   | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=HkgeGeBYDB)
- Novelty Detection Via Blurring  | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=ByeNra4FDB)
- Deep Semi-Supervised Anomaly Detection   | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=HkgH0TEYwH)
- Robust anomaly detection and backdoor attack detection via differential privacy | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=SJx0q1rtvS)
- Classification-Based Anomaly Detection for General Data | **[ICLR' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2005.02359v1.pdf)
- Old is Gold: Redefining the Adversarially Learned One-Class Classifier Training Paradigm | **[CVPR' 20]** |  [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zaheer_Old_Is_Gold_Redefining_the_Adversarially_Learned_One-Class_Classifier_Training_CVPR_2020_paper.pdf)
- Deep End-to-End One-Class Classifier | **[IEEE TNNLS' 20]** |  [`[pdf]`](https://ieeexplore.ieee.org/document/9059022)
- Mirrored Autoencoders with Simplex Interpolation for Unsupervised Anomaly Detection | **[ECCV' 20]** |  [`[pdf]`](https://arxiv.org/abs/2003.10713)
- Backpropagated Gradient Representations for Anomaly Detection	 | **[ECCV' 20]** 
- CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2007.08176.pdf) | [`[code]`](https://github.com/alinlab/CSI)

### Out-of-Distribution(OOD) Detection target
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
- Outlier Detection in Contingency Tables Using Decomposable Graphical Models | **[SJS' 19]** |  [`[pdf]`](https://onlinelibrary.wiley.com/doi/epdf/10.1111/sjos.12407) [`[code]`](https://github.com/mlindsk/molic)
- Input Complexity and Out-of-distribution Detection with Likelihood-based Generative Models | **[ICLR' 20]** |  [`[pdf]`](https://openreview.net/pdf?id=SyxIWpVYvr)
- Generalized ODIN: Detecting Out-of-Distribution Image Without Learning From Out-of-Distribution Data | **[CVPR' 20]** |  [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Hsu_Generalized_ODIN_Detecting_Out-of-Distribution_Image_Without_Learning_From_Out-of-Distribution_Data_CVPR_2020_paper.pdf)
- A Boundary Based Out-Of-Distribution Classifier for Generalized Zero-Shot Learning | **[ECCV' 20]** |  [`[pdf]`](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123690562.pdf)
- Provable Worst Case Guarantees for the Detection of Out-of-distribution Data | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2007.08473.pdf) | [`[code]`](https://gitlab.com/Bitterwolf/GOOD)
- On the Value of Out-of-Distribution Testing: An Example of Goodhart's Law  | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2005.09241.pdf)
- Likelihood Regret: An Out-of-Distribution Detection Score For Variational Auto-encoder | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2003.02977.pdf)
- OOD-MAML: Meta-Learning for Few-Shot Out-of-Distribution Detection and Classification | **[NeurIPS' 20]** 
- Energy-based Out-of-distribution Detection | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2010.03759.pdf)
- Towards Maximizing the Representation Gap between In-Domain & Out-of-Distribution Examples | **[NeurIPS' 20]** 
- Why Normalizing Flows Fail to Detect Out-of-Distribution Data | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2006.08545.pdf) | [`[code]`](https://github.com/PolinaKirichenko/flows_ood)
- Understanding Anomaly Detection with Deep Invertible Networks through Hierarchies of Distributions and Features | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2006.10848.pdf)
- Further Analysis of Outlier Detection with Deep Generative Models | **[NeurIPS' 20]** 
- CSI: Novelty Detection via Contrastive Learning on Distributionally Shifted Instances | **[NeurIPS' 20]** |  [`[pdf]`](https://arxiv.org/pdf/2007.08176.pdf) | [`[code]`](https://github.com/alinlab/CSI)

### Unsupervised Anomaly Segmentation target
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
- Uninformed Students: Student-Teacher Anomaly Detection with Discriminative Latent Embeddings | **[CVPR' 20]** |  [`[pdf]`](https://openaccess.thecvf.com/content_CVPR_2020/papers/Bergmann_Uninformed_Students_Student-Teacher_Anomaly_Detection_With_Discriminative_Latent_Embeddings_CVPR_2020_paper.pdf)
- Attention Guided Anomaly Detection and Localization in Images | **[ECCV' 20]** |  [`[pdf]`](https://arxiv.org/pdf/1911.08616v1.pdf)
- Encoding Structure-Texture Relation with P-Net for Anomaly Detection in Retinal Images	 | **[ECCV' 20]** 
- Sub-Image Anomaly Detection with Deep Pyramid Correspondences  |  **[arxiv]** | [ `[pdf]`](https://arxiv.org/pdf/2005.02357.pdf) | [`[code]`](https://github.com/byungjae89/SPADE-pytorch)
- Patch SVDD, Patch-level SVDD for Anomaly Detection and Segmentation  | **[arxiv]** | [`[pdf]`](https://arxiv.org/pdf/2006.16067.pdf) | [`[code]`](https://github.com/nuclearboy95/Anomaly-Detection-PatchSVDD-PyTorch)

## Contact & Feedback
If you have any suggenstions about papers, feel free to mail me :)
- [e-mail](mailto:Hoseong.Lee@cognex.com)
- [blog](https://hoya012.github.io/)
- [pull request](https://github.com/hoya012/awesome-anomaly-detection/pulls)
