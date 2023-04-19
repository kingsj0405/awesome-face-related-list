# Awesome Face Related List
In summry this repository includes papers and implementations of face modeling and utilization.
A list of things I've used myself and found to be robust and useful.
Many basics of computer vision things will be also included.

## Dataset

- [VoxCeleb2] Chung, J. S., Nagrani, A., & Zisserman, A. (2018). Voxceleb2: Deep speaker recognition. arXiv preprint arXiv:1806.05622. [Homepage](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). [pdf](https://www.robots.ox.ac.uk/~vgg/publications/2018/Chung18a/chung18a.pdf). [kingsj0405/video-preprocessing](https://github.com/kingsj0405/video-preprocessing).
- [WFLW] Wu, W., Qian, C., Yang, S., Wang, Q., Cai, Y., & Zhou, Q. (2018). Look at boundary: A boundary-aware face alignment algorithm. CVPR. [Homepage](https://wywu.github.io/projects/LAB/WFLW.html).
- [LRW] Chung, J. S., & Zisserman, A. (2017). Lip reading in the wild. ACCV. [Homepage](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html). [pdf](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16/chung16.pdf).
- [VFHQ] Xie, L., Wang, X., Zhang, H., Dong, C., & Shan, Y. (2022). VFHQ: A High-Quality Dataset and Benchmark for Video Face Super-Resolution. CVPR Workshop. [pdf](https://openaccess.thecvf.com/content/CVPR2022W/NTIRE/papers/Xie_VFHQ_A_High-Quality_Dataset_and_Benchmark_for_Video_Face_Super-Resolution_CVPRW_2022_paper.pdf).

## Face modeling

### 3D Morphable Face Model (3DMM)
- [survey] Egger, B., Smith, W. A., Tewari, A., Wuhrer, S., Zollhoefer, M., Beeler, T., ... & Vetter, T. (2020). 3d morphable face models—past, present, and future. ACM Transactions on Graphics (TOG), 39(5), 1-38. [arXiv](https://arxiv.org/abs/1909.01815)
- [3DDFA_V2] Guo, J., Zhu, X., Yang, Y., Yang, F., Lei, Z., & Li, S. Z. (2020, November). Towards fast, accurate and stable 3d dense face alignment. ECCV. [code](https://github.com/cleardusk/3DDFA_V2). [arXiv](https://arxiv.org/abs/2009.09960)

## Face Perception

### Face Detection & Tracking
- [dlib] http://dlib.net/face_detector.py.html. [code](http://dlib.net/face_detector.py.html).
- [ArcFace] Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). Arcface: Additive angular margin loss for deep face recognition. CVPR. [pdf](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf). [code](https://github.com/deepinsight/insightface/tree/6b1bc1347798815111212b44334424ff7a9dd1fc/recognition/arcface_torch).
- [RetinaFace] Deng, J., Guo, J., Ververas, E., Kotsia, I., & Zafeiriou, S. (2020). Retinaface: Single-shot multi-level face localisation in the wild. CVPR. [arXiv](https://arxiv.org/abs/1905.00641). [code](https://github.com/deepinsight/insightface/tree/master/detection/retinaface). [code2](https://github.com/ternaus/retinaface).

### Facial Landmark
- [dlib] http://dlib.net/face_landmark_detection.py.html. [code](http://dlib.net/face_landmark_detection.py.html).
- [AdaptiveWingLoss] Wang, X., Bo, L., & Fuxin, L. (2019). Adaptive wing loss for robust face alignment via heatmap regression. ICCV. [code](https://github.com/protossw512/AdaptiveWingLoss). [arXiv](https://arxiv.org/abs/1904.07399).
- [HRNets] Wang, J., Sun, K., Cheng, T., Jiang, B., Deng, C., Zhao, Y., ... & Xiao, B. (2020). Deep high-resolution representation learning for visual recognition. IEEE transactions on pattern analysis and machine intelligence, 43(10), 3349-3364. [code](https://github.com/HRNet/HRNet-Facial-Landmark-Detection). [arXiv](https://arxiv.org/abs/1908.07919). [kingsj0405/hrnet_face_landmark](https://github.com/kingsj0405/hrnet_face_landmark). [kingsj0405/HRNet-Interspecies-Landmark-Detection](https://github.com/kingsj0405/HRNet-Interspecies-Landmark-Detection).
- [FAN] Yang, J., Bulat, A., & Tzimiropoulos, G. (2020, April). Fan-face: a simple orthogonal improvement to deep face recognition. AAAI. [code](https://github.com/1adrianb/face-alignment). [pdf](https://www.adrianbulat.com/downloads/AAAI20/FANFace.pdf).

### Face Tracking
- [SORT] Bewley, A., Ge, Z., Ott, L., Ramos, F., & Upcroft, B. (2016, September). Simple online and realtime tracking. In 2016 IEEE international conference on image processing (ICIP) (pp. 3464-3468). IEEE. [code](https://github.com/abewley/sort). [arXiv](https://arxiv.org/abs/1602.00763).

### Face IQA
- [HyperIQA] Su, S., Yan, Q., Zhu, Y., Zhang, C., Ge, X., Sun, J., & Zhang, Y. (2020). Blindly assess image quality in the wild guided by a self-adaptive hyper network. CVPR. [code](https://github.com/SSL92/hyperIQA). [pdf](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.pdf).

## Generative Model

### Variational Inference (e.g. VAE, Flow-based Model)
- [survey] Kingma, D. P., & Welling, M. (2019). An introduction to variational autoencoders. Foundations and Trends® in Machine Learning, 12(4), 307-392. [arXiv](https://arxiv.org/abs/1906.02691).
- [thesis] Kingma, D. P. (2017). Variational inference & deep learning: A new synthesis. [pdf](https://pure.uva.nl/ws/files/17891313/Thesis.pdf).
- [VAE] Kingma, D. P., & Welling, M. (2014). Auto-encoding variational bayes. ICLR. [arXiv](https://arxiv.org/abs/1312.6114).
- [IAF] Kingma, D. P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., & Welling, M. (2016). Improved variational inference with inverse autoregressive flow. NeurIPS. [arXiv](https://arxiv.org/abs/1606.04934). 
- [Glow] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions. NeurIPS. [arXiv](https://arxiv.org/abs/1807.03039). [code](https://github.com/openai/glow).
- [Flow++] Ho, J., Chen, X., Srinivas, A., Duan, Y., & Abbeel, P. (2019, May). Flow++: Improving flow-based generative models with variational dequantization and architecture design. In International Conference on Machine Learning (pp. 2722-2730). PMLR. [code](https://github.com/aravindsrinivas/flowpp). [arXiv](https://arxiv.org/abs/1902.00275).

### Diffusion Model
- [DDPM] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. NeurIPS. [arXiv](https://arxiv.org/abs/2006.11239). [code](https://github.com/hojonathanho/diffusion).
- [ImprovedDDPM] Nichol, A. Q., & Dhariwal, P. (2021, July). Improved denoising diffusion probabilistic models. In International Conference on Machine Learning (pp. 8162-8171). PMLR. [code](https://github.com/openai/improved-diffusion). [arXiv](https://arxiv.org/abs/2102.09672).
- [GuidedDiffusion] Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis. NuerIPS. [arXiv](https://arxiv.org/abs/2105.05233). [code](https://github.com/openai/guided-diffusion).

### Generative Adversarial Network (GAN)
- [StyleGAN] Karras, T., Laine, S., & Aila, T. (2019). A style-based generator architecture for generative adversarial networks. CVPR. [code](https://github.com/NVlabs/stylegan). [arXiv](https://arxiv.org/abs/1812.04948).
- [StyleGAN2] Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T. (2020). Analyzing and improving the image quality of stylegan. CVPR. [code](https://github.com/NVlabs/stylegan2). [arXiv](https://arxiv.org/abs/1912.04958)
- [StyleGAN2-ADA] Karras, T., Aittala, M., Hellsten, J., Laine, S., Lehtinen, J., & Aila, T. (2020). Training generative adversarial networks with limited data. NuerIPS. [code](https://github.com/NVlabs/stylegan2-ada-pytorch). [arXiv](https://arxiv.org/abs/2006.06676).
- [StyleGAN3] Karras, T., Aittala, M., Laine, S., Härkönen, E., Hellsten, J., Lehtinen, J., & Aila, T. (2021). Alias-free generative adversarial networks. NeurIPS. [code](https://github.com/NVlabs/stylegan3). [arXiv](https://arxiv.org/abs/2106.12423).
- [MoStGAN-V] Shen, X., Li, X., & Elhoseiny, M. (2023). MoStGAN-V: Video Generation with Temporal Motion Styles. CVPR. [code](https://github.com/xiaoqian-shen/MoStGAN-V). [project page](https://xiaoqian-shen.github.io/MoStGAN-V/). [arXiv](https://arxiv.org/abs/2304.02777).
