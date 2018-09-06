# fMRI_Reconstruction

A Tensorflow implementation of the reconstruction algorithm described in "Deep Image Reconstruction from Human Brain Activity" (https://www.biorxiv.org/content/early/2017/12/28/240317). This algorithm takes reconstructs image stimuli participants were viewing given their brain activity. 

The algorithm takes predicted CNN activity decoded from brain data and reconstructs the image using Gradient Descent with Momentum. In order to constrain reconstructions to the space of natural images, a Deep Generative Network (DGN) taken from a Generative Adverserial Network is utilized. The DGN takes a vector z as input and produces an image. Gradients are calculated with respect to z and loss is computed with respect to the generated image. See the preprint (https://www.biorxiv.org/content/early/2017/12/28/240317) for more details.

The data and scripts used to transfer the brain data to model space was a part of the "Generic Object Decoding dataset" by the same authors. It's available here: https://github.com/KamitaniLab/GenericObjectDecoding 

The Generative Adverserial Network used (Dosovitskiy & Brox 2016 NIPS 2016) was written in Caffe and was converted to Tensorflow. Some code from this repo was used to do this: https://github.com/zjuchenlong/sp-aen.cvpr18. 

The following are example reconstructions (left original images, right reconstructed image from brain activity.)

![alt text](https://github.com/sreejank/fMRI_Reconstruction/blob/master/img1.png)
![alt text](https://github.com/sreejank/fMRI_Reconstruction/blob/master/img2.png)

