# ICASSP 2019

This directory contains source codes and examples of quaternion neural networks investigated for the ICASSP 2019 conference.

QRNN/QLSTM: https://www.researchgate.net/publication/328631837_Bidirectional_Quaternion_Long-Short_Term_Memory_Recurrent_Neural_Networks_for_Speech_Recognition

QCNN/QCAE: https://www.researchgate.net/publication/328631687_Quaternion_Convolutional_Neural_Networks_for_Heterogeneous_Image_Processing

Requirements
------------
A GPU with a running CUDA installation is preferable. Please follow the installation instructions of the root directory.

Usage
------------

For Quaternion Convolutional Autoencoders on color image reconstruction:

```bash
python cae {QCAE,CAE}        
```

For Quaternion Recurrent Neural Networks on copy memory task:

```bash
python copy_task.py       
```
