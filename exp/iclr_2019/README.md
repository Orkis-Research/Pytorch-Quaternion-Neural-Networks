# ICLR 2019

This directory contains source codes and examples of quaternion neural networks investigated for the ICLR 2019 conference. We only release the code for the copy memory task, but a model for speech recognition can easily be build following the same experimental protocol, based on Pytorch-Kaldi: https://github.com/mravanelli/pytorch-kaldi .

QRNN/QLSTM: https://arxiv.org/abs/1806.04418

Requirements
------------
A GPU with a running CUDA installation is preferable. Please follow the installation instructions of the root directory.

Usage
------------

For Quaternion Recurrent Neural Networks on copy memory task:

```bash
python copy_task.py       
```
