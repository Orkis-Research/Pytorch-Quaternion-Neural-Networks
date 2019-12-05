# PyTorch-Quaternion-Neural-Networks

This repository offers up-to-date PyTorch implementations of various quaternion neural networks (QNN), such as QRNNs, QLSTMs, QCNNs, or QCAEs. Core components are situated within the core_qnn package, and can be reused to create custom QNNs, based on easy to customize PyTorch modules like QuaternionLinear, QuaternionConv, or QuaternionTransposeConv.

Requirements
------------
1. A GPU with a running CUDA installation is preferable. Please be certain that CUDA is correctly configured if you plan to use GPUs.
2. Install PyTorch and torchvision depending on your environment: [PyTorch](https://pytorch.org/get-started/locally/)
3. Install others needed packages:

```bash
pip install imageio numpy scipy    
```

4. Finally, install our core_qnn package:
```bash
pip install .   
```

Usage
------------

Please navigate through the *exp* directory. Different running examples are proposed based on published papers. If you use this code or part of it, please cite the following paper:

*Titouan Parcollet, Mirco Ravanelli, Mohamed Morchid, Georges Linarès, Chiheb Trabelsi, Renato De Mori, Yoshua Bengio - "Quaternion Recurrent Neural Networks", [OpenReview](https://openreview.net/forum?id=ByMHvs0cFQ)*

```
@inproceedings{
parcollet2018quaternion,
title={Quaternion Recurrent Neural Networks},
author={Titouan Parcollet and Mirco Ravanelli and Mohamed Morchid and Georges Linarès and Chiheb Trabelsi and Renato De Mori and Yoshua Bengio},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=ByMHvs0cFQ},
}
```
