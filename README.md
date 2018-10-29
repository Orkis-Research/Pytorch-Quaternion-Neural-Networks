# pytorch-quaternion-neural-networks

This repository contains source codes and many examples of quaternion neural networks applications, including convolutional (QCNN) (https://arxiv.org/abs/1806.07789), and gated or non gated recurrent (QRNN / QLSTM) (https://arxiv.org/abs/1806.04418)

Requirements
------------
A GPU with a running CUDA installation is preferable.
```bash
pip install torch torchvision matplotlib      
```

Usage
------------
For an example of QRNN / QLSTM / RNN / LSTM on a simple copy-task:

```bash
python copy_task.py {blank_size, default=25}        
```

or for an example of QCAE / CAE ((Q)-Convolutional autoencoders) on a gray-scale to color image reconstruction with the KODAK dataset:

```bash
python cae.py {QCAE,CAE}
```