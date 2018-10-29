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
For an example of QRNN / QLSTM / RNN / LSTM

```bash
python copy_task.py        
```

or for an example of QCAE / CAE ( (Q)-Convolutional autoencoders)

```bash
python cae.py        
```