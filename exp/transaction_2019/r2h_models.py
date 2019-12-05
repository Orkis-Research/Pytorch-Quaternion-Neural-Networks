##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import torch
import torch.nn                         as nn
from   torch.autograd                   import Variable
from   core_qnn.quaternion_layers       import *

#
# R2H and QDNN are intended to work on r2h_ae.py
# Please use quaternion_layers.py for building custom architectures
#

class R2H(nn.Module):
    def __init__(self, proj_dim, proj_act, proj_norm, input_dim):
        super(R2H, self).__init__()

        # Reading options:
        self.proj_dim    = proj_dim
        self.proj_act    = proj_act
        self.proj_norm   = proj_norm
        self.input_dim   = input_dim

        # Projection layer initialization:
        self.proj = nn.Linear(self.input_dim, self.proj_dim)
        if self.proj_act == "tanh":
            self.p_activation = nn.Tanh()
        elif self.proj_act == "hardtanh":
            self.p_activation = nn.Hardtanh()
        else:
            self.p_activation = nn.ReLU()

        self.out_act = nn.Tanh()

        # Decoding layer initialization
        self.d  = QuaternionLinearAutograd(self.proj_dim, self.input_dim)

    def forward(self, x, trained=False):

        # Apply R2H projection
        # proj -> quaternion normalization
        x = self.p_activation(self.proj(x))
        if self.proj_norm:
            x = q_normalize(x)

        if not trained:
            return self.out_act(self.d(x))
        else:
            return x

class QDNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(QDNN, self).__init__()

        self.input_dim   = input_dim
        self.num_classes = num_classes

        # Layers initialization
        self.fc1  = QuaternionLinearAutograd(self.input_dim, 32)
        self.fc2  = QuaternionLinearAutograd(32, 32)
        self.fc3  = QuaternionLinearAutograd(32, 32)
        self.out  = nn.Linear(32, self.num_classes)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.out(x)
