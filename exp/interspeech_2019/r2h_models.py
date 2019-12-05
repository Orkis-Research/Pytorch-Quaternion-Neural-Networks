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
# R2HQDNN is intended to work on r2h.py
# Please use quaternion_layers.py for building custom architectures
#

class R2HQDNN(nn.Module):
    def __init__(self, proj_dim, proj_act, proj_norm, input_dim, num_classes):
        super(R2HQDNN, self).__init__()

        # Reading options:
        self.proj_dim    = proj_dim
        self.proj_act    = proj_act
        self.proj_norm   = proj_norm
        self.num_classes = num_classes
        self.input_dim   = input_dim

        # Projection layer initialization:
        self.proj = nn.Linear(self.input_dim, self.proj_dim)
        if self.proj_act == "tanh":
            self.p_activation = nn.Tanh()
        elif self.proj_act == "hardtanh":
            self.p_activation = nn.Hardtanh()
        else:
            self.p_activation = nn.ReLU()


        # QDNN initialization
        self.fc1  = QuaternionLinearAutograd(self.proj_dim, 32)
        self.fc2  = QuaternionLinearAutograd(32, 32)
        self.fc3  = QuaternionLinearAutograd(32, 32)
        self.out  = nn.Linear(32, self.num_classes)

    def forward(self, x):

        x = x.view(x.shape[0], -1)

        # Apply R2H projection
        # proj -> quaternion normalization
        x = self.p_activation(self.proj(x))

        if self.proj_norm:
            x = q_normalize(x)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        return self.out(x)
