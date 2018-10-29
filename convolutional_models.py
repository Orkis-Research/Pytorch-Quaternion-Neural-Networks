##########################################################
# pytorch-qnn v1.0                                     
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import torch
import torch.nn        as nn
from torch.nn          import Parameter
from torch.nn          import functional as F
import torch.optim
from torch.autograd    import Variable
from torch             import autograd
from quaternion_layers import *

#
# Models are intended to work on cae.py
# Please use quaternion_layers.py for building custom architectures
#


class QCAE(nn.Module):

    def __init__(self):
        super(QCAE, self).__init__()

        self.act        = nn.Hardtanh()
        self.output_act = nn.Hardtanh()

        # ENCODER
        self.e1 = QuaternionConv(4, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = QuaternionConv(32, 40, kernel_size=3, stride=2, padding=1)

        # DECODER
        self.d1 = QuaternionTransposeConv(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d2 = QuaternionTransposeConv(32, 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        

    def forward(self, x):

        e1  = self.act(self.e1(x))
        e2  = self.act(self.e2(e1))

        d1  = self.act(self.d1(e2))
        d2  = self.d2(d1)
        
        out = self.output_act(d2)

        return out

    def name(self):
        return "QCAE"  

class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.act        = nn.Hardtanh()
        self.output_act = nn.Hardtanh()

        # ENCODER
        self.e1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1)
        self.e2 = nn.Conv2d(32, 40, kernel_size=3, stride=2, padding=1)

        # DECODER
        self.d1 = nn.ConvTranspose2d(40, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.d2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1)
        

    def forward(self, x):

        e1  = self.act(self.e1(x))
        e2  = self.act(self.e2(e1))

        d1  = self.act(self.d1(e2))
        d2  = self.d2(d1)

        out = self.output_act(d2)

        return out

    def name(self):
        return "CAE"  
