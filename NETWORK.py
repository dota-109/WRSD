import torch
import torch.nn as nn
import copy
# from tensorboardX import SummaryWriter
from utils import WDSR_RESBLOCK

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.relu = nn.Sequential(
            nn.ReLU())

        self.sigmoid = nn.Sequential(
            nn.Sigmoid())

        self.tanh = nn.Sequential(
            nn.Tanh())

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU())

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=True),
            )

        self.WDSR_block = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=(3,1), stride=(1,1), padding=(1,0), bias=True),
	    	nn.Conv2d(48, 64, kernel_size=(1,3), stride=(1,1), padding=(0,1), bias=True))

	



    def forward(self, x):
        y = copy.copy(x)
        y1 = self.conv1(y)
        for i in range(12):
            y2 = self.WDSR_block(y1)
            y1 = y2 + y1
       
        error = self.tanh(self.conv2(y1))
        return x+error/128.0

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


#
# dummy_input = torch.rand(64, 1, 90, 90)
#
# model = Model()
# with SummaryWriter(comment='WD-SDNet', logdir='scalar') as w:
#     w.add_graph(model, (dummy_input, ))

