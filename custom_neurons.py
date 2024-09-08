import snntorch as snn
import torch


class CustomLeaky(snn.Leaky):
    def __init__(self, beta):
        super().__init__(beta=beta)

    def forward(self, input, mem):
        spk, mem = super().forward(input, mem)
        return spk, mem
