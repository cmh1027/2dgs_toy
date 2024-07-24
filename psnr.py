from PIL import Image
import numpy as np
import torch, torchmetrics
from torchvision.transforms import PILToTensor
x = PILToTensor()(Image.open('fig/render007.png'))
img1, img2 = x[..., :240], x[..., 250:]
metric = torchmetrics.PeakSignalNoiseRatio()
metric.update(img1, img2)
print(metric.compute().item())