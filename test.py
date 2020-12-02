import warnings
from torch.utils.data.dataloader import DataLoader
from dataloaders.datasets.cityscapes import CityscapesSegmentation
from config_utils.search_args import obtain_search_args
from utils.loss import SegmentationLosses
import torch
import numpy as np
from auto_deeplab import AutoDeeplab
from utils.metrics import Evaluator

model = AutoDeeplab(19, 12).cuda()

args = obtain_search_args()


args.cuda = True
criterion = SegmentationLosses(weight=None, cuda=args.cuda).build_loss(mode=args.loss_type)

def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook

args.crop_size = 64

dataset = CityscapesSegmentation(args, '/home/michela/Scrivania/ML/dataset/cityscapes', 'train')


loader = DataLoader(dataset, batch_size=2, shuffle=True)


grads = {}


def save_grad(name):
    def hook(grad):
        grads[name] = grad
    return hook


evaluator = Evaluator(19)
model.eval()
evaluator.reset()

for i, sample in enumerate(loader):
    image, label = sample['image'].cuda(), sample['label'].cuda()

    # from thop import profile

    # params, flops = profile(model, inputs=(image, ))

    # print(params)
    # print(flops)
    model.betas.register_hook(save_grad('y'))

    prediction = model(image)
    # y = 1e-3*torch.randn(12, 4, 3).cuda()
    # criterion = torch.nn.MSELoss()
    # z = criterion(prediction, label)
    z = prediction.mean()

    z.backward()

    print(grads['y'])
    print(grads['y'].shape)


    # print(grads['y1'])
    # print(grads['y1'].shape)

    if i == 0:
        exit()

Acc = evaluator.Pixel_Accuracy()
Acc_class = evaluator.Pixel_Accuracy_Class()
mIoU = evaluator.Mean_Intersection_over_Union()
FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

print('Validation:')
print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
