import os
import torch
import numpy as np
from tqdm import tqdm
from mypath import Path
import matplotlib.pyplot as plt
from auto_deeplab import *
from utils.metrics import Evaluator
from utils.lr_scheduler import LR_Scheduler
from utils.loss import SegmentationLosses
from dataloaders import make_data_loader

class ArchitectureSearcher(object):
    def __init__(self, args):
        self.args = args
        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)


        ##TODO: capire cosa è
        weight = None
        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)

        model = AutoDeeplab(self.nclass, 10, self.criterion, self.args.filter_multiplier,
                            self.args.block_multiplier, self.args.step)

        optimizer = torch.optim.SGD(
            model.weight_parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        self.model, self.optimizer = model, optimizer

        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        # Define Evaluator
        ##TODO:capire cosa è
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loaderA), min_lr=args.min_lr)


        self.model = self.model.cuda()


        self.best_pred = 0.0


        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0


    # helper function for data visualization
    def visualize(**images):
        """PLot images in one row."""
        n = len(images)
        plt.figure(figsize=(32, 10))
        for i, (name, image) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.title(' '.join(name.split('_')).title())
            plt.imshow(image)
        plt.show()

    def training(self, epoch):
        train_loss = 0.0
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)

        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            image, target = image.cuda(), target.cuda()

            #plt.imshow(image[0].permute(2,1,0))
            #plt.show()

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            #reset w.grad for each required_grad=True parameter
            self.optimizer.zero_grad()
            #compute mask prediction for image extracted from datasetA
            output = self.model(image)
            #compute lossA(Segmentation loss) between output and target
            loss = self.criterion(output, target)
            #compute loss grad respect to required_grad true parameter
            #and store the value inside x.grad
            loss.backward()
            #update w nn parameter(which are bounded with optimizer) using w.grad
            self.optimizer.step()


            if epoch >= self.args.alpha_epoch:
                search = next(iter(self.train_loaderB))
                image_search, target_search = search['image'], search['label']
                if self.args.cuda:
                    image_search, target_search = image_search.cuda(), target_search.cuda()

                #reset alpha&beta.grad for each required_grad=True parameter
                self.architect_optimizer.zero_grad()
                #comput mask prediction for image extracted from datasetB
                output_search = self.model(image_search)
                #compute lossB(Segmentation loss) between output and target
                arch_loss = self.criterion(output_search, target_search)
                #compute loss grad respect to required_grad true parameter
                #and store the value inside alpha&beta.grad
                arch_loss.backward()
                #update alpha&beta nn parameter(which are bounded with optimizer) using alpha&beta.grad
                self.architect_optimizer.step()


            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch

            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.3f' % train_loss)

        def validation(self, epoch):
            self.model.eval()
            self.evaluator.reset()
            tbar = tqdm(self.val_loader, desc='\r')
            test_loss = 0.0

            for i, sample in enumerate(tbar):
                image, target = sample['image'], sample['label']
                image, target = image.cuda(), target.cuda()
                with torch.no_grad():
                    output = self.model(image)
                loss = self.criterion(output, target)
                test_loss += loss.item()
                tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
                pred = output.data.cpu().numpy()
                target = target.cpu().numpy()
                pred = np.argmax(pred, axis=1)
                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)

            # Fast test during the training
            Acc = self.evaluator.Pixel_Accuracy()
            Acc_class = self.evaluator.Pixel_Accuracy_Class()
            mIoU = self.evaluator.Mean_Intersection_over_Union()
            FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()

            print('Validation:')
            print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
            print('Loss: %.3f' % test_loss)
            new_pred = mIoU
            if new_pred > self.best_pred:
                is_best = True
                self.best_pred = new_pred

                state_dict = self.model.state_dict()
                


