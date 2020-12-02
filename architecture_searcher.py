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
from collections import OrderedDict
from utils.copy_state_dict import copy_state_dict
from utils.saver import Saver

class ArchitectureSearcher(object):
    def __init__(self, args):
        self.args = args

        #Define Saver
        self.saver = Saver(args)
        #call saver function in which it is created a file
        #where informations train (like dataset,epoch..) are saved
        self.saver.save_experiment_config()


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
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                copy_state_dict(self.model.state_dict(), new_state_dict)

        if args.resume is not None:
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

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

            #state.dict() let to save, update, alter and restore Pytorch model and optimazer
            state_dict = self.model.state_dict()
            #save checkpoint to disk, in this Saver method model_best.pth is created
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


