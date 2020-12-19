import argparse


def obtain_evaluate_args():
    parser = argparse.ArgumentParser(description='---------------------evaluate args---------------------')
    parser.add_argument('--train', action='store_true', default=False, help='training mode')
    parser.add_argument('--exp', type=str, default='bnlr7e-3', help='name of experiment')
    parser.add_argument('--gpu', type=int, default=0, help='test time gpu device id')
    parser.add_argument('--backbone', type=str, default='autodeeplab', help='resnet101')
    parser.add_argument('--dataset', type=str, default='cityscapes', help='pascal or cityscapes')
    parser.add_argument('--groups', type=int, default=None, help='num of groups for group normalization')
    parser.add_argument('--epochs', type=int, default=20, help='num of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.00025, help='base learning rate')
    parser.add_argument('--last_mult', type=float, default=1.0, help='learning rate multiplier for last layers')
    parser.add_argument('--scratch', action='store_true', default=False, help='train from scratch')
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batch normalization parameters')
    parser.add_argument('--weight_std', action='store_true', default=False, help='weight standardization')
    parser.add_argument('--beta', action='store_true', default=False, help='resnet101 beta')
    parser.add_argument('--crop_size', type=int, default=513, help='image crop size')
    parser.add_argument('--resume', type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=12, help='number of data loading workers')
    parser.add_argument('--use_ABN', type=bool, default=False, help='whether use ABN')  # False
    parser.add_argument('--affine', default=True, type=bool, help='whether use affine in BN')  # True
    parser.add_argument('--dist', type=bool, default=False, help='whether to use Distribued Sampler (default: False)')
    parser.add_argument('--net_arch', default='/home/antonioc/Scrivania/autoDeepLab/run/cityscapes/checkname/experiment_10/network_path.npy', type=str)
    parser.add_argument('--cell_arch', default='/home/antonioc/Scrivania/autoDeepLab/run/cityscapes/checkname/experiment_10/genotype.npy', type=str)
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--filter_multiplier', type=int, default=8)  # 8
    parser.add_argument('--block_multiplier', type=int, default=5)
    parser.add_argument('--initial_fm', type=int, default=None)  # 512
    parser.add_argument('--eval_scales', default=(1.0,),
                        type=bool, help='whether use eval_scales')  # (1.0,) (0.25,0.5,0.75,1,1.25,1.5)
    args = parser.parse_args()
    return args
