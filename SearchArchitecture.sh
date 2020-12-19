
CUDA_VISIBLE_DEVICES=0 python main.py \
 --batch-size 2 --dataset cityscapes \
 --alpha_epoch 10 --filter_multiplier 8 --resize 512 --crop_size 257 \
 --base_size 256 --epochs 40 --block_multiplier 5 --resize 512 \
 --resume /home/antonioc/Scrivania/autoDeepLab/run/cityscapes/checkname/experiment_10/checkpoint.pth.tar
