##FIRST RUN

if you want to change search parameter you can change search_args.py into config_utils folder.
If you want to change number of epochs you can change into main.py:

epoches = {
            'cityscapes': 6,
        }
        
obviously you can also change dataset name.

set --resume = None on search_args.py

or

CUDA_VISIBLE_DEVICES=0 python main.py --dataset cityscapes


When you run main.py for the first time 'run' folder is created. 
Into run folder you can find another folder with the name of the dataset that you have choose. 
Into dataset folder and checkname folder you find experiment folder which contain checkpoint.pth.tar.

When you resume train (obviously if you have stopped first), resume command create another folder experiment_number 
that continue to run from checkpoint.pth.tar of experiment_number-1. 
Every experiment folder contain parameters of that specific run and so you can look parameter.txt to decide which 
checkpoint you want to resume.
If you what to change run folder name, or the name of other folder, go to:
  --> utils/saver.py and change run folder name in:
  
    self.directory = os.path.join('run', args.dataset, args.checkname)
    
  --> and other name in config_utils/search_args.py in:
  
    --dataset
    --checkname


##IF YOU WANT TO RESUME TRAIN

set --resume = 'path/to/checkpoint.pth.tar'

or

CUDA_VISIBLE_DEVICES=0 python main.py --dataset cityscapes --resume 'path/to/checkpoint.pth.tar'

you can find this file on run/dataset_name/checkname/experiment_number

##DECODE    

CUDA_VISIBLE_DEVICES=0 python decode_autodeeplab.py --dataset cityscapes --resume 'path/to/checkpoint.pth.tar'


##RE-TRAIN

Before you re-train your model, you have to create 'data' folder into autoDeepLab folder.
If you want to change re-train parameter you can change re-train.py on config_utils folder,
of sure you have to change :

    --net_arch with default = 'path/to/network_path.npy'
    --cell_arch with default = 'path/to/genotype.npy'

you can find both files into the same folder of checkpoint.pth.tar

python train.py

##TEST YOUR RE-TRAIN

If you want to change evaluate parameter you can change evaluate.py on config_utils folder,
of sure you have to change :

    --net_arch with default = 'path/to/network_path.npy'
    --cell_arch with default = 'path/to/genotype.npy'

you can find both files into the same folder of checkpoint.pth.tar


CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset cityscapes

this command automatically create a log folder in which you can find result.txt