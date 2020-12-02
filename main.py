import torch
from architecture_searcher import *
from config_utils.search_args import obtain_search_args


def checkCuda_isAvailable():
    if torch.cuda.is_available():
        print('CUDA is available')
        return True
    else:
        print('CUDA not found')
        return False


def main():

    if not checkCuda_isAvailable():
        return

    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.epochs is None:
        epoches = {
            'cityscapes': 6,
        }
        args.epochs = epoches[args.dataset.lower()]

    print(args)

    searcher = ArchitectureSearcher(args)
    print('Starting Epoch:', args.start_epoch)
    print('Total Epoches:', args.epochs)
    for epoch in range(args.start_epoch, args.epochs):
        searcher.training(epoch)
        searcher.validation(epoch)

    #trainer.writer.close()


if __name__ == "__main__":
   main()