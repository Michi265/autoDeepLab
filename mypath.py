class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            return '/home/antonioc/Scrivania/ML/dataset/cityscape/leftImg8bit_trainvaltest'
            #return '/home/antonioc/Scrivania/ML/dataset/ctyscapeReduced'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
