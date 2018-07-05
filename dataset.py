import torch.utils.data as data
import numpy as np
import torch

import os
import os.path

class RepeatTensor(object):
    def __init__(self, *args):
        self.repeat = args
        
    def __call__(self, tensor):
        tensor = tensor.repeat(*self.repeat)
        return tensor


    def __repr__(self):
        return self.__class__.__name__ + "()"

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)



def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                                                                            "Supported extensions are: " + ",".join(
                extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        
        scene_target = int(self.classes[target].split("_")[0])
        action_target = int(self.classes[target].split("_")[1])

        sample = self.loader(path)
        rgb = torch.Tensor(sample['rgb']).unsqueeze(0)
        audio = torch.Tensor(sample['audio']).unsqueeze(0)

        if self.transform is not None:
            rgb = self.transform(rgb)
            audio = self.transform(audio)

        if self.target_transform is not None:
            scene_target = self.target_transform(scene_target)
            action_target = self.target_transform(action_target)

        return rgb, audio, scene_target, action_target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())                                                              
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def numpy_loader(path):
    return np.load(path, encoding='latin1')


class DataSetFoler_limit(data.Dataset):
    # limit minimum number of dataset's height
    def __init__(self, limit, DSFolder, save, save_file, transform = None):
        self.limit = limit
        self.D_limit = DSFolder
        self.transform = transform

        self.new_idx = []

        print('DataSetFolder_limit\'s Initializer')

        if save != True:
            print('Save File does not exist. Start limiting... limit : %d' % self.limit)
            for i in range(DSFolder.__len__()):
                if i%1000 == 0:
                    p = (i*1.0/DSFolder.__len__()) * 100
                    print('%2.2f%%' % p)
                if DSFolder[i][0].shape[1] >= limit:
                    self.new_idx.append(i)
            
            print('Saving File...')
            f = open(save_file, 'w')
            for i in range(len(self.new_idx)):
                f.write(str(self.new_idx[i])+'\n')

        else:
            print('Save File exist. Loading...')
            f = open(save_file, 'r')
            txt = f.read()
            l = txt.split('\n')
            l = l[:-1]
            self.new_idx = list(map(int,l))

        print('Done')


    def __len__(self):
        return len(self.new_idx)

    def __getitem__(self,index):
        if self.transform is not None:
            rgb = self.transform(self.D_limit[self.new_idx[index]][0])
            audio = self.transform(self.D_limit[self.new_idx[index]][1])

        #if self.target_transform is not None:
        #    scene_target = self.target_transform(scene_target)
        #    action_target = self.target_transform(action_target)

        return rgb, audio, self.D_limit[self.new_idx[index]][2], self.D_limit[self.new_idx[index]][3]