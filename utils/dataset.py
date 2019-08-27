import os
import os.path
import torch
import pickle
import logging
import numpy as np

import torch.utils.data as data
from PIL import Image


class MNIST_split(data.Dataset):
  raw_folder = 'raw'
  processed_folder = 'processed'
  training_file = 'training.pt'
  test_file = 'test.pt'

  def __init__(self, root, split='train', split_size=5000,
               transform=None, target_transform=None):
    assert split in ['train', 'val', 'test']
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.split = split  # training set or test set

    if not self._check_exists():
      raise RuntimeError('Dataset not found.' +
                         ' You can use download=True to download it')

    if self.split == 'test':
      self.test_data, self.test_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.test_file))
    else:
      self.train_data, self.train_labels = torch.load(
        os.path.join(self.root, self.processed_folder, self.training_file))
      if self.split == 'train':
        self.train_data = self.train_data[:split_size]
        self.train_labels = self.train_labels[:split_size]
      else:
        self.train_data = self.train_data[-split_size:]
        self.train_labels = self.train_labels[-split_size:]

  def __getitem__(self, index):
    if self.split == 'test':
      img, target = self.test_data[index], self.test_labels[index]
    else:
      img, target = self.train_data[index], self.train_labels[index]

    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img.numpy(), mode='L')

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    if self.split == 'test':
      return len(self.test_data)
    else:
      return len(self.train_data)

  def _check_exists(self):
    return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
           os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))


class CIFAR10_split(data.Dataset):
  """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
  """
  base_folder = 'cifar-10-batches-py'
  train_list = [['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
                ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
                ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
                ['data_batch_4', '634d18415352ddfa80567beed471001a'],
                ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb']]

  test_list = [['test_batch', '40351d587109b95175f43aff81a1287e']]

  def __init__(self, root, split, ratio, transform=None, target_transform=None):
    assert split in ['train', 'val', 'test']
    self.root = os.path.expanduser(root)
    self.transform = transform
    self.target_transform = target_transform
    self.split = split  # training set or test set

    # now load the picked numpy arrays
    if self.split == 'test':
      f = self.test_list[0][0]
      file = os.path.join(self.root, self.base_folder, f)
      fo = open(file, 'rb')
      entry = pickle.load(fo, encoding='latin1')
      self.data = entry['data']
      if 'labels' in entry:
        self.labels = entry['labels']
      else:
        self.labels = entry['fine_labels']
      fo.close()

      self.data = self.data.reshape((-1, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
    else:
      self.data = []
      self.labels = []
      for fentry in self.train_list:
        f = fentry[0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        entry = pickle.load(fo, encoding='latin1')
        self.data.append(entry['data'])
        if 'labels' in entry:
          self.labels += entry['labels']
        else:
          self.labels += entry['fine_labels']
        fo.close()

      self.data = np.concatenate(self.data)
      self.data = self.data.reshape((-1, 3, 32, 32))
      self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
      if self.split == 'train' and 0.0 < ratio < 1.0:
        split_size = int(np.clip(len(self.data) * ratio, 1.0, len(self.data)))
        print('using %d images from start ...' % split_size)
        # logging.getLogger('logger').info('using %d images from start ...' % split_size)
        self.data = self.data[:split_size]
        self.labels = self.labels[:split_size]
      elif self.split == 'val' and 0.0 < ratio < 1.0:
        split_size = int(np.clip(len(self.data) * ratio, 1.0, len(self.data)))
        print('using %d images from end ...' % split_size)
        # logging.getLogger('logger').info('using %d images from end ...' % split_size)
        self.data = self.data[-split_size:]
        self.labels = self.labels[-split_size:]

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, target = self.data[index], self.labels[index]
    # doing this so that it is consistent with all other datasets
    # to return a PIL Image
    img = Image.fromarray(img)

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data)


class CIFAR100_split(CIFAR10_split):
  base_folder = 'cifar-100-python'
  train_list = [['train', '16019d7e3df5f24257cddd939b257f8d']]
  test_list = [['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc']]


def CIFAR_split(cifar, root, split='train', ratio=1.0, transform=None, target_transform=None):
  if cifar == 10:
    return CIFAR10_split(root, split, ratio, transform, target_transform)
  elif cifar == 100:
    return CIFAR100_split(root, split, ratio, transform, target_transform)
  else:
    raise NotImplementedError


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def pil_loader(path):
  # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
  with open(path, 'rb') as f:
    img = Image.open(f)
    return img.convert('RGB')


def accimage_loader(path):
  import accimage
  try:
    return accimage.Image(path)
  except IOError:
    # Potentially a decoding problem, fall back to PIL.Image
    return pil_loader(path)


def default_loader(path):
  from torchvision import get_image_backend
  if get_image_backend() == 'accimage':
    return accimage_loader(path)
  else:
    return pil_loader(path)


def has_file_allowed_extension(filename, allowed_extensions):
  return any(filename.lower().endswith(ext) for ext in allowed_extensions)


def find_classes(dir):
  class_folder_names = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
  class_folder_names.sort()
  class_to_idx = {class_folder_names[i]: i for i in range(len(class_folder_names))}
  return class_folder_names, class_to_idx


def make_dataset(dir, class_to_idx, allowed_extensions, split, ratio):
  samples = []
  for folder_name in sorted(os.listdir(os.path.expanduser(dir))):
    if not os.path.isdir(os.path.join(dir, folder_name)):
      continue

    for root, _, img_names in sorted(os.walk(os.path.join(dir, folder_name))):
      split_size = int(np.clip(len(img_names) * ratio, 1.0, len(img_names)))
      if split == 'start' and 0 < ratio < 1.0:
        img_names = img_names[:split_size]
      elif split == 'end' and 0 < ratio < 1.0:
        img_names = img_names[-split_size:]
      for img_name in sorted(img_names):
        if has_file_allowed_extension(img_name, allowed_extensions):
          samples.append((os.path.join(root, img_name), class_to_idx[folder_name]))

  return samples


class ImgNet_split(data.Dataset):
  '''
  when split is 'train', split_size means num of images excluded for EACH class
  when split is 'val', split_size means num of images choosed for EACH class
  when split is 'test', split_size will not be used!
  '''

  def __init__(self, root, loader=default_loader,
               split='all', ratio=1.0, transform=None, target_transform=None):
    assert split in ['all', 'start', 'end']
    classes, class_to_idx = find_classes(root)
    samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS, split, ratio)
    if len(samples) == 0:
      raise (RuntimeError("Found 0 files in subfolders of: " +
                          root + "\nSupported extensions are: " +
                          ",".join(IMG_EXTENSIONS)))

    self.root = root
    self.loader = loader
    self.extensions = IMG_EXTENSIONS

    self.classes = classes
    self.class_to_idx = class_to_idx
    self.samples = samples
    self.imgs = samples

    self.transform = transform
    self.target_transform = target_transform
    print(len(samples))
    # logging.getLogger('logger').info(len(samples))

  def __getitem__(self, index):
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
      sample = self.transform(sample)
    if self.target_transform is not None:
      target = self.target_transform(target)

    return sample, target

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


if __name__ == '__main__':
  # ds = MNIST_split('../data', split='train', split_size=30000)
  # pass
  ds = CIFAR100_split('../data', split='train')

  # dataset = ImgNet_split(root='E:\\imagenet_raw\\train', split='test', split_size=50)
  # print(len(dataset))
