import numpy as np

import torch
import torchvision.transforms as transforms


class Cutout:
  def __init__(self, length):
    self.length = length

  def __call__(self, img):
    h, w = img.size(1), img.size(2)
    mask = np.ones((h, w), np.float32)
    y = np.random.randint(h)
    x = np.random.randint(w)

    y1 = np.clip(y - self.length // 2, 0, h)
    y2 = np.clip(y + self.length // 2, 0, h)
    x1 = np.clip(x - self.length // 2, 0, w)
    x2 = np.clip(x + self.length // 2, 0, w)

    mask[y1: y2, x1: x2] = 0.
    mask = torch.from_numpy(mask)
    mask = mask.expand_as(img)
    img *= mask
    return img


def mnist_transform(is_training=True):
  if is_training:
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
  else:
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.1307,), (0.3081,))])
  return transform_list


def cifar_search_transform(is_training=True, cutout=None):
  transform_list = []
  if is_training:
    transform_list += [transforms.RandomCrop(32, padding=4),
                       transforms.RandomHorizontalFlip()]

  transform_list += [transforms.ToTensor(),
                     transforms.Normalize([0.49139968, 0.48215827, 0.44653124],
                                          [0.24703233, 0.24348505, 0.26158768])]

  if cutout is not None:
    transform_list += [Cutout(cutout)]

  return transforms.Compose(transform_list)


def cifar_transform(is_training=True):
  # Data
  if is_training:
    transform_list = transforms.Compose([transforms.RandomHorizontalFlip(),
                                         transforms.Pad(4, padding_mode='reflect'),
                                         transforms.RandomCrop(32, padding=0),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))])

  else:
    transform_list = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                              (0.2023, 0.1994, 0.2010))])

  return transform_list


def imgnet_transform(is_training=True):
  if is_training:
    transform_list = transforms.Compose([transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ColorJitter(brightness=0.4,
                                                                contrast=0.4,
                                                                saturation=0.4,
                                                                hue=0.2),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
  else:
    transform_list = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])
  return transform_list
