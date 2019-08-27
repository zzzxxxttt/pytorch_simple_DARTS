import os
import sys
import glob
import shutil
import pickle

import torch
import torch.nn as nn


def snapshot(ckpt_name):
  os.makedirs(os.path.join('./scripts', ckpt_name), exist_ok=True)
  os.makedirs(os.path.join('./scripts', ckpt_name, 'nets'), exist_ok=True)
  os.makedirs(os.path.join('./scripts', ckpt_name, 'utils'), exist_ok=True)
  for script in glob.glob('*.py'):
    dst_file = os.path.join('scripts', ckpt_name, script)
    shutil.copyfile(script, dst_file)
  for script in glob.glob('nets/*.py'):
    dst_file = os.path.join('scripts', ckpt_name, script)
    shutil.copyfile(script, dst_file)
  for script in glob.glob('utils/*.py'):
    dst_file = os.path.join('scripts', ckpt_name, script)
    shutil.copyfile(script, dst_file)


def save_genotype(model, ckpt_dir):
  print('Genotype: ', model.genotype())
  genotype = {'genotype': model.genotype()}
  with open(os.path.join(ckpt_dir, 'genotype.pickle'), 'wb') as handle:
    pickle.dump(genotype, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_checkpoint(model, ckpt_dir, append=''):
  torch.save(model.state_dict(), os.path.join(ckpt_dir, 'checkpoint%s.t7' % append))
  print('checkpoint saved to %s !' % os.path.join(ckpt_dir, 'checkpoint%s.t7' % append))


def count_parameters(model):
  num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
  print("Total num of param = %f M" % sum(num_paras))


def count_flops(model, input_size=224):
  flops = []
  handles = []

  def conv_hook(self, input, output):
    flops.append(output.shape[2] ** 2 *
                 self.kernel_size[0] ** 2 *
                 self.in_channels *
                 self.out_channels /
                 self.groups / 1e6)

  def fc_hook(self, input, output):
    flops.append(self.in_features * self.out_features / 1e6)

  for m in model.modules():
    if isinstance(m, nn.Conv2d):
      handles.append(m.register_forward_hook(conv_hook))
    if isinstance(m, nn.Linear):
      handles.append(m.register_forward_hook(fc_hook))

  _ = model(torch.randn(2, 3, input_size, input_size))
  print("Total FLOPs = %f M" % sum(flops))


class DisablePrint:
  def __enter__(self):
    self._original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

  def __exit__(self, exc_type, exc_val, exc_tb):
    sys.stdout.close()
    sys.stdout = self._original_stdout
