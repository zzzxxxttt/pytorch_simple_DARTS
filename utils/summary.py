import os
import sys
import logging
from datetime import datetime

import torch

# return a fake summarywriter if tensorbaordX is not installed
try:
  from tensorboardX import SummaryWriter
except ImportError:
  class SummaryWriter:
    def __init__(self, log_dir=None, comment='', **kwargs):
      print('\nunable to import tensorboardX, summary will be recorded by torch!\n')
      self.log_dir = log_dir if log_dir is not None else './logs'
      os.makedirs('./logs', exist_ok=True)
      self.logs = {'comment': comment}
      return

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
      if tag in self.logs:
        self.logs[tag].append((scalar_value, global_step, walltime))
      else:
        self.logs[tag] = [(scalar_value, global_step, walltime)]
      return

    def close(self):
      timestamp = str(datetime.now()).replace(' ', '_').replace(':', '_')
      torch.save(self.logs, os.path.join(self.log_dir, 'summary_%s.pickle' % timestamp))
      return


class EmptySummaryWriter:
  def __init__(self, **kwargs):
    pass

  def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
    pass

  def close(self):
    pass


def create_summary(distributed_rank=0, **kwargs):
  if distributed_rank > 0:
    return EmptySummaryWriter(**kwargs)
  else:
    return SummaryWriter(**kwargs)


def create_logger(distributed_rank=0, save_dir=None):
  logger = logging.getLogger('logger')
  logger.setLevel(logging.DEBUG)

  filename = "log_%s.txt" % (datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

  # don't log results for the non-master process
  if distributed_rank > 0:
    return logger
  ch = logging.StreamHandler(stream=sys.stdout)
  ch.setLevel(logging.DEBUG)
  # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
  formatter = logging.Formatter("[%(asctime)s] %(message)s")
  ch.setFormatter(formatter)
  logger.addHandler(ch)

  if save_dir is not None:
    fh = logging.FileHandler(os.path.join(save_dir, filename))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

  return logger


if __name__ == '__main__':
  sw = create_summary(distributed_rank=1, log_dir='./')
  sw.close()
if __name__ == '__main__':
  logger = create_logger(save_dir='./', distributed_rank=0)
  logger.info('this is info')
  print(logging.getLogger('logger').info('this is info'))
