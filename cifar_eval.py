import os
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

from nets.eval_model import NetworkCIFAR

from utils.utils import count_parameters, count_flops, DisablePrint
from utils.dataset import CIFAR_split
from utils.preprocessing import cifar_search_transform
from utils.summary import create_summary, create_logger

torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='darts')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--lr', type=float, default=0.025)
parser.add_argument('--wd', type=float, default=3e-4)

parser.add_argument('--init_ch', type=int, default=36)
parser.add_argument('--num_cells', type=int, default=20)

parser.add_argument('--auxiliary', type=float, default=0.4)
parser.add_argument('--cutout', type=int, default=16)
parser.add_argument('--drop_path_prob', type=float, default=0.2)

parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--max_epochs', type=int, default=600)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=3)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name + '_eval')
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)


def main():
  logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
  summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)
  print = logger.info

  print(cfg)
  num_gpus = torch.cuda.device_count()
  if cfg.dist:
    device = torch.device('cuda:%d' % cfg.local_rank) if cfg.dist else torch.device('cuda')
    torch.cuda.set_device(cfg.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://',
                            world_size=num_gpus, rank=cfg.local_rank)
  else:
    device = torch.device('cuda')

  print('==> Preparing data..')
  cifar = 100 if 'cifar100' in cfg.log_name else 10
  train_dataset = CIFAR_split(cifar=cifar, root=cfg.data_dir, split='train', ratio=1.0,
                              transform=cifar_search_transform(is_training=True, cutout=cfg.cutout))
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=cfg.local_rank)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus if cfg.dist
                                             else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             sampler=train_sampler if cfg.dist else None)

  test_dataset = CIFAR_split(cifar=cifar, root=cfg.data_dir, split='test',
                             transform=cifar_search_transform(is_training=False))
  test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=cfg.batch_size,
                                            shuffle=False,
                                            num_workers=cfg.num_workers)

  print('==> Building model..')
  genotype = torch.load(os.path.join(cfg.ckpt_dir, 'genotype.pickle'))['genotype']
  model = NetworkCIFAR(genotype, cfg.init_ch, cfg.num_cells, cfg.auxiliary, num_classes=cifar)

  if not cfg.dist:
    model = nn.DataParallel(model).to(device)
  else:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)

  optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  criterion = nn.CrossEntropyLoss().to(device)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.max_epochs)

  # Training
  def train(epoch):
    model.train()

    start_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
      inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)

      outputs, outputs_aux = model(inputs)
      loss = criterion(outputs, targets)
      loss_aux = criterion(outputs_aux, targets)
      loss += cfg.auxiliary * loss_aux

      optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('[%d/%d - %d/%d] cls_loss= %.5f (%d samples/sec)' %
              (epoch, cfg.max_epochs, batch_idx, len(train_loader),
               loss.item(), cfg.batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], step)

  def test(epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)

        outputs, _ = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

      acc = 100. * correct / len(test_loader.dataset)
      print(' Precision@1 ==> %.2f%% \n' % acc)
      summary_writer.add_scalar('Precision@1', acc, global_step=epoch)
    return

  for epoch in range(cfg.max_epochs):
    print('\nEpoch: %d lr: %.5f drop_path_prob: %.3f' %
          (epoch, scheduler.get_lr()[0], cfg.drop_path_prob * epoch / cfg.max_epochs))
    model._modules['module'].drop_path_prob = cfg.drop_path_prob * epoch / cfg.max_epochs
    train_sampler.set_epoch(epoch)
    train(epoch)
    test(epoch)
    scheduler.step(epoch)  # move to here after pytorch1.1.0
    print(model.module.genotype())
    if cfg.local_rank == 0:
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()
  count_parameters(model)
  count_flops(model, input_size=32)


if __name__ == '__main__':
  if cfg.local_rank == 0:
    main()
  else:
    with DisablePrint():
      main()
