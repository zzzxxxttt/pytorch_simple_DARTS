import os
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from nets.search_model import Network

from utils.utils import count_parameters, DisablePrint
from utils.dataset import CIFAR_split
from utils.preprocessing import cifar_search_transform
from utils.second_order_update import *
from utils.summary import create_summary, create_logger

torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='darts')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')

parser.add_argument('--order', type=str, default='1st', choices=['1st', '2nd'])

parser.add_argument('--w_lr', type=float, default=0.025)
parser.add_argument('--w_min_lr', type=float, default=0.001)
parser.add_argument('--w_wd', type=float, default=3e-4)

parser.add_argument('--a_lr', type=float, default=3e-4)
parser.add_argument('--a_wd', type=float, default=1e-3)
parser.add_argument('--a_start', type=int, default=-1)

parser.add_argument('--init_ch', type=int, default=16)
parser.add_argument('--num_cells', type=int, default=8)
parser.add_argument('--num_nodes', type=int, default=4)
parser.add_argument('--replica', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_epochs', type=int, default=50)

parser.add_argument('--log_interval', type=int, default=10)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=2)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus


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

  train_dataset = CIFAR_split(cifar=cifar, root=cfg.data_dir, split='train', ratio=0.5,
                              transform=cifar_search_transform(is_training=True))
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=cfg.local_rank)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus if cfg.dist
                                             else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             sampler=train_sampler if cfg.dist else None)

  val_dataset = CIFAR_split(cifar=cifar, root=cfg.data_dir, split='val', ratio=0.5,
                            transform=cifar_search_transform(is_training=False))
  val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                num_replicas=num_gpus,
                                                                rank=cfg.local_rank)
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.batch_size // num_gpus if cfg.dist
                                           else cfg.batch_size,
                                           shuffle=not cfg.dist,
                                           num_workers=cfg.num_workers,
                                           sampler=val_sampler if cfg.dist else None)

  print('==> Building model..')
  model = Network(C=cfg.init_ch, num_cells=cfg.num_cells,
                  num_nodes=cfg.num_nodes, multiplier=cfg.num_nodes, num_classes=cifar)

  if not cfg.dist:
    model = nn.DataParallel(model).to(device)
  else:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)

  # proxy_model is used for 2nd order update
  if cfg.order == '2nd':
    proxy_model = Network(cfg.init_ch, cfg.num_cells, cfg.num_nodes).cuda()

  count_parameters(model)

  weights = [v for k, v in model.named_parameters() if 'alpha' not in k]
  alphas = [v for k, v in model.named_parameters() if 'alpha' in k]
  optimizer_w = optim.SGD(weights, cfg.w_lr, momentum=0.9, weight_decay=cfg.w_wd)
  optimizer_a = optim.Adam(alphas, lr=cfg.a_lr, betas=(0.5, 0.999), weight_decay=cfg.a_wd)
  criterion = nn.CrossEntropyLoss().cuda()
  scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_w, cfg.max_epochs, eta_min=cfg.w_min_lr)

  alphas = []

  def train(epoch):
    model.train()
    print('\nEpoch: %d lr: %f' % (epoch, scheduler.get_lr()[0]))
    alphas.append([])
    start_time = time.time()

    for batch_idx, ((inputs_w, targets_w), (inputs_a, targets_a)) \
        in enumerate(zip(train_loader, val_loader)):

      inputs_w, targets_w = inputs_w.to(device), targets_w.to(device, non_blocking=True)
      inputs_a, targets_a = inputs_a.to(device), targets_a.to(device, non_blocking=True)

      # 1. update alpha
      if epoch > cfg.a_start:
        optimizer_a.zero_grad()

        if cfg.order == '1st':
          # using 1st order update
          outputs = model(inputs_a)
          val_loss = criterion(outputs, targets_a)
          val_loss.backward()
        else:
          # using 2nd order update
          val_loss = update(model, proxy_model, criterion, optimizer_w,
                            inputs_a, targets_a, inputs_w, targets_w)

        optimizer_a.step()
      else:
        val_loss = torch.tensor([0]).cuda()

      # 2. update weights
      outputs = model(inputs_w)
      cls_loss = criterion(outputs, targets_w)

      optimizer_w.zero_grad()
      cls_loss.backward()
      nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      optimizer_w.step()

      if batch_idx % cfg.log_interval == 0:
        step = len(train_loader) * epoch + batch_idx
        duration = time.time() - start_time

        print('[%d/%d - %d/%d] cls_loss: %5f val_loss: %5f (%d samples/sec)' %
              (epoch, cfg.max_epochs, batch_idx, len(train_loader),
               cls_loss.item(), val_loss.item(), cfg.batch_size * cfg.log_interval / duration))

        start_time = time.time()
        summary_writer.add_scalar('cls_loss', cls_loss.item(), step)
        summary_writer.add_scalar('val_loss', val_loss.item(), step)
        summary_writer.add_scalar('learning rate', optimizer_w.param_groups[0]['lr'], step)

        alphas[-1].append(model.module.alpha_normal.detach().cpu().numpy())
        alphas[-1].append(model.module.alpha_reduce.detach().cpu().numpy())
    return

  def eval(epoch):
    model.eval()

    correct = 0
    total_loss = 0
    with torch.no_grad():
      for step, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)

        outputs = model(inputs)
        total_loss += criterion(outputs, targets).item()
        _, predicted = torch.max(outputs.data, 1)
        correct += predicted.eq(targets.data).cpu().sum().item()

      acc = 100. * correct / len(val_loader.dataset)
      total_loss = total_loss / len(val_loader)
      print('Val_loss==> %.5f Precision@1 ==> %.2f%% \n' % (total_loss, acc))
      summary_writer.add_scalar('Precision@1', acc, global_step=epoch)
      summary_writer.add_scalar('val_loss_per_epoch', total_loss, global_step=epoch)
    return

  for epoch in range(cfg.max_epochs):
    train_sampler.set_epoch(epoch)
    val_sampler.set_epoch(epoch)
    train(epoch)
    eval(epoch)
    scheduler.step(epoch)  # move to here after pytorch1.1.0
    print(model.module.genotype())
    if cfg.local_rank == 0:
      torch.save(alphas, os.path.join(cfg.ckpt_dir, 'alphas.t7'))
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'search_checkpoint.t7'))
      torch.save({'genotype': model.module.genotype()}, os.path.join(cfg.ckpt_dir, 'genotype.t7'))

  summary_writer.close()


if __name__ == '__main__':
  if cfg.local_rank == 0:
    main()
  else:
    with DisablePrint():
      main()
