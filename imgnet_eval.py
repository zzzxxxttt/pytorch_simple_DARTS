import os
import time
import argparse

import torch
import torch.nn as nn
import torch.distributed as dist

from nets.eval_model import NetworkImageNet

from utils.utils import count_parameters, count_flops, DisablePrint
from utils.losses import CrossEntropyLabelSmooth
from utils.dataset import ImgNet_split
from utils.preprocessing import imgnet_transform
from utils.summary import create_summary, create_logger

torch.backends.cudnn.benchmark = True

# Training settings
parser = argparse.ArgumentParser(description='classification_baselines')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='cifar_meta_16_20_4_run4')

parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--lr_steps', type=str, default='30,60,90')
parser.add_argument('--wd', type=float, default=3e-5)

parser.add_argument('--init_ch', type=int, default=48)
parser.add_argument('--num_cells', type=int, default=16)

parser.add_argument('--auxiliary', type=float, default=0.4)
parser.add_argument('--drop_path_prob', type=float, default=0)
parser.add_argument('--label_smooth', type=float, default=0.1)

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--max_epochs', type=int, default=250)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--gpus', type=str, default='0')
parser.add_argument('--num_workers', type=int, default=7)

cfg = parser.parse_args()

cfg.lr_steps = [int(s) for s in cfg.lr_steps.split(',')]

os.chdir(cfg.root_dir)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name + '_imgnet_eval')
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
  train_dataset = ImgNet_split(root=os.path.join(cfg.data_dir, 'train'),
                               transform=imgnet_transform(is_training=True))
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                  num_replicas=num_gpus,
                                                                  rank=cfg.local_rank)
  train_loader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=cfg.batch_size // num_gpus if cfg.dist
                                             else cfg.batch_size,
                                             shuffle=not cfg.dist,
                                             num_workers=cfg.num_workers,
                                             sampler=train_sampler if cfg.dist else None)

  val_dataset = ImgNet_split(root=os.path.join(cfg.data_dir, 'val'),
                             transform=imgnet_transform(is_training=False))
  val_loader = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=cfg.batch_size,
                                           shuffle=False,
                                           num_workers=cfg.num_workers)

  print('==> Building model..')
  genotype = torch.load(os.path.join(cfg.ckpt_dir, 'genotype.pickle'))['genotype']
  model = NetworkImageNet(genotype, cfg.init_ch, cfg.num_cells, cfg.auxiliary, num_classes=1000)

  if not cfg.dist:
    model = nn.DataParallel(model).to(device)
  else:
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[cfg.local_rank, ],
                                                output_device=cfg.local_rank)

  optimizer = torch.optim.SGD(model.parameters(), cfg.lr, momentum=0.9, weight_decay=cfg.wd)
  criterion = CrossEntropyLabelSmooth(num_classes=1000, epsilon=cfg.label_smooth).to(device)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.97)
  warmup = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=2)

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

  def val(epoch):
    # switch to evaluate mode
    model.eval()
    top1 = 0
    top5 = 0
    with torch.no_grad():
      for i, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)

        output, _ = model(inputs)

        # measure accuracy and record loss
        _, pred = output.data.topk(5, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        top1 += correct[:1].view(-1).float().sum(0, keepdim=True).item()
        top5 += correct[:5].view(-1).float().sum(0, keepdim=True).item()

    top1 *= 100 / len(val_dataset)
    top5 *= 100 / len(val_dataset)
    print(' Precision@1 ==> %.2f%%  Precision@1: %.2f%%\n' % (top1, top5))
    summary_writer.add_scalar('Precision@1', top1, epoch)
    summary_writer.add_scalar('Precision@5', top5, epoch)
    return

  for epoch in range(cfg.max_epochs):
    print('\nEpoch: %d lr: %.5f drop_path_prob: %.3f' %
          (epoch, scheduler.get_lr()[0], cfg.drop_path_prob * epoch / cfg.max_epochs))
    model.module.drop_path_prob = cfg.drop_path_prob * epoch / cfg.max_epochs
    train_sampler.set_epoch(epoch)
    train(epoch)
    val(epoch)
    if epoch < 5:
      warmup.step(epoch)
    else:
      scheduler.step(epoch)  # move to here after pytorch1.1.0
    print(model.module.genotype())
    if cfg.local_rank == 0:
      torch.save(model.state_dict(), os.path.join(cfg.ckpt_dir, 'checkpoint.t7'))

  summary_writer.close()
  count_parameters(model)
  count_flops(model, input_size=224)


if __name__ == '__main__':
  if cfg.local_rank == 0:
    main()
  else:
    with DisablePrint():
      main()
