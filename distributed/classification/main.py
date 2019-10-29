import os
import argparse
from importlib import import_module
import shutil
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchsso
from torchsso.optim import DistributedFirstOrderOptimizer, DistributedSecondOrderOptimizer, DistributedVIOptimizer
from torchsso.optim.lr_scheduler import MomentumCorrectionLR
from torchsso.utils import Logger

from mpi4py import MPI
import torch.distributed as dist

DATASET_CIFAR10 = 'CIFAR-10'
DATASET_CIFAR100 = 'CIFAR-100'
DATASET_IMAGENET = 'ImageNet'


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--dataset', type=str,
                        choices=[DATASET_CIFAR10, DATASET_CIFAR100, DATASET_IMAGENET],
                        default=DATASET_CIFAR10,
                        help='name of dataset')
    parser.add_argument('--root', type=str, default='./data',
                        help='root of dataset')
    parser.add_argument('--train_root', type=str, default=None,
                        help='root of train dataset')
    parser.add_argument('--val_root', type=str, default=None,
                        help='root of validate dataset')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for valing')
    parser.add_argument('--normalizing_data', action='store_true',
                        help='[data pre processing] normalizing data')
    parser.add_argument('--random_crop', action='store_true',
                        help='[data augmentation] random crop')
    parser.add_argument('--random_resized_crop', action='store_true',
                        help='[data augmentation] random resised crop')
    parser.add_argument('--random_horizontal_flip', action='store_true',
                        help='[data augmentation] random horizontal flip')
    parser.add_argument('--dataset_size_scale', type=float, default=1.,
                        help='ratio multiplied to the actual dataset size')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default=None,
                        help='name of file which defines the architecture')
    parser.add_argument('--arch_name', type=str, default='LeNet5',
                        help='name of the architecture')
    parser.add_argument('--arch_args', type=json.loads, default=None,
                        help='[JSON] arguments for the architecture')
    parser.add_argument('--optim_name', type=str, default=DistributedSecondOrderOptimizer.__name__,
                        help='name of the optimizer')
    parser.add_argument('--optim_args', type=json.loads, default=None,
                        help='[JSON] arguments for the optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the curvature')
    parser.add_argument('--fisher_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the fisher')
    parser.add_argument('--scheduler_name', type=str, default=None,
                        help='name of the learning rate scheduler')
    parser.add_argument('--scheduler_args', type=json.loads, default=None,
                        help='[JSON] arguments for the scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='number of epochs for warmup lr')
    parser.add_argument('--warmup_scheduler_name', type=str, default=None,
                        help='name of the learning rate scheduler')
    parser.add_argument('--warmup_scheduler_args', type=json.loads, default=None,
                        help='[JSON] arguments for the wamup scheduler')
    parser.add_argument('--momentum_correction', action='store_true',
                        help='if True, momentum/LR ratio is kept to be constant')
    parser.add_argument('--non_wd_for_bn', action='store_true',
                        help='(FirstOrderOptimizer only) if True, weight decay is not applied for BatchNorm')
    parser.add_argument('--lars', action='store_true',
                        help='if True, LARS is applied for first-order optimizer')
    # Options
    parser.add_argument('--download', action='store_true', default=False,
                        help='if True, downloads the dataset (CIFAR-10 or 100) from the internet')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of sub processes for data loading')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log_file_name', type=str, default='log',
                        help='log file name')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint path for resume training')
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save output files')
    parser.add_argument('--config', default=None,
                        help='config file path')
    # [COMM]
    parser.add_argument('--dist_init_method', type=str,
                        help='torch.distributed init_method')
    parser.add_argument('--size_data_group', type=int, default=1,
                        help='size of the process groups in which input data are shared')
    parser.add_argument('--num_mc_groups', type=int, default=1,
                        help='number of the process groups in which mc sampled params are shared')

    args = parser.parse_args()
    dict_args = vars(args)

    # Load config file
    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)
        dict_args.update(config)

    # Set random seed
    torch.manual_seed(args.seed)

    # [COMM] Initialize process group
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    ranks = list(range(size))
    rank = comm.Get_rank()
    n_per_node = torch.cuda.device_count()
    device = rank % n_per_node
    torch.cuda.set_device(device)
    init_method = 'tcp://{}:23456'.format(args.dist_init_method)
    dist.init_process_group('nccl', init_method=init_method, world_size=size, rank=rank)

    # [COMM] Setup process group for MC sample parallel
    size_data_group = args.size_data_group
    assert size % size_data_group == 0
    num_mc_groups = args.num_mc_groups
    assert size % num_mc_groups == 0

    if size_data_group > 1:
        num_data_group = size / size_data_group
        data_group_id = rank % num_data_group
        data_group_ranks = ranks[data_group_id:size:num_data_group]
        data_group = dist.new_group(data_group_ranks)

        master_ranks = ranks[0:num_data_group]
        master_group = dist.new_group(master_ranks)
    else:
        num_data_group = size
        data_group_id = rank
        data_group = None
        master_group = dist.new_group(ranks)

    if num_mc_groups > 1:
        size_mc_group = int(size / num_mc_groups)
        mc_group_id = int(rank/size_mc_group)
    else:
        size_mc_group = size
        mc_group_id = 0

    # Setup data augmentation & data pre processing
    train_transforms, val_transforms = [], []

    if args.dataset in [DATASET_CIFAR10, DATASET_CIFAR100]:
        # CIFAR-10/100
        if args.random_crop:
            train_transforms.append(transforms.RandomCrop(32, padding=4))

        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        # ImageNet
        if args.random_resized_crop:
            train_transforms.append(transforms.RandomResizedCrop(224))
        else:
            train_transforms.append(transforms.Resize(256))
            if args.random_crop:
                train_transforms.append(transforms.RandomCrop(224))
            else:
                train_transforms.append(transforms.CenterCrop(224))

        normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        val_transforms.append(transforms.Resize(256))
        val_transforms.append(transforms.CenterCrop(224))

    if args.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())

    train_transforms.append(transforms.ToTensor())
    val_transforms.append(transforms.ToTensor())

    if args.normalizing_data:
        train_transforms.append(normalize)
        val_transforms.append(normalize)

    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)

    # Setup data loader
    if args.dataset == DATASET_IMAGENET:
        # ImageNet
        num_classes = 1000

        train_root = args.root if args.train_root is None else args.train_root
        val_root = args.root if args.val_root is None else args.val_root
        train_dataset = datasets.ImageFolder(root=train_root, transform=train_transform)
        val_dataset = datasets.ImageFolder(root=val_root, transform=val_transform)
    else:
        if args.dataset == DATASET_CIFAR10:
            # CIFAR-10
            num_classes = 10
            dataset_class = datasets.CIFAR10
        else:
            # CIFAR-100
            num_classes = 100
            dataset_class = datasets.CIFAR100

        train_dataset = dataset_class(
            root=args.root, train=True, download=args.download, transform=train_transform)
        val_dataset = dataset_class(
            root=args.root, train=False, download=args.download, transform=val_transform)

    # [COMM] Setup distributed sampler for data parallel & MC sample parallel
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=num_data_group, rank=data_group_id)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        pin_memory=True, sampler=train_sampler, num_workers=args.num_workers)

    # [COMM] Setup distributed sampler for data parallel
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        sampler=val_sampler, num_workers=args.num_workers)

    # Setup model
    if args.arch_file is None:
        arch_class = getattr(models, args.arch_name)
    else:
        _, ext = os.path.splitext(args.arch_file)
        dirname = os.path.dirname(args.arch_file)

        if dirname == '':
            module_path = args.arch_file.replace(ext, '')
        elif dirname == '.':
            module_path = os.path.basename(args.arch_file).replace(ext, '')
        else:
            module_path = '.'.join(os.path.split(args.arch_file)).replace(ext, '')

        module = import_module(module_path)
        arch_class = getattr(module, args.arch_name)

    arch_kwargs = {} if args.arch_args is None else args.arch_args
    arch_kwargs['num_classes'] = num_classes

    model = arch_class(**arch_kwargs)
    setattr(model, 'num_classes', num_classes)
    model = model.to(device)

    # [COMM] Broadcast model parameters
    for param in list(model.parameters()):
        dist.broadcast(param.data, src=0)

    # Setup optimizer
    optim_kwargs = {} if args.optim_args is None else args.optim_args
    acc_steps = optim_kwargs.get('acc_steps', 1)
    global_batch_size = num_data_group * args.batch_size * acc_steps
    total_steps = math.ceil(args.epochs * len(train_loader.dataset) / global_batch_size)

    # Setup optimizer
    if args.optim_name == DistributedVIOptimizer.__name__:
        optimizer = DistributedVIOptimizer(model,
                                           mc_group_id=mc_group_id,
                                           dataset_size=len(train_loader.dataset) * args.dataset_size_scale,
                                           total_steps=total_steps,
                                           seed=args.seed,
                                           **optim_kwargs, curv_kwargs=args.curv_args)
    else:
        assert args.num_mc_groups == 1, 'You cannot use MC sample groups with non-VI optimizers.'
        if args.optim_name == DistributedSecondOrderOptimizer.__name__:
            optimizer = DistributedSecondOrderOptimizer(model, **optim_kwargs, curv_kwargs=args.curv_args)
        else:
            if args.non_wd_for_bn:
                group, group_non_wd = {'params': []}, {'params': [], 'non_wd': True}
                for m in model.children():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                        group_non_wd['params'].extend(m.parameters())
                    else:
                        group['params'].extend(m.parameters())

                params = [group, group_non_wd]
            else:
                params = model.parameters()

            optim_class = getattr(torch.optim, args.optim_name)
            optimizer = optim_class(params, **optim_kwargs)

            for group in optimizer.param_groups:
                if group.get('non_wd', False):
                    group['weight_decay'] = 0

            optimizer = DistributedFirstOrderOptimizer(optimizer, model, dist, lars=args.lars)

    # Setup lr scheduler
    def get_scheduler(name, kwargs):
        scheduler_class = getattr(torchsso.optim.lr_scheduler, name, None)
        if scheduler_class is None:
            scheduler_class = getattr(torch.optim.lr_scheduler, name)
        scheduler_kwargs = {} if kwargs is None else kwargs
        _scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        if args.momentum_correction:
            _scheduler = MomentumCorrectionLR(_scheduler)
        return _scheduler

    if args.scheduler_name is None:
        main_scheduler = None
    else:
        main_scheduler = get_scheduler(args.scheduler_name, args.scheduler_args)

    if args.warmup_scheduler_name is None:
        warmup_scheduler = main_scheduler
    else:
        warmup_scheduler = get_scheduler(args.warmup_scheduler_name, args.warmup_scheduler_args)

    logger = None
    start_epoch = 1

    # Load checkpoint
    if args.resume is not None:
        print('==> Resuming from checkpoint..')
        assert os.path.exists(args.resume), 'Error: no checkpoint file found'
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    if rank == 0:

        # All config
        print('===========================')
        print('dataset: {}'.format(vars(args)['dataset']))
        print('train data size: {}'.format(len(train_loader.dataset)))
        print('val data size: {}'.format(len(val_loader.dataset)))

        print('MPI.COMM_WORLD size: {}'.format(size))
        print('global mini-batch size: {}'.format(global_batch_size))
        print('steps/epoch: {}'.format(math.ceil(len(train_loader.dataset) / global_batch_size)))

        num_mc_samples = optim_kwargs.get('num_mc_samples', None)
        if num_mc_samples is not None:
            print('global num MC samples: {}'.format(num_mc_groups * num_mc_samples))
            print('MC sample group: {} processes/group x {} group'.format(size_mc_group, num_mc_groups))
            print('data group: {} processes/group x {} group'.format(size_data_group, num_data_group))

        if hasattr(optimizer, 'indices'):
            print('layer assignment: {}'.format(optimizer.indices))

        print('---------------------------')

        for key, val in vars(args).items():
            if key == 'dataset':
                continue
            else:
                print('{}: {}'.format(key, val))
        print('===========================')

        # Copy this file & config to args.out
        if not os.path.isdir(args.out):
            os.makedirs(args.out)
        try:
            shutil.copy(os.path.realpath(__file__), args.out)
        except shutil.SameFileError:
            pass
        if args.config is not None:
            try:
                shutil.copy(args.config, args.out)
            except shutil.SameFileError:
                pass
        if args.arch_file is not None:
            try:
                shutil.copy(args.arch_file, args.out)
            except shutil.SameFileError:
                pass

        # Setup logger
        logger = Logger(args.out, args.log_file_name)
        logger.start()

    # Run training
    for epoch in range(start_epoch, args.epochs + 1):

        scheduler = main_scheduler if epoch > args.warmup_epochs else warmup_scheduler

        # train
        accuracy, loss = train(rank, epoch, model, device, train_loader, optimizer, scheduler,
                               args, master_group, data_group_id, data_group, logger)
        # val
        val_accuracy, val_loss = validate(rank, model, val_loader, device, optimizer)

        if rank == 0:
            # write to log
            iteration = epoch * len(train_loader)
            elapsed_time = logger.elapsed_time
            log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                   'accuracy': accuracy, 'loss': loss,
                   'val_accuracy': val_accuracy, 'val_loss': val_loss,
                   'lr': optimizer.param_groups[0]['lr'],
                   'momentum': optimizer.param_groups[0].get('momentum', 0),
                   }
            logger.write(log)

            # save checkpoint
            if epoch % args.checkpoint_interval == 0 or epoch > args.epochs - 3:
                path = os.path.join(args.out, 'epoch{}.ckpt'.format(epoch))
                data = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch
                }
                torch.save(data, path)


def train(rank, epoch, model, device, train_loader, optimizer, scheduler,
          args, master_group, data_group_id=0, data_group=None, logger=None):

    def scheduler_type(_scheduler):
        if _scheduler is None:
            return 'none'
        return getattr(_scheduler, 'scheduler_type', 'epoch')

    if scheduler_type(scheduler) == 'epoch':
        scheduler.step(epoch - 1)

    model.train()

    total_correct = 0
    loss = None
    total_data_size = 0
    epoch_size = len(train_loader.dataset)
    num_iters_in_epoch = len(train_loader)
    base_num_iter = (epoch - 1) * num_iters_in_epoch

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        if scheduler_type(scheduler) == 'iter':
            scheduler.step()

        for name, param in model.named_parameters():
            attr = 'p_pre_{}'.format(name)
            setattr(model, attr, param.detach().clone())

        # update params
        def closure():
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()

            return loss, output

        if isinstance(optimizer, DistributedSecondOrderOptimizer) \
                and optimizer.curv_type == 'Fisher':
            closure = torchsso.get_closure_for_fisher(optimizer, model, data, target, **args.fisher_args)

        loss, output = optimizer.step(closure=closure)
        data_size = torch.tensor(len(data)).to(device)

        # [COMM] reduce across the all processes
        dist.reduce(loss, dst=0)

        # [COMM] reduce across the processes in a data group
        if data_group is not None:
            dist.reduce(output, dst=data_group_id, group=data_group)

        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().data

        # [COMM] reduce across the processes in the master MC sample group
        if dist.get_world_size(master_group) > 1:
            dist.reduce(correct, dst=0, group=master_group)
            dist.reduce(data_size, dst=0, group=master_group)

        # refresh results
        if rank == 0:
            loss = loss.item() / dist.get_world_size()

            correct = correct.item()
            data_size = data_size.item()

            total_correct += correct

            iteration = base_num_iter + batch_idx + 1
            total_data_size += data_size

            is_log_timing = (epoch == 1 and batch_idx == 0) or \
                            (batch_idx + 1) % args.log_interval == 0

            # save log
            if logger is not None and is_log_timing:
                accuracy = 100. * total_correct / total_data_size
                elapsed_time = logger.elapsed_time
                print('epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}, '
                      'accuracy: {:.0f}/{} ({:.2f}%), '
                      'elapsed: {:.1f}s, iters/sec: {:.2f}'.format(
                        epoch, total_data_size, epoch_size, 100. * (batch_idx + 1) / num_iters_in_epoch,
                        loss, total_correct, total_data_size, accuracy, elapsed_time, iteration/elapsed_time))

                lr = optimizer.param_groups[0]['lr']
                m = optimizer.param_groups[0].get('momentum', 0)
                log = {'epoch': epoch, 'iteration': iteration, 'elapsed_time': elapsed_time,
                       'accuracy': accuracy, 'loss': loss, 'lr': lr, 'momentum': m}

                for name, param in model.named_parameters():
                    attr = 'p_pre_{}'.format(name)
                    p_pre = getattr(model, attr)
                    p_norm = param.norm().item()
                    p_shape = list(param.size())
                    p_pre_norm = p_pre.norm().item()
                    g_norm = param.grad.norm().item()
                    upd_norm = param.sub(p_pre).norm().item()
                    noise_scale = getattr(param, 'noise_scale', 0)

                    p_log = {'p_shape': p_shape, 'p_norm': p_norm, 'p_pre_norm': p_pre_norm,
                             'g_norm': g_norm, 'upd_norm': upd_norm, 'noise_scale': noise_scale}
                    log[name] = p_log

                logger.write(log)

    accuracy = 100. * total_correct / epoch_size

    return accuracy, loss


def validate(rank, model, val_loader, device, optimizer):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            if isinstance(optimizer, DistributedVIOptimizer):
                prob = optimizer.prediction(data)
                val_loss += F.nll_loss(torch.log(prob), target, reduction='sum')
                pred = prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            elif hasattr(model, 'mc_prediction'):
                prob = model.mc_prediction(data)
                val_loss += F.nll_loss(torch.log(prob), target, reduction='sum')
                pred = prob.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            else:
                output = model(data)
                val_loss += F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum()

    dist.reduce(val_loss, dst=0)
    dist.reduce(correct, dst=0)

    val_loss = val_loss.item() / len(val_loader.dataset)
    val_accuracy = 100. * correct.item() / len(val_loader.dataset)

    if rank == 0:
        print('\nEval: average loss: {:.4f}, accuracy: {:.0f}/{} ({:.2f}%)\n'.format(
            val_loss, correct, len(val_loader.dataset), val_accuracy))

    return val_accuracy, val_loss


if __name__ == '__main__':
    main()
