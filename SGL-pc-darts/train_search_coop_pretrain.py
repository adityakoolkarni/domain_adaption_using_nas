import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model_search_coop import Network
from architect_coop import Architect
from genotypes import PRIMITIVES


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../../data',
                    help='location of the data corpus')
parser.add_argument('--set', type=str, default='cifar10',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.0, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=45,
                    help='num of training epochs')
parser.add_argument('--init_channels', type=int,
                    default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8,
                    help='total number of layers')
parser.add_argument('--model_path', type=str,
                    default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true',
                    default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int,
                    default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float,
                    default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float,
                    default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true',
                    default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float,
                    default=6e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')

# new hyperparams.
parser.add_argument('--weight_lambda', type=float, default=1.0)
parser.add_argument('--pretrain_steps', type=int, default=15)
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10
if args.set == 'cifar100':
  CIFAR_CLASSES = 100


# def initialize_alphas(steps=4):
#   k = sum(1 for i in range(steps) for n in range(2 + i))
#   num_ops = len(PRIMITIVES)

#   alphas_normal = Variable(
#       1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
#   alphas_reduce = Variable(
#       1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
#   _arch_parameters = [
#       alphas_normal,
#       alphas_reduce,
#   ]
#   return _arch_parameters, alphas_normal, alphas_reduce

def initialize_alphas(steps=4):
  k = sum(1 for i in range(steps) for n in range(2 + i))
  num_ops = len(PRIMITIVES)

  alphas_normal = Variable(
      1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
  alphas_reduce = Variable(
      1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
  betas_normal = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
  betas_reduce = Variable(1e-3 * torch.randn(k).cuda(), requires_grad=True)
  _arch_parameters = [
      alphas_normal,
      alphas_reduce,
      betas_normal,
      betas_reduce,
  ]
  return _arch_parameters, alphas_normal, alphas_reduce, \
      betas_normal, betas_reduce


def softXEnt(input, target):
    logprobs = F.log_softmax(input, dim=1)
    return -(target * logprobs).sum() / input.shape[0]


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  arch1, alphas_normal1, alphas_reduce1,\
      betas_normal1, betas_reduce1 = initialize_alphas()
  arch2, alphas_normal2, alphas_reduce2,\
      betas_normal2, betas_reduce2 = initialize_alphas()

  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  model1 = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model1 = model1.cuda()
  # model for pretraining.
  model_pretrain = Network(
      args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model_pretrain = model_pretrain.cuda()
  model1_pretrain = Network(
      args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model1_pretrain = model1_pretrain.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  model._arch_parameters = arch1
  model1._arch_parameters = arch2
  model.alphas_reduce = alphas_reduce1
  model.alphas_normal = alphas_normal1
  model1.alphas_reduce = alphas_reduce2
  model1.alphas_normal = alphas_normal2

  model.betas_reduce = betas_reduce1
  model.betas_normal = betas_normal1
  model1.betas_reduce = betas_reduce2
  model1.betas_normal = betas_normal2

  model_pretrain._arch_parameters = arch1
  model1_pretrain._arch_parameters = arch2
  model_pretrain.alphas_reduce = alphas_reduce1
  model_pretrain.alphas_normal = alphas_normal1
  model1_pretrain.alphas_reduce = alphas_reduce2
  model1_pretrain.alphas_normal = alphas_normal2

  model_pretrain.betas_reduce = betas_reduce1
  model_pretrain.betas_normal = betas_normal1
  model1_pretrain.betas_reduce = betas_reduce2
  model1_pretrain.betas_normal = betas_normal2

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  optimizer1 = torch.optim.SGD(
      model1.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  optimizer_pretrain = torch.optim.SGD(
      model_pretrain.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  optimizer1_pretrain = torch.optim.SGD(
      model1_pretrain.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  if args.set == 'cifar100':
    train_transform, valid_transform = utils._data_transforms_cifar100(args)
    train_data = dset.CIFAR100(
        root=args.data, train=True, download=True, transform=train_transform)
  else:
    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(
        root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  external_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(
          indices[split:num_train]),
      pin_memory=True, num_workers=2)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer1, float(args.epochs), eta_min=args.learning_rate_min)
  scheduler_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer_pretrain, float(args.epochs + args.pretrain_steps), eta_min=args.learning_rate_min)
  scheduler1_pretrain = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer1_pretrain, float(args.epochs + args.pretrain_steps), eta_min=args.learning_rate_min)

  architect = Architect(model, model1, args)

  for epoch in range(args.epochs + args.pretrain_steps):
    lr = scheduler.get_lr()[0]
    lr1 = scheduler1.get_lr()[0]
    lr_pretrain = scheduler_pretrain.get_lr()[0]
    lr1_pretrain = scheduler1_pretrain.get_lr()[0]
    logging.info('epoch %d lr %e lr1 %e lr_pretrain %e lr1_pretrain %e',
                 epoch, lr, lr1, lr_pretrain, lr1_pretrain)
    if epoch >= args.pretrain_steps:
        genotype = model.genotype()
        genotype1 = model1.genotype()
        logging.info('genotype1 = %s', genotype)
        logging.info('genotype2 = %s', genotype1)

        print(F.softmax(model.alphas_normal, dim=-1))
        print(F.softmax(model.alphas_reduce, dim=-1))

        print(F.softmax(model1.alphas_normal, dim=-1))
        print(F.softmax(model1.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj, train_acc1, train_obj1 = train(
        args,
        epoch,
        train_queue,
        valid_queue,
        external_queue,
        model,
        model1,
        model_pretrain,
        model1_pretrain,
        architect,
        criterion,
        optimizer,
        optimizer1,
        optimizer_pretrain,
        optimizer1_pretrain,
        lr,
        lr1,
        lr_pretrain,
        lr1_pretrain)
    if epoch >= args.pretrain_steps:
        logging.info('train_acc %f train_acc1 %f', train_acc, train_acc1)
    else:
        logging.info('pretrain_acc %f pretrain_acc1 %f', train_acc, train_acc1)
    if epoch >= args.pretrain_steps:
        scheduler_pretrain.step()
        scheduler1_pretrain.step()
        scheduler.step()
        scheduler1.step()
    else:
        scheduler_pretrain.step()
        scheduler1_pretrain.step()
    # validation
    if epoch >= args.pretrain_steps and (args.epochs + args.pretrain_steps) - epoch <= 1:
        valid_acc, valid_obj, valid_acc1, valid_obj1 = infer(
            valid_queue,
            model,
            model1,
            criterion)
        logging.info('valid_acc %f valid_acc1 %f', valid_acc, valid_acc1)

        utils.save(model, os.path.join(args.save, 'weights.pt'))
        utils.save(model1, os.path.join(args.save, 'weights1.pt'))



    # lr = scheduler.get_lr()[0]
    # logging.info('epoch %d lr %e', epoch, lr)

    # genotype = model.genotype()
    # logging.info('genotype = %s', genotype)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    # train_acc, train_obj = train(
    #     train_queue, valid_queue, model, architect, criterion, optimizer, lr, epoch)
    # scheduler.step()
    # logging.info('train_acc %f', train_acc)

    # validation
    # if args.epochs - epoch <= 1:
    #   valid_acc, valid_obj = infer(valid_queue, model, criterion)
    #   logging.info('valid_acc %f', valid_acc)

    # utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(args,
          epoch,
          train_queue,
          valid_queue,
          external_queue,
          model,
          model1,
          model_pretrain,
          model1_pretrain,
          architect,
          criterion,
          optimizer,
          optimizer1,
          optimizer_pretrain,
          optimizer1_pretrain,
          lr,
          lr1,
          lr_pretrain,
          lr1_pretrain):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  objs1 = utils.AvgrageMeter()
  top1_1 = utils.AvgrageMeter()
  top5_1 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    if epoch >= args.pretrain_steps:
        model.train()
        model1.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    # input_search, target_search = next(iter(valid_queue))
    try:
        input_search, target_search = next(valid_queue_iter)
    except:
        valid_queue_iter = iter(valid_queue)
        input_search, target_search = next(valid_queue_iter)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    # external data
    # input_external, target_external = next(iter(external_queue))
    try:
        input_external, target_external = next(external_queue_iter)
    except:
        external_queue_iter = iter(external_queue)
        input_external, target_external = next(external_queue_iter)
    input_external = Variable(input_external, requires_grad=False).cuda()
    target_external = Variable(target_external, requires_grad=False).cuda(non_blocking=True)


    if epoch >= 15 + args.pretrain_steps:
      architect.step(input, target,
                       input_external, target_external,
                       input_search, target_search,
                       lr, lr1, optimizer, optimizer1, unrolled=args.unrolled)
    if epoch >= args.pretrain_steps:
        assert (model_pretrain._arch_parameters[0]
                - model._arch_parameters[0]).sum() == 0
        assert (model_pretrain._arch_parameters[1]
                - model._arch_parameters[1]).sum() == 0
        assert (model1_pretrain._arch_parameters[0]
                - model1._arch_parameters[0]).sum() == 0
        assert (model1_pretrain._arch_parameters[1]
                - model1._arch_parameters[1]).sum() == 0
        # architect.step(input, target,
        #                input_external, target_external,
        #                input_search, target_search,
        #                lr, lr1, optimizer, optimizer1, unrolled=args.unrolled)

        # train the models for pretrain.
        optimizer_pretrain.zero_grad()
        optimizer1_pretrain.zero_grad()
        logits = model_pretrain(input)
        logits1 = model1_pretrain(input)
        loss = criterion(logits, target)
        loss1 = criterion(logits1, target)
        loss = loss + loss1
        loss.backward()
        nn.utils.clip_grad_norm_(model_pretrain.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(model1_pretrain.parameters(), args.grad_clip)
        optimizer_pretrain.step()
        optimizer1_pretrain.step()

        # train the models for search.
        optimizer.zero_grad()
        optimizer1.zero_grad()
        logits = model(input)
        logits1 = model1(input)
        loss = criterion(logits, target)
        loss1 = criterion(logits1, target)
        external_out = model(input_external)
        external_out1 = model1(input_external)
        with torch.no_grad():
            softlabel_other = F.softmax(model_pretrain(input_external), 1)
            softlabel_other1 = F.softmax(model1_pretrain(input_external), 1)
        loss_soft = softXEnt(external_out1, softlabel_other)
        loss_soft1 = softXEnt(external_out, softlabel_other1)
        loss_all = loss + loss1 + args.weight_lambda * (loss_soft1 + loss_soft)

        loss_all.backward()

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(model1.parameters(), args.grad_clip)
        optimizer.step()
        optimizer1.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        prec1, prec5 = utils.accuracy(logits1, target, topk=(1, 5))
        objs1.update(loss1.item(), n)
        top1_1.update(prec1.item(), n)
        top5_1.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('train 1st %03d %e %f %f', step,
                       objs.avg, top1.avg, top5.avg)
          logging.info('train 2nd %03d %e %f %f', step,
                       objs1.avg, top1_1.avg, top5_1.avg)
        # return top1.avg, objs.avg, top1_1.avg, objs1.avg
    else:
        assert (model_pretrain._arch_parameters[0]
                - model._arch_parameters[0]).sum() == 0
        assert (model_pretrain._arch_parameters[1]
                - model._arch_parameters[1]).sum() == 0
        assert (model1_pretrain._arch_parameters[0]
                - model1._arch_parameters[0]).sum() == 0
        assert (model1_pretrain._arch_parameters[1]
                - model1._arch_parameters[1]).sum() == 0
        # train the models for pretrain.
        optimizer_pretrain.zero_grad()
        optimizer1_pretrain.zero_grad()
        logits = model_pretrain(input)
        logits1 = model1_pretrain(input)
        loss = criterion(logits, target)
        loss1 = criterion(logits1, target)
        loss = loss + loss1
        loss.backward()
        nn.utils.clip_grad_norm_(model_pretrain.parameters(), args.grad_clip)
        nn.utils.clip_grad_norm_(model1_pretrain.parameters(), args.grad_clip)
        optimizer_pretrain.step()
        optimizer1_pretrain.step()

        # evaluate the pretrained models.
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        prec1, prec5 = utils.accuracy(logits1, target, topk=(1, 5))
        objs1.update(loss1.item(), n)
        top1_1.update(prec1.item(), n)
        top5_1.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('pretrain 1st %03d %e %f %f', step,
                       objs.avg, top1.avg, top5.avg)
          logging.info('pretrain 2nd %03d %e %f %f', step,
                       objs1.avg, top1_1.avg, top5_1.avg)



    # optimizer.zero_grad()
    # logits = model(input)
    # loss = criterion(logits, target)

    # loss.backward()
    # nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    # optimizer.step()

    # prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    # objs.update(loss.data[0], n)
    # top1.update(prec1.data[0], n)
    # top5.update(prec5.data[0], n)

    # if step % args.report_freq == 0:
    #   logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg, top1_1.avg, objs1.avg


def infer(valid_queue, model, model1, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  objs1 = utils.AvgrageMeter()
  top1_1 = utils.AvgrageMeter()
  top5_1 = utils.AvgrageMeter()
  model.eval()
  model1.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
        #input = input.cuda()
        #target = target.cuda(non_blocking=True)
        input = Variable(input, volatile=True).cuda()
        target = Variable(target, volatile=True).cuda(non_blocking=True)
        logits = model(input)
        loss = criterion(logits, target)
        logits1 = model1(input)
        loss1 = criterion(logits1, target)

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.item(), n)
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)
        prec1, prec5 = utils.accuracy(logits1, target, topk=(1, 5))
        n = input.size(0)
        objs1.update(loss1.item(), n)
        top1_1.update(prec1.item(), n)
        top5_1.update(prec5.item(), n)

        if step % args.report_freq == 0:
          logging.info('valid 1st %03d %e %f %f', step,
                       objs.avg, top1.avg, top5.avg)
          logging.info('valid 2nd %03d %e %f %f', step,
                       objs1.avg, top1_1.avg, top5_1.avg)

  return top1.avg, objs.avg, top1_1.avg, objs1.avg


if __name__ == '__main__':
  main()
