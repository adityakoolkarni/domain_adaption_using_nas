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
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from model_search_pretrain import Network
from architect_coop_pretrain import Architect, softXEnt
# from genotypes import PRIMITIVES
from genotypes import PRIMITIVES_LIGHT as PRIMITIVES

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data',
                    help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate1', type=float,
                    default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float,
                    default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
parser.add_argument('--epochs', type=int, default=20,
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
                    default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float,
                    default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--is_parallel', type=int, default=0)
parser.add_argument('--weight_lambda', type=float, default=1.0)
parser.add_argument('--mmd_lambda', type=float, default=1.0)
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--pretrain_steps', type=int, default=5)
parser.add_argument('--is_ab2', type=int, default=0)
parser.add_argument('--is_changemmd', type=int, default=0)
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
COMMON_CLASSES = 10


def initialize_alphas(steps=4):
    k = sum(1 for i in range(steps) for n in range(2 + i))
    num_ops = len(PRIMITIVES)

    alphas_normal = Variable(
        1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    alphas_reduce = Variable(
        1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True)
    _arch_parameters = [
        alphas_normal,
        alphas_reduce,
    ]
    return _arch_parameters, alphas_normal, alphas_reduce

def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    ## Logging Code
    train_writer = SummaryWriter(log_dir=os.path.join(args.save, 'train'))
    valid_writer = SummaryWriter(log_dir=os.path.join(args.save, 'valid'))
    test_writer = SummaryWriter(log_dir=os.path.join(args.save, 'test'))
    args.train_writer = train_writer
    args.valid_writer = valid_writer
    args.test_writer = test_writer
    np.random.seed(args.seed)
    if not args.is_parallel:
        torch.cuda.set_device(int(args.gpu))
        logging.info('gpu device = %d' % int(args.gpu))
    else:
        logging.info('gpu device = %s' % args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    arch1, alphas_normal1, alphas_reduce1 = initialize_alphas()
    arch2, alphas_normal2, alphas_reduce2 = initialize_alphas()

    model = Network(args.init_channels, COMMON_CLASSES, args.layers, criterion)
    model1 = Network(args.init_channels, COMMON_CLASSES, args.layers, criterion)
    model_pretrain = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
    model1_pretrain = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)

    model._arch_parameters = arch1
    model1._arch_parameters = arch2
    model.alphas_reduce = alphas_reduce1
    model.alphas_normal = alphas_normal1
    model1.alphas_reduce = alphas_reduce2
    model1.alphas_normal = alphas_normal2

    model_pretrain._arch_parameters = arch1
    model1_pretrain._arch_parameters = arch2
    model_pretrain.alphas_reduce = alphas_reduce1
    model_pretrain.alphas_normal = alphas_normal1
    model1_pretrain.alphas_reduce = alphas_reduce2
    model1_pretrain.alphas_normal = alphas_normal2

    # model1.init_weights()
    model = model.cuda()
    model1 = model1.cuda()
    model_pretrain.cuda()
    model1_pretrain.cuda()
    logging.info("param size of model1 = %fMB",
                 utils.count_parameters_in_MB(model))
    logging.info("param size of model2 = %fMB",
                 utils.count_parameters_in_MB(model1))
    # if args.is_parallel:
    #   # import ipdb; ipdb.set_trace()
    #   gpus = [int(i) for i in args.gpu.split(',')]
    #   model = nn.parallel.DataParallel(
    #       model, device_ids=gpus, output_device=gpus[0])
    #   model = model.module

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer1 = torch.optim.SGD(
        model1.parameters(),
        args.learning_rate1,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer_pretrain = torch.optim.SGD(
        model_pretrain.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    optimizer1_pretrain = torch.optim.SGD(
        model1_pretrain.parameters(),
        args.learning_rate1,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    # load target domain - train,test data
    test_transform, valid_transform = utils._data_transforms_cifar10(args)  # same transform applicable for stl-10
    target_domain_data = dset.STL10(root=args.data, split='train', download=True, transform=test_transform)
    # target_domain_test_data = dset.STL10(root=args.data, split='test', download=True, transform=test_transform)
    target_domain_test_data = target_domain_data  ## test on same images that were used in training! No labels are involved

    # source_domain_test_data = dset.CIFAR10(root=args.data, split='test', download=True, transform=test_transform)
    _, test_transform1 = utils._data_transforms_cifar10(args)
    source_domain_test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform1)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(args.train_portion * num_train))

    target_domain_indices = list(range(len(target_domain_data)))
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=False, num_workers=4, drop_last=True)

    external_queue = torch.utils.data.DataLoader(
        target_domain_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(target_domain_indices),
        pin_memory=False, num_workers=4, drop_last=True)

    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]),
        pin_memory=False, num_workers=4)

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
        if epoch >= args.pretrain_steps:
            valid_acc, valid_obj, valid_acc1, valid_obj1 = infer(
                valid_queue,
                model,
                model1,
                criterion,
                args,
                epoch)
            logging.info('valid_acc %f valid_acc1 %f', valid_acc, valid_acc1)

            if epoch % 10 == 0:
                utils.save(model, os.path.join(args.save, 'weights0_' + str(epoch) + '.pt'))
                utils.save(model1, os.path.join(args.save, 'weights1_' + str(epoch) + '.pt'))
                torch.save(alphas_reduce1, 'alphas_reduce1_' + str(epoch) + '.pt')
                torch.save(alphas_normal1, 'alphas_normal1_' + str(epoch) + '.pt')
                torch.save(alphas_reduce2, 'alphas_reduce2_' + str(epoch) + '.pt')
                torch.save(alphas_normal2, 'alphas_normal2_' + str(epoch) + '.pt')

        # test on target domain data
        target_num_test = len(target_domain_test_data)
        target_test_queue = torch.utils.data.DataLoader(
            target_domain_test_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                list(range(target_num_test))),
            pin_memory=False, num_workers=4)

        target_l1_test_acc_top1, target_l1_obj, target_l2_test_acc_top1, target_l2_obj = test_infer(
            target_test_queue,
            model, model1,
            criterion, args,
            epoch, 1)
        logging.info('TARGET TEST - learner 1 - test_acc_top1 %f', target_l1_test_acc_top1)
        logging.info('TARGET TEST - learner 1 - test_acc_obj %f', target_l1_obj)
        logging.info('TARGET TEST - learner 2 - test_acc_top1 %f', target_l2_test_acc_top1)
        logging.info('TARGET TEST - learner 2 - test_acc_obj %f', target_l2_obj)

        # test on source domain data
        source_num_test = len(source_domain_test_data)
        source_test_queue = torch.utils.data.DataLoader(
            source_domain_test_data, batch_size=args.batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                list(range(source_num_test))),
            pin_memory=False, num_workers=4)

        source_l1_test_acc_top1, source_l1_obj, source_l2_test_acc_top1, source_l2_obj = test_infer(
            source_test_queue,
            model, model1,
            criterion, args,
            epoch, 0)
        logging.info('SOURCE TEST - learner 1 - test_acc_top1 %f', source_l1_test_acc_top1)
        logging.info('SOURCE TEST - learner 1 - test_acc_obj %f', source_l1_obj)
        logging.info('SOURCE TEST - learner 2 - test_acc_top1 %f', source_l2_test_acc_top1)
        logging.info('SOURCE TEST - learner 2 - test_acc_obj %f', source_l2_obj)
        ############

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

    valid_queue_iter = iter(valid_queue)
    external_queue_iter = iter(external_queue)
    for step, (input, target) in enumerate(train_queue):
        if epoch >= args.pretrain_steps:
            model.train()
            model1.train()
        n = input.size(0)
        input = input.cuda()
        target = target.cuda(non_blocking=True)

        # get a random minibatch from the search queue with replacement
        try:
            input_search, target_search = next(valid_queue_iter)
        except:
            valid_queue_iter = iter(valid_queue)
            input_search, target_search = next(valid_queue_iter)
        try:
            input_external, target_external = next(external_queue_iter)
        except:
            external_queue_iter = iter(external_queue)
            input_external, target_external = next(external_queue_iter)

        # input_external, target_external = next(iter(external_queue))
        # input_search, target_search = next(iter(valid_queue))
        input_search = input_search.cuda()
        target_search = target_search.cuda(non_blocking=True)

        input_external = input_external.cuda()
        target_external = target_external.cuda(non_blocking=True)
        # import ipdb; ipdb.set_trace()
        if epoch >= args.pretrain_steps:
            assert (model_pretrain._arch_parameters[0]
                    - model._arch_parameters[0]).sum() == 0
            assert (model_pretrain._arch_parameters[1]
                    - model._arch_parameters[1]).sum() == 0
            assert (model1_pretrain._arch_parameters[0]
                    - model1._arch_parameters[0]).sum() == 0
            assert (model1_pretrain._arch_parameters[1]
                    - model1._arch_parameters[1]).sum() == 0
            architect.step(input, target,
                           input_external, target_external,
                           input_search, target_search,
                           lr, lr1, optimizer, optimizer1, unrolled=args.unrolled)

            # train the models for pretrain.
            optimizer_pretrain.zero_grad()
            optimizer1_pretrain.zero_grad()
            logits = model_pretrain(input)
            logits1 = model1_pretrain(input)
            loss = criterion(logits, target)
            loss1 = criterion(logits1, target)

            pretrain_external_out = model_pretrain(input_external)
            pretrain_external_out1 = model1_pretrain(input_external)

            mmd_loss, mmd_loss1 = calculate_mmd(logits, logits1, pretrain_external_out, pretrain_external_out1)

            if args.is_changemmd:
                scale = (epoch + 1)/ (args.pretrain_steps + args.epochs)
                # print(scale)
                loss = loss + loss1 + args.mmd_lambda *(mmd_loss + mmd_loss1)*scale
            else:
                loss = loss + loss1 + args.mmd_lambda *(mmd_loss + mmd_loss1)

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

            mmd, mmd1 = calculate_mmd(logits, logits1, external_out, external_out1)

            if args.is_ab2:
                loss_all = args.weight_lambda * (loss_soft1 + loss_soft) + args.mmd_lambda*(mmd1 + mmd)
            if args.is_changemmd:
                scale = (epoch + 1) / (args.pretrain_steps + args.epochs)
                # print(scale)
                loss_all = loss + loss1 + args.weight_lambda * (loss_soft1 + loss_soft) + args.mmd_lambda *(mmd1 + mmd)*scale
            else:
                loss_all = loss + loss1 + args.weight_lambda * (loss_soft1 + loss_soft) + args.mmd_lambda*(mmd1 + mmd)

            loss_all.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            nn.utils.clip_grad_norm_(model1.parameters(), args.grad_clip)
            optimizer.step()
            optimizer1.step()

            ##Plotting the overall loss - Including both source domain and target domain
            logits_merged = torch.cat((logits, external_out), dim=0)
            target_merged = torch.cat((target, softlabel_other1.argmax(dim=1)), dim=0)
            prec1, prec5 = utils.accuracy(logits_merged, target_merged, topk=(1, 5))

            loss_merged = loss + args.weight_lambda * (loss_soft1) + args.mmd_lambda*(mmd1)

            objs.update(loss_merged.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            ##Plotting the overall loss - Including both source domain and target domain
            logits1_merged = torch.cat((logits1, external_out1), dim=0)
            target1_merged = torch.cat((target, softlabel_other.argmax(dim=1)), dim=0)
            prec1, prec5 = utils.accuracy(logits1_merged, target1_merged, topk=(1, 5))

            loss_merged1 = loss1 + args.weight_lambda * (loss_soft) + args.mmd_lambda*(mmd)

            objs1.update(loss_merged1.item(), n)
            top1_1.update(prec1.item(), n)
            top5_1.update(prec5.item(), n)

            # print('mmd, mmd1, loss_soft, loss_soft1:', mmd.item(), mmd1.item(), loss_soft.item(), loss_soft1.item())

            if step % args.report_freq == 0:
                logging.info('train 1st %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)
                logging.info('train 2nd %03d %e %f %f', step,
                             objs1.avg, top1_1.avg, top5_1.avg)
                args.train_writer.add_scalar('Model_0/TrainLoss', objs.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_0/TrainAccuracyTop1', top1.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_0/TrainAccuracyTop5', top5.avg, epoch * len(train_queue) + step)

                args.train_writer.add_scalar('Model_1/TrainLoss', objs1.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_1/TrainAccuracyTop1', top1_1.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_1/TrainAccuracyTop5', top5_1.avg, epoch * len(train_queue) + step)
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

            pretrain_external_out = model_pretrain(input_external)
            pretrain_external_out1 = model1_pretrain(input_external)
            mmd_loss, mmd_loss1 = calculate_mmd(logits, logits1, pretrain_external_out, pretrain_external_out1)

            if args.is_changemmd:
                scale = (epoch + 1) / (args.pretrain_steps + args.epochs)
                # print(scale)
                loss = loss + loss1 + args.mmd_lambda *(mmd_loss + mmd_loss1)*scale
            else:
                loss = loss + loss1 + args.mmd_lambda *(mmd_loss + mmd_loss1)


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

                args.train_writer.add_scalar('Model_0/PreTrainLoss', objs.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_0/PreTrainAccuracyTop1', top1.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_0/PreTrainAccuracyTop5', top5.avg, epoch * len(train_queue) + step)

                args.train_writer.add_scalar('Model_1/PreTrainLoss', objs1.avg, epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_1/PreTrainAccuracyTop1', top1_1.avg,
                                             epoch * len(train_queue) + step)
                args.train_writer.add_scalar('Model_1/PreTrainAccuracyTop5', top5_1.avg,
                                             epoch * len(train_queue) + step)
    return top1.avg, objs.avg, top1_1.avg, objs1.avg


def calculate_mmd(model0_source_feature, model0_target_feature, model1_source_feature, model1_target_feature):
    data_source_domain = model0_source_feature
    data_source_domain1 = model1_source_feature

    data_target_domain = model0_target_feature
    data_target_domain1 = model1_target_feature

    xx, yy, zz = torch.mm(data_source_domain, data_source_domain.t()), torch.mm(data_target_domain,
                                                                                data_target_domain.t()), torch.mm(
        data_source_domain, data_target_domain.t())
    xx1, yy1, zz1 = torch.mm(data_source_domain1, data_source_domain1.t()), torch.mm(data_target_domain1,
                                                                                     data_target_domain1.t()), torch.mm(
        data_source_domain1, data_target_domain1.t())

    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    rx1 = (xx1.diag().unsqueeze(0).expand_as(xx1))
    ry1 = (yy1.diag().unsqueeze(0).expand_as(yy1))

    alpha = 0.1
    K = torch.exp(- alpha * (rx.t() + rx - 2 * xx))
    L = torch.exp(- alpha * (ry.t() + ry - 2 * yy))
    P = torch.exp(- alpha * (rx.t() + ry - 2 * zz))

    K1 = torch.exp(- alpha * (rx1.t() + rx1 - 2 * xx1))
    L1 = torch.exp(- alpha * (ry1.t() + ry1 - 2 * yy1))
    P1 = torch.exp(- alpha * (rx1.t() + ry1 - 2 * zz1))

    B = model0_source_feature.size(0)
    beta = (1. / (B * (B - 1)))
    gamma = (2. / (B * B))

    mmd =  beta * (torch.sum(K) + torch.sum(L)) - gamma * torch.sum(P)
    mmd1 =  beta * (torch.sum(K1) + torch.sum(L1)) - gamma * torch.sum(P1)

    return mmd, mmd1

def infer(valid_queue, model, model1, criterion, args, epoch):
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
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # for the second model.
            logits = model1(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs1.update(loss.item(), n)
            top1_1.update(prec1.item(), n)
            top5_1.update(prec5.item(), n)
            if step % args.report_freq == 0:
                logging.info('valid 1st %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)
                logging.info('valid 2nd %03d %e %f %f', step,
                             objs1.avg, top1_1.avg, top5_1.avg)
                args.valid_writer.add_scalar('Model_0/ValLoss', objs.avg, epoch * len(valid_queue) + step)
                args.valid_writer.add_scalar('Model_0/ValAccuracyTop1', top1.avg, epoch * len(valid_queue) + step)
                args.valid_writer.add_scalar('Model_0/ValAccuracyTop5', top5.avg, epoch * len(valid_queue) + step)

                args.valid_writer.add_scalar('Model_1/ValLoss', objs1.avg, epoch * len(valid_queue) + step)
                args.valid_writer.add_scalar('Model_1/ValAccuracyTop1', top1_1.avg, epoch * len(valid_queue) + step)
                args.valid_writer.add_scalar('Model_1/ValAccuracyTop5', top5_1.avg, epoch * len(valid_queue) + step)

    return top1.avg, objs.avg, top1_1.avg, objs1.avg

def test_infer(queue, model, model1, criterion, args, epoch, is_target_data):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    objs1 = utils.AvgrageMeter()
    top1_1 = utils.AvgrageMeter()
    top5_1 = utils.AvgrageMeter()

    if (is_target_data):
        test_domain = 'Target'
    else:
        test_domain = 'Source'

    model.eval()
    model1.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)

            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

            # for the second model.
            logits = model1(input)
            loss = criterion(logits, target)

            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            objs1.update(loss.item(), n)
            top1_1.update(prec1.item(), n)
            top5_1.update(prec5.item(), n)
            if step % args.report_freq == 0:
                logging.info('test 1st %03d %e %f %f', step,
                             objs.avg, top1.avg, top5.avg)
                logging.info('test 2nd %03d %e %f %f', step,
                             objs1.avg, top1_1.avg, top5_1.avg)
                args.test_writer.add_scalar(test_domain + 'Model_0/TestLoss', objs.avg, epoch * len(queue) + step)
                args.test_writer.add_scalar(test_domain + 'Model_0/TestAccuracyTop1', top1.avg,
                                            epoch * len(queue) + step)
                args.test_writer.add_scalar(test_domain + 'Model_0/TestAccuracyTop5', top5.avg,
                                            epoch * len(queue) + step)

                args.test_writer.add_scalar(test_domain + 'Model_1/TestLoss', objs1.avg, epoch * len(queue) + step)
                args.test_writer.add_scalar(test_domain + 'Model_1/TestAccuracyTop1', top1_1.avg,
                                            epoch * len(queue) + step)
                args.test_writer.add_scalar(test_domain + 'Model_1/TestAccuracyTop5', top5_1.avg,
                                            epoch * len(queue) + step)

    return top1.avg, objs.avg, top1_1.avg, objs1.avg

if __name__ == '__main__':
    main()
