# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 22/4/2020

from libmot.tools import LogManager, set_random_seed, \
                         Config, AverageMeter, Timer
from libmot.tracker.DAN import DANLoss, DANAugmentation, \
                               MOTTrainDataset, build_dan, collate_fn
from torch.optim import SGD
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import cv2
import torchvision
import numpy as np


def weight_init(module):
    if isinstance(module, nn.Conv2d):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


def main(cfg):
    # Set GPU/CPU Environment
    print('=> Setting GPU/CPU Environment...')
    set_random_seed(26, deterministic=True, benchmark=False)
    devices = cfg.solver.device.split(',')
    cfg['solver']['device'] = []
    for device in devices:
        device_id = int(device)
        if device_id >= 0:
            cfg['solver']['device'].append(torch.device(device_id))
        else:
            cfg['solver']['device'].append(torch.device('cpu'))

    # Set Data Loader
    print('=> Fetching Data from {}'.format(cfg['io']['mot_root']))
    is_valid = (cfg['io']['valid'] is not None) and (len(cfg['io']['valid']) > 0)
    train_set = MOTTrainDataset(cfg,
                                transform=DANAugmentation(cfg),
                                phase='train')
    train_loader = DataLoader(train_set,
                              batch_size=cfg['solver']['batch_size'],
                              num_workers=cfg['solver']['num_workers'],
                              shuffle=True,
                              collate_fn=collate_fn,
                              pin_memory=True)
    print('{} samples found in train sets'.format(len(train_set)))

    if is_valid:
        val_set = MOTTrainDataset(cfg,
                                  transform=DANAugmentation(cfg, type='valid'),
                                  phase='valid')

        print('{} samples found in valid sets'.format(len(val_set)))

        val_loader = DataLoader(val_set,
                                batch_size=cfg['solver']['batch_size'],
                                num_workers=cfg['solver']['num_workers'],
                                shuffle=True,
                                collate_fn=collate_fn,
                                pin_memory=True)

    batch_size = cfg['solver']['batch_size']
    epoch_size = cfg['solver']['epoch_size']
    max_epoch = cfg['solver']['max_epoch']
    #iterations = min(len(train_loader), epoch_size)
    iterations = len(train_loader)

    # Create Model
    print('=> Creating Model')
    dan_net = build_dan(cfg).to(cfg['solver']['device'][0])
    optimizer = SGD(dan_net.parameters(),
                    lr=cfg['solver']['learning_rate'],
                    momentum=cfg['solver']['momentum'],
                    weight_decay=cfg['solver']['weight_decay'])

    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=list(cfg['solver']['learning_rate_steps']),
                                                       gamma=cfg['solver']['gamma'])
    start_epoch = 0
    if cfg['io']['resume'] is not None and len(cfg['io']['resume']) > 0:
        checkpoint = torch.load(cfg['io']['resume'])
        dan_net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_schedule.load_state_dict(checkpoint['lr_schedule'])
    else:
        vgg_weights = torch.load(cfg['io']['base_net_path'])
        dan_net.vgg.load_state_dict(vgg_weights)
        dan_net.extras.apply(weight_init)
        dan_net.selector.apply(weight_init)
        dan_net.final_net.apply(weight_init)

    if cfg['solver']['device'][0].type != 'cpu':
        dan_net = nn.DataParallel(dan_net, cfg['solver']['device'])
        # warmup
        print('=> Warmup GPUs...')
        for d in cfg['solver']['device']:
            tmp = torch.rand(cfg['solver']['batch_size'], 3, cfg['datasets']['image_size'],
                             cfg['datasets']['image_size'], device=d)

    # Set logger for log.txt/screen display/tensorboard/checkpoint
    logger = LogManager(log_path=cfg['io']['log_folder'])
    logger.save_config(cfg)
    logger.web_logger(is_valid=is_valid)
    if is_valid:
        logger.screen_logger(n_epochs=cfg['solver']['max_epoch'],
                             train_iters=iterations,
                             valid_iters=len(val_loader),
                             train_bar_size=2,
                             valid_bar_size=2)
    else:
        logger.screen_logger(n_epochs=cfg['solver']['max_epoch'],
                             train_iters=iterations,
                             train_bar_size=3)
    for i in range(start_epoch-1):
        logger.screen_displayer.epoch_bar.update(i)

    # Begin training
    n_iters = 0
    best_accuracy = -np.inf
    for epoch in range(start_epoch, max_epoch):
        logger.screen_displayer.epoch_bar.update(epoch)
        logger.screen_displayer.reset_train_bar()

        lr = lr_schedule.get_lr()[0]
        train_loss = train(cfg, train_loader, dan_net, optimizer, iterations, logger, n_iters, epoch, lr)
        n_iters += iterations
        lr_schedule.step(epoch)

        is_best = False
        if is_valid and epoch % cfg['solver']['valid_frequency'] == 0:
            logger.screen_displayer.reset_valid_bar()
            accuracy = valid(cfg, val_loader, dan_net, logger, epoch)
            is_best = accuracy > best_accuracy
            best_accuracy = max(best_accuracy, accuracy)

        if epoch % cfg['solver']['saving_frequency'] == 0:
            logger.save_checkpoint(epoch, dan_net.module.state_dict(),
                                   optimizer.state_dict(), lr_schedule.state_dict(),
                                   'dan', is_best)

    logger.stop()


def train(cfg, train_loader, dan_net, optimizer, iterations, logger, n_iter, epoch, lr):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(n=8, precision=4)
    dan_net.train()
    criterion = DANLoss(cfg)

    timer = Timer()
    logger.screen_displayer.train_bar.update(0)
    timer.tic()
    for i, (img_pre, img_next, boxes_pre, boxes_next, labels, mask_pre, mask_next) in enumerate(train_loader):

        img_pre = img_pre.to(cfg['solver']['device'][0])
        img_next = img_next.to(cfg['solver']['device'][0])
        boxes_pre = boxes_pre.to(cfg['solver']['device'][0])
        boxes_next = boxes_next.to(cfg['solver']['device'][0])

        labels = labels.to(cfg['solver']['device'][0])
        mask_pre = mask_pre.to(cfg['solver']['device'][0])
        mask_next = mask_next.to(cfg['solver']['device'][0])

        data_time.update(timer.since_last())

        res = dan_net(img_pre, img_next, boxes_pre, boxes_next)
        optimizer.zero_grad()
        loss_pre, loss_next, loss_similarity, loss_assemble, loss, accuracy_pre, accuracy_next, \
            accuracy, predict_indexes = criterion(res, labels, mask_pre, mask_next)
        losses.update([loss.item(), loss_pre.item(), loss_next.item(), loss_similarity.item(),
                       loss_assemble.item(), accuracy.item(), accuracy_pre.item(), accuracy_next.item()])
        loss.backward()
        optimizer.step()

        logger.train_displayer.add_scalar('loss/total_loss', losses.val[0], n_iter)
        logger.train_displayer.add_scalar('loss/loss_pre', losses.val[1], n_iter)
        logger.train_displayer.add_scalar('loss/loss_next', losses.val[2], n_iter)
        logger.train_displayer.add_scalar('loss/loss_similarity', losses.val[3], n_iter)
        logger.train_displayer.add_scalar('loss/loss_assemble', losses.val[4], n_iter)

        logger.train_displayer.add_scalar('accuracy/accuracy', losses.val[5], n_iter)
        logger.train_displayer.add_scalar('accuracy/accuracy_pre', losses.val[6], n_iter)
        logger.train_displayer.add_scalar('accuracy/accuracy_next', losses.val[7], n_iter)

        if n_iter % 1000 == 0:
            matching_images = show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next,
                                                      mask_pre, mask_next, predict_indexes, cfg)

            logger.train_displayer.add_image('train/match_images', torchvision.utils.make_grid(
                matching_images, nrow=4, normalize=True, scale_each=True), n_iter)

        log_string = 'Epoch{} {}/{}: Avg Loss {} loss_pre {:.4f} loss_next {:.4f} loss_similarity {:.4f} ' \
                     'loss_assemble {:.4f} accuracy {:.4f} lr {:.6f}'.format(epoch, i+1, iterations,
                                                                             *losses.avg[:-2], lr)

        display_string = 'Train: Batch time:{} || Data time:{} || Avg Loss:{:.4f} || Lf:{:.4f} || Lb:{:.4f} || ' \
                         'Lc:{:.4f} || La:{:.4f} || Avg Accuracy {:.4f} || lr:{:.6f}'.format(
                                                                batch_time, data_time, losses.avg[0],
                                                                *losses.val[1:-3], losses.avg[5], lr)

        logger.write(log_string)
        logger.screen_displayer.train_bar.update(i + 1)

        logger.screen_displayer.train_writer.write(display_string)
        if i >= iterations - 1:
            break

        n_iter += 1
        batch_time.update(timer.since_last())

    return losses.avg[0]


@torch.no_grad()
def valid(cfg, valid_loader, dan_net, logger, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(n=8, precision=4)
    dan_net.eval()
    criterion = DANLoss(cfg)

    timer = Timer()
    logger.screen_displayer.valid_bar.update(0)
    logger.write('Begin Validation...')
    timer.tic()
    choice = np.random.randint(len(valid_loader))
    for i, (img_pre, img_next, boxes_pre, boxes_next, labels, mask_pre, mask_next) in enumerate(valid_loader):

        img_pre = img_pre.to(cfg['solver']['device'][0])
        img_next = img_next.to(cfg['solver']['device'][0])
        boxes_pre = boxes_pre.to(cfg['solver']['device'][0])
        boxes_next = boxes_next.to(cfg['solver']['device'][0])
        labels = labels.to(cfg['solver']['device'][0])
        mask_pre = mask_pre.to(cfg['solver']['device'][0])
        mask_next = mask_next.to(cfg['solver']['device'][0])

        data_time.update(timer.since_last())

        res = dan_net(img_pre, img_next, boxes_pre, boxes_next)
        loss_pre, loss_next, loss_similarity, loss_assemble, loss, accuracy_pre, accuracy_next, \
            accuracy, predict_indexes = criterion(res, labels, mask_pre, mask_next)

        losses.update([loss.item(), loss_pre.item(), loss_next.item(), loss_similarity.item(),
                       loss_assemble.item(), accuracy.item(), accuracy_pre.item(), accuracy_next.item()])

        if i == choice:
            matching_images = show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next,
                                                      mask_pre, mask_next, predict_indexes, cfg)

            logger.valid_displayer.add_image('valid/match_images', torchvision.utils.make_grid(
                matching_images, nrow=4, normalize=True, scale_each=True), epoch)

        display_string = 'Valid: Batch time:{} || Data time:{} || Avg Loss:{:.4f} || Avg Accuracy:{:.4f}' \
                         ' || Af:{:.4f} || Ab:{:.4f}'.format(batch_time, data_time, losses.avg[0],
                                                             losses.avg[5], *losses.val[6:])

        logger.screen_displayer.valid_bar.update(i + 1)
        logger.screen_displayer.valid_writer.write(display_string)

        batch_time.update(timer.since_last())

    logger.valid_displayer.add_scalar('loss/total_loss', losses.avg[0], epoch)
    logger.valid_displayer.add_scalar('loss/loss_pre', losses.avg[1], epoch)
    logger.valid_displayer.add_scalar('loss/loss_next', losses.avg[2], epoch)
    logger.valid_displayer.add_scalar('loss/loss_similarity', losses.avg[3], epoch)
    logger.valid_displayer.add_scalar('loss/loss_assemble', losses.avg[4], epoch)

    logger.valid_displayer.add_scalar('accuracy/accuracy', losses.avg[5], epoch)
    logger.valid_displayer.add_scalar('accuracy/accuracy_pre', losses.avg[6], epoch)
    logger.valid_displayer.add_scalar('accuracy/accuracy_next', losses.avg[7], epoch)

    log_string = 'Epoch{} : Loss {} loss_pre {:.4f} loss_next {:.4f} loss_similarity {:.4f} loss_assemble {:.4f} ' \
                 'accuracy:{:.4f}'.format(epoch, *losses.avg[:-2])
    logger.write(log_string)

    return losses.avg[5] - 0.01*losses.avg[0]


def image2list(tensors, cfg):
    if not isinstance(tensors, list):
        tensors = [tensors]
    mean = np.array(cfg['augmentation']['mean_pixel'], dtype=np.float32)
    image_list = []
    for tensor in tensors:
        tensor = tensor.detach().cpu()
        arrays = []
        for array in tensor.numpy():
            array = array.transpose(1, 2, 0)
            array += mean
            array = np.clip(array, 0, 255)
            arrays.append(array.astype(np.uint8))
        image_list.append(arrays)
    return image_list


def tensor2array(tensors):
    if not isinstance(tensors, list):
        tensors = [tensors]
    return [tensor.detach().cpu().numpy() for tensor in tensors]


def show_batch_circle_image(img_pre, img_next, boxes_pre, boxes_next, mask_pre, mask_next, indexes, cfg):
    images = image2list([img_pre, img_next], cfg)
    boxes = tensor2array([boxes_pre, boxes_next])
    masks = tensor2array([mask_pre, mask_next])
    indexes = tensor2array([indexes])

    batch_size = len(images)
    height, width, channels = images[0][0].shape
    gap = 20

    draws = []
    for i in range(batch_size):
        img1 = images[0][i].copy()
        mask1 = masks[0][i, 0, :-1]
        boxes1 = boxes[0][i, :, 0, 0, :][mask1 == 1]
        index = indexes[0][i, 0, :][mask1 == 1]

        img2 = images[1][i].copy()  # [H, W, C]
        mask2 = masks[1][i, 0, :-1]  # [N]
        boxes2 = boxes[1][i, :, 0, 0, :][mask2 == 1]  # [N,2]

        # draw all circle
        for b in boxes1:
            img1 = cv2.circle(img1, tuple(((b + 1) / 2.0 * height).astype(int)), 20, [0, 0, 255],
                              thickness=3)

        for b in boxes2:
            img2 = cv2.circle(img2, tuple(((b + 1) / 2.0 * height).astype(int)), 20, [0, 0, 255],
                              thickness=3)

        img = np.ones((2*height+gap, width, channels), dtype=np.uint8)*255
        img[:height, :width, :] = img1
        img[gap+height:, :] = img2

        # draw the connected boxes
        for j, b1 in enumerate(boxes1):
            if index[j] >= cfg['datasets']['max_object']:
                continue

            color = tuple((np.random.rand(3) * 255).astype(int).tolist())
            start_pt = tuple(((b1 + 1) / 2.0 * height).astype(int))
            b2 = boxes[1][i, :, 0, 0, :][index[j]]
            end_pt = tuple(((b2 + 1) / 2.0 * height).astype(int))
            end_pt = (end_pt[0], end_pt[1]+height+gap)
            img = cv2.circle(img, start_pt, 20, color, thickness=3)
            img = cv2.circle(img, end_pt, 20, color, thickness=3)
            img = cv2.line(img, start_pt, end_pt, color, thickness=3)

        img = torch.from_numpy(img.astype(np.float)).permute(2, 0, 1)
        draws.append(img)
    return torch.stack(draws, dim=0)


if __name__ == '__main__':
    cfg = Config.fromfile('examples/DAN/config.yaml')
    main(cfg)
