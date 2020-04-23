# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 15/4/2020

import random, time
from libmot.tools import Timer, LogManager, AverageMeter

epochs = 10
train_iters = 200
valid_iters = 100
logger = LogManager(log_path='log')
logger.screen_logger(epochs, train_iters, valid_iters)
logger.web_logger(is_train=True, is_valid=True)
timer = Timer()
batch_time = AverageMeter()
losses = AverageMeter(precision=4)
errors = AverageMeter(n=3)
error_names = ['l1', 'l2', 'l3']

logger.write('Begin Training...')
for epoch in range(epochs):
    logger.screen_displayer.epoch_bar.update(epoch)
    logger.screen_displayer.reset_train_bar()

    batch_time.reset()
    losses.reset()
    logger.screen_displayer.train_bar.update(0)

    timer.tic()
    for i in range(train_iters):
        time.sleep(0.05*random.random())
        losses.update(random.random())
        batch_time.update(timer.since_last())
        logger.train_displayer.add_scalar('avg_loss', losses.avg[0], train_iters*epoch+i+1)
        logger.screen_displayer.train_bar.update(i + 1)
        logger.screen_displayer.train_writer.write('Train: Time {}s Loss {}'.format(
            batch_time, losses))
        logger.write('Epoch {}: Batch Time {}s Loss {}'.format(epoch, batch_time, losses))

    if valid_iters <= 0:
        continue
    logger.write('Begin Validation')

    logger.screen_displayer.reset_valid_bar()
    batch_time.reset()
    timer.tic()
    logger.screen_displayer.valid_bar.update(0)
    for j in range(valid_iters):
        time.sleep(0.02*random.random())
        errors.update([random.random(), random.random(), random.random()])
        batch_time.update(timer.since_last())
        logger.screen_displayer.valid_bar.update(j + 1)
        logger.screen_displayer.valid_writer.write(
            'valid: Time {}s Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.write('Cost Time {}s '.format(batch_time.sum[0]))
    logger.write('l1 loss: {:.4f} l2 loss: {:.4f} l3 loss: {:.4f}'.format(
        errors.avg[0], errors.avg[1], errors.avg[2]
    ))
    logger.screen_displayer.valid_bar.update(valid_iters)
    logger.valid_displayer.add_scalar('l1', errors.avg[0], valid_iters * (epoch + 1))
    logger.valid_displayer.add_scalar('l2', errors.avg[1], valid_iters * (epoch + 1))
    logger.valid_displayer.add_scalar('l3', errors.avg[2], valid_iters * (epoch + 1))

    error_string = ', '.join(
        '{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors.avg))
    logger.screen_displayer.valid_writer.write(' * Avg {}'.format(error_string))

logger.stop()







