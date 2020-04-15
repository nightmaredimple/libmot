# -*- coding: utf-8 -*-
# Author : HuangPiao
# Email  : huangpiao2985@163.com
# Date   : 4/11/2019

import random, time
from libmot.tools import Timer, TermLogger, AverageMeter

epochs = 10
train_iters = 200
valid_iters = 100
T = TermLogger(epochs, train_iters, valid_iters)
timer = Timer()
batch_time = AverageMeter()
losses = AverageMeter(precision=4)
errors = AverageMeter(n=3)
error_names = ['l1', 'l2', 'l3']

for epoch in range(epochs):
    T.epoch_bar.update(epoch)
    T.reset_train_bar()

    batch_time.reset()
    losses.reset()
    T.train_bar.update(0)

    timer.tic()
    for i in range(train_iters):
        time.sleep(0.1*random.random())
        losses.update(random.random())
        batch_time.update(timer.since_last())
        T.train_bar.update(i + 1)
        T.train_writer.write('Train: Time {}s Loss {}'.format(
            batch_time, losses))

    if valid_iters <= 0:
        continue

    T.reset_valid_bar()
    batch_time.reset()
    timer.tic()
    T.valid_bar.update(0)
    for j in range(valid_iters):
        time.sleep(0.05*random.random())
        errors.update([random.random(), random.random(), random.random()])
        batch_time.update(timer.since_last())
        T.valid_bar.update(j + 1)
        T.valid_writer.write(
            'valid: Time {}s Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    T.valid_bar.update(valid_iters)

    error_string = ', '.join(
        '{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors.avg))
    T.valid_writer.write(' * Avg {}'.format(error_string))

T.epoch_bar.finish()







