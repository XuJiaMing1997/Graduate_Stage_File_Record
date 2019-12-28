import numpy as np
import math
import os
import matplotlib.pyplot as plt

def plot_loss_acc_lr_OP(loss_lr_dir='./OP/loss_lr.txt', eval_dir='./OP/acc.txt',
                        ifSave=False, SaveDir='./OP/loss_acc_lr.jpg', if_show=True):
        # loss_lr:  'epoch {} loss {} acc {} lr {} time {}:{}:{}\n'
        # acc:  'epoch: {0}\tAvg_acc: {}\tAvg_f_acc: {}\tAvg_b_acc: {}\tAvg_s_acc: {}\tTime: {0}:{1}:{2}\n'

        lossf = open(loss_lr_dir, 'r')
        accf = open(eval_dir, 'r')
        loss_lines = lossf.readlines()
        acc_lines = accf.readlines()
        lossf.close()
        accf.close()

        loss_ep = []
        loss_list = []
        train_acc_list = []
        lr = []
        acc_ep = []
        acc_list = []
        f_acc_list = []
        b_acc_list = []
        s_acc_list = []

        for line in loss_lines:
            line_split = line.split(' ')
            loss_ep.append(int(line_split[1]))
            loss_list.append(float(line_split[3]))
            train_acc_list.append(float(line_split[5]))
            lr.append(float(line_split[7]))
        for line in acc_lines:
            line_split = line.split('\t')
            acc_ep.append(int(line_split[0].split(' ')[1]))
            acc_list.append(float(line_split[1].split(' ')[1]))
            f_acc_list.append(float(line_split[2].split(' ')[1]))
            b_acc_list.append(float(line_split[3].split(' ')[1]))
            s_acc_list.append(float(line_split[4].split(' ')[1]))

        loss_x_axis = np.arange(len(loss_ep))
        eval_x_axis = np.arange(len(acc_ep))
        eval_x_label = ['{0}\n{1}'.format(ep, cnt) for cnt, ep in zip(eval_x_axis, acc_ep)]

        fig = plt.figure(figsize=(16, 6))
        subfig = fig.add_subplot(1, 4, 1)
        subfig.set_title('train loss')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')
        subfig.set_xticks(loss_x_axis)

        train_loss_line, = subfig.plot(loss_x_axis, loss_list,
                                       color='red', linewidth=1.2, linestyle='-', label='train_loss')

        subfig.legend(handles=[train_loss_line, ], labels=['train_loss', ], loc='best')

        subfig = fig.add_subplot(1, 4, 2)
        subfig.set_title('train acc')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')
        subfig.set_xticks(loss_x_axis)

        train_acc_line, = subfig.plot(loss_x_axis, train_acc_list,
                                      color='red', linewidth=1.2, linestyle='-', label='train_acc')

        subfig.legend(handles=[train_acc_line, ], labels=['train_acc', ], loc='best')

        subfig = fig.add_subplot(1, 4, 3)
        subfig.set_title('test acc')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('percentage')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        total_acc_line, = subfig.plot(eval_x_axis, acc_list, color='blue', linewidth=1,
                                      linestyle='-', label='acc')
        f_acc_line, = subfig.plot(eval_x_axis, f_acc_list, color='green', linewidth=1,
                                  linestyle='-', label='f_acc')
        b_acc_line, = subfig.plot(eval_x_axis, b_acc_list, color='yellow', linewidth=1,
                                  linestyle='-', label='b_acc')
        s_acc_line, = subfig.plot(eval_x_axis, s_acc_list, color='red', linewidth=1,
                                  linestyle='-', label='s_acc')

        subfig.legend(handles=[total_acc_line, f_acc_line, b_acc_line, s_acc_line],
                      labels=['total_acc', 'front_acc', 'back_acc', 'side_acc'], loc='best')

        subfig = fig.add_subplot(1, 4, 4)
        subfig.set_title('learning rate')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')

        lr_line, = subfig.plot(loss_ep, lr, color='red', linewidth=1, linestyle='-', label='learning_rate')

        subfig.legend(handles=[lr_line, ], labels=['learning_rate'], loc='best')

        if ifSave:
            fig.savefig(SaveDir)
            print('save plot to {}'.format(SaveDir))
        if if_show:
            plt.show()

        # report max value
        max_acc_value = np.max(acc_list)
        max_acc_idx = acc_list.index(max_acc_value)
        max_acc_ep = acc_ep[max_acc_idx]

        print('MAX acc: {} EP: {}'.format(max_acc_value,max_acc_ep))
        return max_acc_ep


plot_loss_acc_lr_OP(loss_lr_dir='./OP/loss_lr-lr_0.001_SGD.txt',
                    eval_dir='./OP/acc-lr_0.001_SGD.txt',ifSave=False)



