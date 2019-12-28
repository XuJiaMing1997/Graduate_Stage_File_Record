import numpy as np
import os
import math
import matplotlib.pyplot as plt

def plot_loss_acc_lr_MIE(eval_dir='./MIE/acc.txt', loss_lr_dir = './MIE/loss_lr.txt',
                         ifSave=False, SaveDir='./MIE/loss_acc_lr.jpg'):
        # loss file: 'epoch {} train_loss {} softmax_loss {} triplet_loss {} center_loss {} '
        #            'mask_loss {} lr {} time {}:{}:{}\n'
        # acc file:  'epoch: {0}\t''1: {0}\t5: {1}\t10: {2}\t20: {3}\tmAP: {4}\t''Time: {0}:{1}:{2}\n'
        f_loss = open(loss_lr_dir, 'r')
        f_eval = open(eval_dir, 'r')
        loss_lines = f_loss.readlines()
        eval_lines = f_eval.readlines()
        f_loss.close()
        f_eval.close()

        loss_epoch = []
        loss = []
        s_loss = []
        t_loss = []
        c_loss = []
        m_loss = []
        lr = []
        for line in loss_lines:
            line_split = line.strip('\n').split(' ')
            loss_epoch.append(int(line_split[1]))
            loss.append(float(line_split[3]))
            s_loss.append(float(line_split[5]))
            t_loss.append(float(line_split[7]))
            c_loss.append(float(line_split[9]))
            m_loss.append(float(line_split[11]))
            lr.append(float(line_split[13]))

        eval_epoch = []
        rank = []
        mAP = []
        for line in eval_lines:
            tem_rank = []
            line_split = line.strip('\n').split('\t')
            eval_epoch.append(int(line_split[0].split(' ')[1]))
            tem_rank.append(float(line_split[1].split(' ')[1]))
            tem_rank.append(float(line_split[2].split(' ')[1]))
            tem_rank.append(float(line_split[3].split(' ')[1]))
            tem_rank.append(float(line_split[4].split(' ')[1]))
            mAP.append(float(line_split[5].split(' ')[1]))
            rank.append(np.array(tem_rank))

        rank = np.array(rank)
        loss_x_axis = np.arange(len(loss_epoch))

        eval_x_axis = np.arange(len(eval_epoch))
        eval_x_label = ['{0}\n{1}'.format(ep, cnt) for cnt, ep in zip(eval_x_axis, eval_epoch)]

        fig = plt.figure(figsize=(16, 6))
        subfig = fig.add_subplot(1, 4, 1)
        subfig.set_title('train loss')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')
        subfig.set_xticks(loss_x_axis)

        train_loss_line, = subfig.plot(loss_x_axis, loss,
                                       color='red', linewidth=1.2, linestyle='-', label='train_loss')
        softmax_loss_line, = subfig.plot(loss_x_axis, s_loss,
                                         color='yellow', linewidth=1.0, linestyle='--',
                                         label='softmax_loss')
        triplet_loss_line, = subfig.plot(loss_x_axis, t_loss,
                                         color='green', linewidth=1.0, linestyle='--', label='triplet_loss')
        center_loss_line, = subfig.plot(loss_x_axis, c_loss,
                                        color='blue', linewidth=1.0, linestyle='--', label='center_loss')
        mask_loss_line, = subfig.plot(loss_x_axis, m_loss,
                                      color='black', linewidth=1.0, linestyle='--', label='mask_loss')

        subfig.legend(handles=[train_loss_line, softmax_loss_line, triplet_loss_line, center_loss_line, mask_loss_line],
                      labels=['train_loss', 'softmax_loss', 'triplet_loss', 'center_loss', 'mask_loss'],
                      loc='best')

        subfig = fig.add_subplot(1, 4, 2)
        subfig.set_title('rank-n')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('percentage')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        rank_1_line, = subfig.plot(eval_x_axis, rank[:, 0], color='blue', linewidth=1,
                                   linestyle='-', label='rank_1')
        rank_5_line, = subfig.plot(eval_x_axis, rank[:, 1], color='green', linewidth=1,
                                   linestyle='-', label='rank_5')
        rank_10_line, = subfig.plot(eval_x_axis, rank[:, 2], color='yellow', linewidth=1,
                                    linestyle='-', label='rank_10')
        rank_20_line, = subfig.plot(eval_x_axis, rank[:, 3], color='red', linewidth=1,
                                    linestyle='-', label='rank_20')

        subfig.legend(handles=[rank_1_line, rank_5_line, rank_10_line, rank_20_line],
                      labels=['rank_1', 'rank_5', 'rank_10', 'rank_20'], loc='best')

        subfig = fig.add_subplot(1, 4, 3)
        subfig.set_title('mAP')
        subfig.set_xlabel('epoch-No.')
        subfig.set_ylabel('value')
        subfig.set_xticks(eval_x_axis)

        subfig.set_xticklabels(eval_x_label)

        mAP_line, = subfig.plot(eval_x_axis, mAP, color='red', linewidth=1,
                                linestyle='-', label='mAP')

        subfig.legend(handles=[mAP_line, ],
                      labels=['mAP'], loc='best')

        subfig = fig.add_subplot(1, 4, 4)
        subfig.set_title('learning rate')
        subfig.set_xlabel('epoch')
        subfig.set_ylabel('value')

        lr_line, = subfig.plot(loss_epoch, lr, color='red', linewidth=1, linestyle='-', label='learning_rate')

        subfig.legend(handles=[lr_line, ],
                      labels=['learning_rate'], loc='best')

        if ifSave:
            fig.savefig(SaveDir)
        plt.show()

        # report max value
        max_rank_1_idx = np.argmax(rank[:, 0])  # only return first max value index
        max_rank_1_value = rank[:, 0][max_rank_1_idx]
        max_rank_1_ep = eval_epoch[max_rank_1_idx]

        max_mAP_idx = np.argmax(mAP)  # only return first max value index
        max_mAP_value = mAP[max_mAP_idx]
        max_mAP_ep = eval_epoch[max_mAP_idx]

        print('MAX rank_1: {} EP: {}'.format(max_rank_1_value,max_rank_1_ep))
        print('MAX mAP: {} EP: {}'.format(max_mAP_value, max_mAP_ep))
        return max_rank_1_ep, max_mAP_ep


plot_loss_acc_lr_MIE(eval_dir='./MIE/acc-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-MK-0.8-0.2-0.001-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET.txt',
                     loss_lr_dir='./MIE/loss_lr-CS-OP(Ly2)-STN(3-ImageNet1.0-SBP)-AF-AM8-CR2048-SGD-REA-TR-CT-MK-0.8-0.2-0.001-OPEXlr(decay)0.001-OPEXfz0-STNlr(wb_decay)1e-05-STNfz90-4080-CH-OP-lr_0.0001-WT4.0-DP0.6-ZP0.3-SBPLy2_256x128_59.pthMARKET.txt')

