import numpy as np
import os
import time
import matplotlib.pyplot as plt

def lr_scheduler(current_epoch, scheduler_setting):
    # PASS!
    # scheduler_setting = {'lr_start':0.1,
    #                      'decay_rate':10,
    #                      'warmup_rate':100
    #                      }
    # one epoch call once
    # !!!!!!!!!  STN independent lr may occur problems for UnKnow which var is belong to STN !!!!!!!!!!
    if current_epoch == 0:
        factor = 1. / scheduler_setting['warmup_rate']
    elif current_epoch <= 10:
        factor = float(current_epoch) / 10
    elif current_epoch <= 40:
        factor = 1
    elif current_epoch <= 70:
        factor = 1. / scheduler_setting['decay_rate']
    else:
        factor = 1. / float(scheduler_setting['decay_rate'] ** 2)

    return scheduler_setting['lr_start'] * factor

# [0,119]
scheduler_setting = {'lr_start':1,
                     'decay_rate':10,
                     'warmup_rate':100
                     }
recvlr = []
for i in range(120):
    recvlr.append(lr_scheduler(i,scheduler_setting))
x = np.arange(120)
plt.figure()
line, = plt.plot(x,recvlr)
plt.legend(handles=[line,],labels=['lr'],loc='best')
plt.show()



