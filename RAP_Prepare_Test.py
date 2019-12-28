import os
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import h5py

RAP_H5_Dir = '../Spur/RAP/RAP_ALL.hdf5'
hf = h5py.File(RAP_H5_Dir,'r')
image_all = hf['image']
pid_all = hf['person_id']
cam_all = hf['camera']
view_label_all = hf['view']
print('image shape: {0}'.format(image_all.shape)) # (84928, 128, 64, 3)
print('person_id shape: {0}'.format(pid_all.shape)) # (84928,)
print('camera shape: {0}'.format(cam_all.shape))
print('view_label shape: {0}'.format(view_label_all.shape))

label = np.array(view_label_all)
count_front = len(np.where(label == 0)[0])
count_back = len(np.where(label == 1)[0])
count_side = len(np.where(label == 2)[0])
print('image num: {0}'.format(len(label)))
print('percentage for each view:')
print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
    count_front / float(len(label)), count_back / float(len(label)), count_side / float(len(label))
))
# Front: 19678 Back: 20651 Side: 44599
# Front: 23.17% Back: 24.32% Side: 52.51%


# if del distractor --------------  percentage?
# original_image_all = np.array(image_all)
original_pid_all = np.array(pid_all)
original_cam_all = np.array(cam_all)
original_label_all = np.array(view_label_all)

hf.close()

distractor_loc = np.where(original_pid_all == -1)[0]
print('dixtractor  -1 num: {0}'.format(len(distractor_loc))) # 14947
current_num = len(original_label_all) - len(distractor_loc)
print('current num: {0}'.format(current_num))

unlabeled_loc = np.where(original_pid_all == -2)[0]
print('unlabeled  -2 num: {0}'.format(len(unlabeled_loc))) # 43343
current_num = len(original_label_all) - len(distractor_loc) - len(unlabeled_loc)
print('current num: {0}'.format(current_num))

distractor_loc =  list(distractor_loc) + list(unlabeled_loc)
# distractor_loc =  list(distractor_loc)
# #####
# run fast !!!!!! Good !!!!!!!!
# also waste time when add unlabeled data !!!!!!!!!!

distractor_loc.sort() # !!!!!!!!!!!!!!!!!!!!  very very important !!!!!! save time !!!!!!!!

new_index = []
for idx in range(len(original_label_all)):
    if idx == distractor_loc[0]:
        distractor_loc.pop(0)
        if len(distractor_loc) == 0: # Here may get BUG !!!!!! at edge situation
            new_index.extend(list(range(idx+1,len(original_label_all))))
            break
        else:
            continue
    else:
        new_index.append(idx)

# another choice
# ???????? very very slow  why ?????????
# new_index = list(np.arange(len(original_label_all)))
# for idx,i in enumerate(range(len(distractor_loc))):
#     if idx % 1000 == 0:
#         print('current: {0}'.format(idx))
#     new_index.remove(distractor_loc[i])
# ######

# image_all = original_image_all[new_index]
pid_all = original_pid_all[new_index]
cam_all = original_cam_all[new_index]
view_label_all = original_label_all[new_index]


# print('if equal: {0}'.format(len(image_all) == current_num))
count_front = len(np.where(view_label_all == 0)[0])
count_back = len(np.where(view_label_all == 1)[0])
count_side = len(np.where(view_label_all == 2)[0])
print('image num: {0}'.format(len(view_label_all)))
print('percentage for each view:')
print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
    count_front / float(len(view_label_all)), count_back / float(len(view_label_all)),
    count_side / float(len(view_label_all))
))





