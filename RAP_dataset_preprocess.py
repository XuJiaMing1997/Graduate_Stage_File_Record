import os
import time
import matplotlib.pyplot as plt
import math
import numpy as np
import h5py
import shutil
import PIL
import PIL.Image as Image

RAP_H5_Dir = '../VIE/RAP_ALL.hdf5'



def generate_folder_from_H5(H5_Dir =  '../Spur/RAP/RAP_ALL.hdf5', Folder_Dir = '../Generated_RAP',
                            if_drop_distractor = True, if_drop_unlabeled = False,
                            target_size = [256,128], if_generate = False):
    # return fname_list, orient_label_list
    # Notice: orient [0,1,2]
    # folder fname: imgid_pid_cam_orient  eg. 1_-2_01_2.jpg
    Folder_Dir = Folder_Dir + '_' + '{}x{}'.format(target_size[0],target_size[1])
    if os.path.exists(Folder_Dir):
        fname_list = os.listdir(Folder_Dir)
        print('{} include images: {}'.format(Folder_Dir,len(fname_list)))
        label_list = []
        for fname in fname_list:
            fname_split = fname.split('_')
            label_list.append(int(fname_split[3][0]))
        return fname_list, label_list
    else:
        os.makedirs(Folder_Dir)
        print('create {}'.format(Folder_Dir))

    hf = h5py.File(H5_Dir,'r')
    image_all = hf['image']
    pid_all = hf['person_id']
    cam_all = hf['camera']
    view_label_all = hf['view']
    print('image shape: {0}'.format(image_all.shape)) # (84928, 128, 64, 3)
    print('person_id shape: {0}'.format(pid_all.shape)) # (84928,)
    print('camera shape: {0}'.format(cam_all.shape))
    print('view_label shape: {0}'.format(view_label_all.shape))

    label = np.array(view_label_all) # or np.where will not work !!!!!
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

    original_image_all = np.array(image_all)
    original_pid_all = np.array(pid_all)
    original_cam_all = np.array(cam_all)
    original_label_all = label

    hf.close()

    distractor_loc = []
    unlabeled_loc = []
    if if_drop_distractor:
        distractor_loc = np.where(original_pid_all == -1)[0].tolist()
        print('dixtractor  -1 num: {0}'.format(len(distractor_loc)))
    if if_drop_unlabeled:
        unlabeled_loc = np.where(original_pid_all == -2)[0].tolist()
        print('unlabeled  -2 num: {0}'.format(len(unlabeled_loc)))

    drop_list = distractor_loc + unlabeled_loc
    current_num = len(original_label_all) - len(drop_list)
    print('rest num: {0}'.format(current_num))

    new_fname_list = []
    new_label_list = []
    drop_set = set(drop_list)
    file_id_count = 1
    for idx,(img,pid,cam,orient) in enumerate(zip(
            original_image_all,original_pid_all,original_cam_all,original_label_all)):
        if idx in drop_set:
            continue
        file_name = '{}_{}_{}_{}.png'.format(file_id_count,pid,cam,orient)
        file_id_count += 1
        new_fname_list.append(file_name)
        new_label_list.append(orient)
        if if_generate:
            if idx % 1000 == 0:
                print('current img {}'.format(idx))
            recv_img = Image.fromarray(img)
            recv_img = recv_img.resize((target_size[1],target_size[0]),resample=PIL.Image.BILINEAR)
            recv_img.save(os.path.join(Folder_Dir,file_name))

    label = np.array(new_label_list) # or np.where will not work !!!!!
    count_front = len(np.where(label == 0)[0])
    count_back = len(np.where(label == 1)[0])
    count_side = len(np.where(label == 2)[0])
    print('image num: {0}'.format(len(label)))
    print('percentage for each view:')
    print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
    print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
            count_front / float(len(label)), count_back / float(len(label)), count_side / float(len(label))
    ))
    return new_fname_list, new_label_list


# generate_folder_from_H5(RAP_H5_Dir,'../RAP_Dataset',if_drop_distractor=True,if_drop_unlabeled=False,
#                         target_size=[256,128],if_generate=False)


def RAP_folder_annotation_quality_check(check_num = 20, FolderDir ='../Spur/RAP/my_RAP_dataset_256x128'):
    fname_list = os.listdir(FolderDir)
    orientation_dict = {0:'Front',1:'Back',2:'Side'}
    for ct in range(check_num):
        target = np.random.randint(0,len(fname_list)+1)
        fname = fname_list[target]
        fname_split = fname.split('_')
        fileid = int(fname_split[0])
        pid = int(fname_split[1])
        cam = int(fname_split[2])
        orient = int(fname_split[3][0])

        img = Image.open(os.path.join(FolderDir,fname))
        plt.imshow(img)
        plt.title('fileid: {0}\nid: {1} cam: {2}\norientation: {3}'.format(fileid,pid,cam,
                                                                           orientation_dict[orient]))
        plt.show()

RAP_folder_annotation_quality_check(50, '../RAP_Dataset_256x128')



def drop_part_from_full_image_folder(FolderDir = './RAP',if_drop_distractor = True, if_drop_unlabeled = False,
                                     if_del = False):
    # fname: imgid_pid_cam_orient  eg. 1_-2_01_0.jpg
    # Notice: orient [0,1,2]
    # distractor: pid == -1  unlabeled: pid == -2

    drop_fname_list = []
    distractor_list = []
    unlabeled_list = []
    orientation_list = []
    fname_list = os.listdir(FolderDir)
    for fname in fname_list:
        fname_split = fname.split('_')
        pid = int(fname_split[1])
        orient = int(fname_split[3][0])
        orientation_list.append(orient)
        if pid == -1:
            distractor_list.append(fname)
        if pid == -2:
            unlabeled_list.append(fname)

    orientation_array = np.array(orientation_list)
    # count_front = len(np.where(orientation_array == 1)[0])
    # count_back = len(np.where(orientation_array == 2)[0])
    # count_left = len(np.where(orientation_array == 3 )[0])
    # count_right = len(np.where(orientation_array == 4)[0])
    # count_side = count_left + count_right
    count_front = len(np.where(orientation_array == 0)[0])
    count_back = len(np.where(orientation_array == 1)[0])
    count_side = len(np.where(orientation_array == 2 )[0])


    print('current folder images num: {}'.format(len(fname_list)))
    assert len(fname_list) == 84928 # fail means not full folder

    print('percentage for each view:')
    print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
    print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
            count_front / float(len(orientation_list)), count_back / float(len(orientation_list)),
            count_side / float(len(orientation_list))))


    print('distractor images num: {}'.format(len(distractor_list)))
    print('unlabeled images num: {}'.format(len(unlabeled_list)))

    if if_drop_distractor:
        drop_fname_list += distractor_list
    if if_drop_unlabeled:
        drop_fname_list += unlabeled_list
    print('waiting to drop: {} rest: {}'.format(len(drop_fname_list),len(fname_list) - len(drop_fname_list)))

    drop_fname_set = set(drop_fname_list)
    orientation_list = []
    for fname in fname_list:
        if fname in drop_fname_set:
            continue
        fname_split = fname.split('_')
        pid = int(fname_split[1])
        orient = int(fname_split[3][0])
        orientation_list.append(orient)

    orientation_array = np.array(orientation_list)
    # count_front = len(np.where(orientation_array == 1)[0])
    # count_back = len(np.where(orientation_array == 2)[0])
    # count_left = len(np.where(orientation_array == 3 )[0])
    # count_right = len(np.where(orientation_array == 4)[0])
    # count_side = count_left + count_right
    count_front = len(np.where(orientation_array == 0)[0])
    count_back = len(np.where(orientation_array == 1)[0])
    count_side = len(np.where(orientation_array == 2 )[0])

    print('After Drop: percentage for each view:')
    print('Front: {0} Back: {1} Side: {2}'.format(count_front, count_back, count_side))
    print('Front: {0:.2%} Back: {1:.2%} Side: {2:.2%}'.format(
            count_front / float(len(orientation_list)), count_back / float(len(orientation_list)),
            count_side / float(len(orientation_list))))


    if if_del:
        for del_fname in drop_fname_list:
            os.remove(os.path.join(FolderDir,del_fname))
        fname_list = os.listdir(FolderDir)
        print('after del folder images num: {}'.format(len(fname_list)))



# drop_part_from_full_image_folder(FolderDir='../Spur/RAP/my_RAP_dataset_256x128',if_drop_distractor=True,if_drop_unlabeled=False,
#                                  if_del=False)

# Raw: 84928
# Front: 19678 Back: 20651 Side: 44599
# Front: 23.17% Back: 24.32% Side: 52.51%

# Drop distractor, Rest: 69981
# Front: 15042 Back: 15528 Side: 39411
# Front: 21.49% Back: 22.19% Side: 56.32%

# Drop unlabeled, Rest: 41585
# Front: 13270 Back: 13892 Side: 14423
# Front: 31.91% Back: 33.41% Side: 34.68%

# Drop both, Rest: 26638
# Front: 8634 Back: 8769 Side: 9235
# Front: 32.41% Back: 32.92% Side: 34.67%

















