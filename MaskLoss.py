import torch
import torch.nn as nn

def binarylization(matrix, threshold = 0.5, use_GPU = False):
    zero_fill = torch.zeros_like(matrix)
    if use_GPU:
        zero_fill = zero_fill.cuda()
    out = torch.where(matrix < threshold, zero_fill, matrix) # fill zero
    out_data = out.data.reshape((-1,))
    for idx in range(len(out_data)):
        if out_data[idx] > 0:
            out_data[idx] = 1 # without fill 1 may bring uncertain optimization target
    return out

class LocLoss(object):
    def __init__(self, threshold = 0.7, use_GPU = False):
        '''
        Bigger threshold means less pixel are selected to calculate overlap, e.g. Bigger T Easier Loss

        :param threshold:
        :return:
        '''
        self.threshold = threshold
        self.use_GPU = use_GPU

    def __call__(self, masks_list):
        # masks_list (list): [mask_num, torch.Size([B, 1, H, W])]
        masks = torch.cat(masks_list,dim=1) # torch.Size([B, mask_num, H, W])
        binary_masks = binarylization(masks, threshold=self.threshold, use_GPU=self.use_GPU)
        spatial_wise_cube = binary_masks.reshape((masks.shape[0],masks.shape[1],-1)).permute((2,0,1)) # [HW,B,mask_num]
        spatial_wise_cube_sum = spatial_wise_cube.sum(dim=2,keepdim=False) # sum value: [0,8]
        select_mask =spatial_wise_cube_sum.gt(1.) # 1 suitable ???
        select_value = torch.masked_select(spatial_wise_cube_sum,select_mask)
        loss = select_value.sum() / masks.shape[0]
        return loss, binary_masks


class AreaLoss(object):
    def __init__(self, binary_threshold = 0.7, area_constrain_proportion = 0.3, use_GPU = False):
        self.binary_threshold = binary_threshold
        self.use_GPU = use_GPU
        self.area_constrain_proportion = area_constrain_proportion

    def __call__(self, masks_list):
        masks = torch.cat(masks_list,dim=1) # torch.Size([B, mask_num, H, W])
        binary_masks = binarylization(masks, threshold=self.binary_threshold, use_GPU=self.use_GPU)
        area_cube = binary_masks.reshape((masks.shape[0],masks.shape[1],-1)) # [B,mask_num,HW]
        area_cube_sum = area_cube.sum(dim=2,keepdim=False) # area value: [0,H*W]
        area_constrain = int(self.area_constrain_proportion * masks.shape[-1] * masks.shape[-2])
        # print('area_constrain: {}'.format(area_constrain))
        select_mask = area_cube_sum.ge(area_constrain)
        select_value = torch.masked_select(area_cube_sum,select_mask)
        loss = select_value.sum() / masks.shape[0]
        return loss, binary_masks


if __name__ == '__main__':
    if torch.cuda.is_available():
        use_GPU = True
        print('Use GPU')
    else:
        use_GPU = False
        print('No GPU, Use CPU')
    available_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    m1 = [[0.1,0.2,0.1,0.5,0.8,0.8],
          [0.1,0.2,0.3,0.5,0.5,0.8],
          [0.1,0.0,0.0,0.2,0.6,0.6],
          [0.0,0.1,0.4,0.2,0.5,0.1],
          [0.1,0.0,0.0,0.2,0.2,0.3],
          [0.1,0.0,0.0,0.2,0.3,0.1]]

    m2 = [[0.1,0.2,0.5,0.5,0.4,0.0],
          [0.1,0.5,0.7,0.5,0.5,0.0],
          [0.1,0.6,0.9,0.9,0.4,0.0],
          [0.0,0.5,0.8,0.7,0.5,0.1],
          [0.1,0.3,0.5,0.6,0.9,0.3],
          [0.1,0.2,0.5,0.3,0.2,0.1]]

    m3 = [[0.0,0.2,0.4,0.5,0.2,0.0],
          [0.0,0.2,0.8,0.7,0.5,0.0],
          [0.2,0.6,0.7,0.5,0.6,0.4],
          [0.4,0.7,0.4,0.2,0.5,0.5],
          [0.7,0.8,0.4,0.2,0.7,0.6],
          [0.7,0.9,0.3,0.5,0.8,0.6]]


    m1 = torch.tensor(m1,dtype=torch.float32,requires_grad=True,device=available_device)
    m2 = torch.tensor(m2,dtype=torch.float32,requires_grad=True,device=available_device)
    m3 = torch.tensor(m3,dtype=torch.float32,requires_grad=True,device=available_device)
    one1 = torch.ones((1,6,6),dtype = torch.float32,requires_grad=True,device=available_device)
    one2 = torch.ones((1,6,6),dtype = torch.float32,requires_grad=True,device=available_device)
    zero1 = torch.zeros((1,6,6),dtype = torch.float32,requires_grad=True,device=available_device)

    print(m1.is_leaf)
    print(m2.is_leaf)
    print(m3.is_leaf)
    print(one1.is_leaf)
    print(one2.is_leaf)
    print(zero1.is_leaf)

    # if use_GPU:
    #     mm1 = mm1.cuda()
    #     mm2 = mm2.cuda()
    #     mm3 = mm3.cuda()
    #     one1 = one1.cuda()
    #     one2 = one2.cuda()
    #     zero1 = zero1.cuda()


    tem_m1 = m1.unsqueeze(0)
    tem_m2 = m2.unsqueeze(0)
    tem_m3 = m3.unsqueeze(0)
    branch_1 = torch.stack((tem_m1,one1),dim=0) # Batch 2, Mask_num 3
    branch_2 = torch.stack((tem_m2,zero1),dim=0)
    branch_3 = torch.stack((tem_m3,one2),dim=0)
    original_masks = torch.cat((branch_1,branch_2,branch_3),dim=1)

    Binary_Threshold = 0.7
    Area_Proportion = 0.3
    # my_loss = MaskLoss(Binary_Threshold,use_GPU=use_GPU)
    my_loss = AreaLoss(Binary_Threshold,Area_Proportion,use_GPU=use_GPU)
    my_optimizer = torch.optim.SGD([m1,m2,m3,one1,one2,zero1], lr =0.01, momentum=0.9)

    for Iter in range(100):
        my_optimizer.zero_grad()
        mm1 = m1.unsqueeze(0)
        mm2 = m2.unsqueeze(0)
        mm3 = m3.unsqueeze(0)
        branch_1 = torch.stack((mm1,one1),dim=0) # Batch 2, Mask_num 3
        branch_2 = torch.stack((mm2,zero1),dim=0)
        branch_3 = torch.stack((mm3,one2),dim=0)

        masks_list = [branch_1,branch_2,branch_3]
        # loss = my_loss(masks_list)[0]
        loss, recv_bmasks = my_loss(masks_list)
        loss.backward()
        print('Iter: {} loss: {}'.format(Iter,loss.item()))
        if loss.item() == 0:
            break
        my_optimizer.step()

    print('----------> Mask Comparison <----------')
    original_masks_binary = binarylization(original_masks,Binary_Threshold)
    for idx,(batch_orign, batch_new) in enumerate(zip(original_masks_binary,recv_bmasks)):
        print('Batch {}'.format(idx+1))
        for idx,(orignM, newM) in enumerate(zip(batch_orign,batch_new)):
            print('branch{}'.format(idx+1))
            print('----> Original\n{}'.format(orignM))
            print('------------> New\n{}'.format(newM))


    print('----------> Value Comparison <----------')
    tem_m1 = m1.unsqueeze(0)
    tem_m2 = m2.unsqueeze(0)
    tem_m3 = m3.unsqueeze(0)
    branch_1 = torch.stack((tem_m1,one1),dim=0) # Batch 4, Mask_num 3
    branch_2 = torch.stack((tem_m2,zero1),dim=0)
    branch_3 = torch.stack((tem_m3,one2),dim=0)
    new_masks = torch.cat((branch_1,branch_2,branch_3),dim=1)
    for idx,(batch_orign, batch_new) in enumerate(zip(original_masks,new_masks)):
        print('Batch {}'.format(idx+1))
        for idx,(orignM, newM) in enumerate(zip(batch_orign,batch_new)):
            print('branch{}'.format(idx+1))
            print('----> Original\n{}'.format(orignM))
            print('------------> New\n{}'.format(newM))


