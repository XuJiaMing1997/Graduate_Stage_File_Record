import torch

# different Dataloader use same collate function seems will cause Wrong Operation !!!!!

def collate_fn_OP_1(batch):
    imgs, labels, _ = zip(*batch)
    labels = torch.tensor(labels,dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels

def collate_fn_OP_2(batch):
    imgs, labels, _ = zip(*batch)
    labels = torch.tensor(labels,dtype=torch.int64)
    return torch.stack(imgs, dim=0), labels