import numpy as np
import PIL.Image as Image
import torchvision.transforms.functional as F


class RandomZoomOutPad(object):
    def __init__(self,size = [256,128], ZoomOutMin = 0.5, Probability = 0.5, TopPadMode = True):
        self.size = size
        self.zoom_min = ZoomOutMin
        self.probability = Probability
        self.top_pad_mode = TopPadMode
        assert ZoomOutMin <= 1.0, 'Zoom out minimum value must lower than 1.0'

    def __call__(self, img):
        # resize to random scale in ZoomRange
        # compute pad
        # pad to target size
        probability_split = 100 * self.probability
        random_point = np.random.randint(0,100)
        if random_point < probability_split:
            # zoom out
            random_scale = np.random.uniform(self.zoom_min, 1.0)
            resize_H = int(random_scale * self.size[0])
            resize_W = int(random_scale * self.size[1])
            recv_img = F.resize(img, [resize_H,resize_W], Image.BILINEAR)

            pad_all_H = self.size[0] - resize_H
            pad_all_W = self.size[1] - resize_W

            if self.top_pad_mode:
                # Bottom Align Mode
                pad_top = pad_all_H
                pad_down = 0
                pad_left = int(pad_all_W // 2)
                pad_right = pad_all_W - pad_left
            else:
                # Center Mode
                pad_top = int(pad_all_H // 2)
                pad_down = pad_all_H - pad_top
                pad_left = int(pad_all_W // 2)
                pad_right = pad_all_W - pad_left

            out = F.pad(recv_img,(pad_left,pad_top,pad_right,pad_down))
            # left, top, right and bottom
        else:
            # raw
            out = F.resize(img, self.size, Image.BILINEAR)
        return out