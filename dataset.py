import os
import os.path as osp

import chainer
import numpy as np
import scipy.misc
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from skimage.io import imread


class TestVae(chainer.dataset.DatasetMixin):

    #mean_bgr = np.array((104.00698793, 116.66876762, 122.67891434))
    mean_d = np.array((127, 127, 127))
    mean_hand = np.array((127))

    def __init__(self, split, return_image=False, return_all=False):
        assert split in ('train', 'val')
        ids = self._get_ids()
        iter_train, iter_val = train_test_split(
            ids, test_size=0.2, random_state=np.random.RandomState(1234))
        self.ids = iter_train if split == 'train' else iter_val
        self._return_image = return_image
        self._return_all = return_all

    def __len__(self):
        return len(self.ids)

    def _get_ids(self):
        ids = []
        dataset_dir = chainer.dataset.get_dataset_directory(
            '2019_11_28_pr2')
        for data_id in os.listdir(dataset_dir):
            ids.append(osp.join('cloth', data_id))
        return ids

    def img_to_datum(self, img):
        img = img.copy()
        datum = img.astype(np.float32)
        #datum = datum[:, :, ::-1]  # RGB -> BGR
        #datum -= self.mean_bgr
        #datum = datum.transpose((2, 0, 1))
        return datum

    def get_example(self, i):
        ann_id, data_id = self.ids[i].split('/')
        assert ann_id in ('cloth')

        dataset_dir = chainer.dataset.get_dataset_directory(
            '2019_11_28_pr2')

        img_file_rgb = osp.join(dataset_dir, data_id, 'image.png')
        img_rgb = imread(img_file_rgb)
        img = resize(img_rgb, (img_rgb.shape[0]/10, img_rgb.shape[1]/10))
        img = 255* img
        rescaled_img = img.astype(np.uint8)
        datum = self.img_to_datum(rescaled_img)
        datum = datum.flatten()
        if self._return_image:
            return rescaled_img
        else:
            return datum


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = TestVae('train', return_image=True)
    for i in range(len(dataset)):
        img_rgb = dataset.get_example(i)
        #img_rgb = img_rgb.reshape(48, 64, 3)
        #img_rgb = img_rgb[:,:,::-1]
        plt.imshow(img_rgb)
        plt.show()
