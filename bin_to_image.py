import numbers
import os

import mxnet as mx
import numpy as np
import torch
from torch.utils.data import Dataset


class MXFaceDataset(Dataset):
    def __init__(self, root_dir, local_rank):
        super(MXFaceDataset, self).__init__()
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        return sample, label

    def __len__(self):
        return len(self.imgidx)


train_set = MXFaceDataset('/mnt/train_tmp/faces_emore', 0)

# len(train_set) -> 5822653


from PIL import Image
f = open('ms1mv2_path.txt', 'w')

for i in range(len(train_set)):
    img, label = train_set[i]
    pil_img = Image.fromarray(img)
    pil_img.save('./ms1mv2/{}.jpg'.format(i))
    f.write('{}.jpg {}\n'.format(i, label))
    if i % 1000 == 0:
        print('Complete {}images'.format(i))

f.close()

