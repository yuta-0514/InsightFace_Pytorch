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


train_set = MXFaceDataset('/mnt/faces_webface_112x112', 0)


from PIL import Image
f = open('faces_webface.txt', 'w')

for i in range(len(train_set)):
    img, label = train_set[i]
    pil_img = Image.fromarray(img)
    pil_img.save('./faces_webface/{}.jpg'.format(i))
    f.write('{}.jpg {}\n'.format(i, label))
    if i % 1000 == 0:
        print('Complete {}images'.format(i))

f.close()


# Write Image_path
'''
import glob
import re

datas=[]

with open('/mnt/faces_umd/faces_umd_path.txt')as f:
    for data in f:
        data = data.split()

        datas.append(data[1])


mask_path = []
files = glob.glob("/mnt/faces_umd/faces_umd_masked/*")

for file in files:
    mask_path.append(file)


mask_path = sorted(mask_path, key=lambda s: int(re.search(r'\d+', s).group()))

f = open('faces_umd_masked_path.txt', 'w')
for i in range(len(mask_path)):
    f.write(mask_path[i] + ' ' + datas[i] + '\n')
f.close()
'''
