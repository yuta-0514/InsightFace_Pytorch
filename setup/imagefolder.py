import os
import shutil


images = []
labels = []
with open('/mnt/CelebA/identity_CelebA.txt') as f:
    for line in f:
        sample = line.strip('\n').split(' ')
        images.append(sample[0])
        labels.append(sample[1])


for i, label in enumerate(labels):
    folder_dir = os.path.join('/mnt/CelebA/' + str(label))
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    # copy image + rename
    image = os.path.join('/mnt/data/CelebA/img_celeba/', images[i])
    shutil.copy(image, folder_dir)

# import glob
# cnt = glob.glob('/mnt/CelebA/*/*.jpg')
# import pdb
# pdb.set_trace()
