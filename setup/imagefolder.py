import os
import shutil


images = []
labels = []
with open('/mnt/faces_umd/faces_umd_path.txt') as f:
    for line in f:
        sample = line.strip('\n').split(' ')
        images.append(sample[0])
        labels.append(sample[1])


for i, label in enumerate(labels):
    folder_dir = os.path.join('/mnt/test/' + str(label))
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    # copy image + rename
    image = os.path.join('/mnt/faces_umd/faces_umd/', images[i])
    shutil.copy(image, folder_dir)

# import glob
# cnt = glob.glob('/mnt/umd_face/*/*.jpg')
# import pdb
# pdb.set_trace()