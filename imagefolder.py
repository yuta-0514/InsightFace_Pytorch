import os
import shutil


images = []
labels = []
with open('/mnt/ms1mv2/ms1mv2_path.txt') as f:
    for line in f:
        sample = line.strip('\n').split(' ')
        images.append(sample[0])
        labels.append(sample[1])


for i, label in enumerate(labels):
    folder_dir = os.path.join('/mnt/test/' + str(label))
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    # copy image + rename
    image = os.path.join('/mnt/ms1mv2/ms1mv2/', images[i])
    shutil.copy(image, folder_dir)
