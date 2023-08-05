import os
import shutil


images = []
labels = []
with open('./faces_webface.txt') as f:
    for line in f:
        sample = line.strip('\n').split(' ')
        images.append(sample[0])
        labels.append(sample[1])


for i, label in enumerate(labels):
    folder_dir = os.path.join('/mnt/faces_webface_imagefolder_u2net', str(label))
    if not os.path.isdir(folder_dir):
        os.mkdir(folder_dir)
    # copy image + rename
    image = os.path.join('/mnt/Faces_webface/faces_webface_u2net', images[i])
    image = image.replace("jpg", "png")
    shutil.copy(image, folder_dir)
    if i % 1000 == 0:
        print('Complete {}images'.format(i))
