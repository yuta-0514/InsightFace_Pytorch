import datetime
import os
import argparse
import cv2
import numpy as np
import sklearn
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from backbones import get_model


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def evaluate(embeddings, actual_issame):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]

    sims = []
    labels = []
    for i in range(len(embeddings1)):
        sim = cosin_metric(embeddings1[i], embeddings2[i])
        sims.append(sim)
        labels.append(actual_issame[i])
    acc, th = cal_accuracy(sims, labels)

    return acc

@torch.no_grad()
def test(data_set, backbone, batch_size):
    print('testing verification..')
    data_list = data_set[0]
    issame_list = data_set[1]
    embeddings_list = []
    time_consumed = 0.0
    for i in range(len(data_list)):
        data = data_list[i]
        embeddings = None
        ba = 0
        while ba < data.shape[0]:
            bb = min(ba + batch_size, data.shape[0])
            count = bb - ba
            _data = data[bb - batch_size: bb]
            time0 = datetime.datetime.now()
            img = ((_data / 255) - 0.5) / 0.5
            net_out: torch.Tensor = backbone(img)
            _embeddings = net_out.detach().cpu().numpy()
            time_now = datetime.datetime.now()
            diff = time_now - time0
            time_consumed += diff.total_seconds()
            if embeddings is None:
                embeddings = np.zeros((data.shape[0], _embeddings.shape[1]))
            embeddings[ba:bb, :] = _embeddings[(batch_size - count):, :]
            ba = bb
        embeddings_list.append(embeddings)

    _xnorm = 0.0
    _xnorm_cnt = 0
    for embed in embeddings_list:
        for i in range(embed.shape[0]):
            _em = embed[i]
            _norm = np.linalg.norm(_em)
            _xnorm += _norm
            _xnorm_cnt += 1
    _xnorm /= _xnorm_cnt

    embeddings = embeddings_list[0].copy()
    embeddings = sklearn.preprocessing.normalize(embeddings)
    embeddings = embeddings_list[0] + embeddings_list[1]
    embeddings = sklearn.preprocessing.normalize(embeddings)
    print(embeddings.shape)
    print('infer time', time_consumed)
    accuracy = evaluate(embeddings, issame_list)
    return accuracy, _xnorm, embeddings_list


def get_lfw_list(pair_list):
    with open(pair_list, 'r') as fd:
        pairs = fd.readlines()
    data_list = []
    labels = []
    for pair in pairs:
        splits = pair.split()
        data_list.append(splits[0])
        data_list.append(splits[1])
        if splits[2] == 1:
            labels.append(True)
        else:
            labels.append(False)
    return data_list, labels


def load_image(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, dsize=(112, 112))
    if image is None:
        return None
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :, :, :]
    image = image.astype(np.float32, copy=False)
    image -= 127.5
    image /= 127.5
    return image


def load_data(test_list, labels):
    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(test_list), 3, 112, 112))
        data_list.append(data)
    for i, image_path in enumerate(test_list):
        image = load_image(image_path)
        for flip in [0, 1]:
            if flip == 1:
                image = np.fliplr(image)
            data_list[flip][i][:] = torch.from_numpy(image.copy())
    return data_list, labels


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='do verification')
    # general
    parser.add_argument('--data-dir', default='', help='')
    parser.add_argument('--prefix',
                        default='/mnt/weight/backbone.pth',
                        help='path to load model.')
    parser.add_argument('--target',
                        default='lfw,cfp_ff,cfp_fp,agedb_30',
                        help='test targets.')
    parser.add_argument('--batch_size', default=64, type=int, help='batch_size')
    args = parser.parse_args()

    weight = torch.load(args.prefix)
    model = get_model('r50', dropout=0, fp16=False).cuda()
    model.load_state_dict(weight)
    model = torch.nn.DataParallel(model)

    identity_list, labels = get_lfw_list('/mnt/lfw/pairs.txt')
    img_paths = [os.path.join('/mnt/lfw', each) for each in identity_list]
    data_set = load_data(img_paths, labels)

    acc, xnorm, embeddings_list = test(
        data_set, model, args.batch_size)
    print('LFW XNorm: %f' % (xnorm))
    print('LFW Accuracy: %1.5f' % (acc))
