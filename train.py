import argparse
import logging
import os

import numpy as np
import torch
from torch import distributed
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from backbones import get_model
from dataset import get_dataloader
from losses import CombinedMarginLoss
from lr_scheduler import PolyScheduler
from partial_fc import PartialFC, PartialFCAdamW
from utils.utils_callbacks import CallBackLogging, CallBackVerification
from utils.utils_config import get_config
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_distributed_sampler import setup_seed

#SCOPS
from torchvision import transforms
import loss_scops
import utils_scops
from model.feature_extraction import FeatureExtraction, featureL2Norm
from tps.rand_tps import RandTPS

assert torch.__version__ >= "1.9.0", "In order to enjoy the features of the new torch, \
we have upgraded the torch to 1.9.0. torch before than 1.9.0 may not work in the future."

try:
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    distributed.init_process_group("nccl")
except KeyError:
    world_size = 1
    rank = 0
    distributed.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:12584",
        rank=rank,
        world_size=world_size,
    )


IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)


class PartBasisGenerator(torch.nn.Module):
    def __init__(self, feature_dim, K, normalize=False):
        super(PartBasisGenerator, self).__init__()
        self.w = torch.nn.Parameter(
            torch.abs(torch.cuda.FloatTensor(K, feature_dim).normal_()))
        self.normalize = normalize

    def forward(self, x=None):
        out = torch.nn.ReLU()(self.w)
        if self.normalize:
            return featureL2Norm(out)
        else:
            return out


def lambda_scheduler(loss, loss_seg, epoch):
    if epoch <= 2:
        l = 0.1
    elif 2 < epoch < 10:
        l = 0.01
    else:
        l = 0
    total_loss = loss + l*loss_seg
    return total_loss


def main(args):

    # get config
    cfg = get_config(args.config)
    # global control random seed
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    torch.cuda.set_device(args.local_rank)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(rank, cfg.output)

    summary_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))
        if rank == 0
        else None
    )

    train_loader = get_dataloader(
        cfg.rec,
        args.local_rank,
        cfg.batch_size,
        cfg.dali,
        cfg.seed,
        cfg.num_workers
    )

    backbone = get_model(
        cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()

    backbone = torch.nn.parallel.DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[args.local_rank], bucket_cap_mb=16,
        find_unused_parameters=True)

    backbone.train()
    # FIXME using gradient checkpoint if there are some unused parameters will cause error
    backbone._set_static_graph()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold
    )

    if cfg.optimizer == "sgd":
        module_partial_fc = PartialFC(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)

    elif cfg.optimizer == "adamw":
        module_partial_fc = PartialFCAdamW(
            margin_loss, cfg.embedding_size, cfg.num_classes,
            cfg.sample_rate, cfg.fp16)
        module_partial_fc.train().cuda()
        opt = torch.optim.AdamW(
            params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}],
            lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size * world_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolyScheduler(
        optimizer=opt,
        base_lr=cfg.lr,
        max_steps=cfg.total_step,
        warmup_steps=cfg.warmup_step,
        last_epoch=-1
    )

    start_epoch = 0
    global_step = 0
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])
        del dict_checkpoint

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    callback_verification = CallBackVerification(
        val_targets=cfg.val_targets, rec_prefix=cfg.rec, summary_writer=summary_writer
    )
    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer
    )

    loss_am = AverageMeter()
    amp = torch.cuda.amp.grad_scaler.GradScaler(growth_interval=100)

    # SCOPS---------------------
    # Initialize spatial/color transform for Equuivariance loss.
    tps = RandTPS(cfg.input_size[1], cfg.input_size[0],
                  batch_size=cfg.batch_size,
                  sigma=cfg.tps_sigma,
                  border_padding=False, random_mirror=False,
                  random_scale=(cfg.random_scale_low, cfg.random_scale_high),
                  mode=cfg.tps_mode).cuda(cfg.gpu)

    # Color Transorm.
    cj_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.2),
        transforms.ToTensor(), ])

    # KL divergence loss for equivariance
    kl = torch.nn.KLDivLoss().cuda(cfg.gpu)

    # loss/ bilinear upsampling
    interp = torch.nn.Upsample(
        size=(cfg.input_size[1], cfg.input_size[0]), mode='bilinear', align_corners=True)

    # Initialize feature extractor and part basis for the semantic consistency loss.
    zoo_feat_net = FeatureExtraction(
        feature_extraction_cnn=cfg.ref_net, normalization=cfg.ref_norm, last_layer=cfg.ref_layer)
    zoo_feat_net.eval()

    part_basis_generator = PartBasisGenerator(zoo_feat_net.out_dim, cfg.num_parts, normalize=cfg.ref_norm)
    part_basis_generator.cuda(cfg.gpu)
    part_basis_generator.train()

    # Initialize optimizers.
    optimizer_sc = torch.optim.SGD(part_basis_generator.parameters(
    ), lr=cfg.learning_rate_w, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    optimizer_sc.zero_grad()
    #---------------------------

    for epoch in range(start_epoch, cfg.num_epoch):

        if isinstance(train_loader, DataLoader):
            train_loader.sampler.set_epoch(epoch)
        for _, (img, local_labels, saliency_map) in enumerate(train_loader):
            global_step += 1
            local_embeddings, pred_low = backbone(img)
            
            # SCOSPS---------------------
            optimizer_sc.zero_grad()
            
            pred = interp(pred_low)
            # prepare for torch model_zoo models images
            zoo_mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            zoo_var = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
            images_zoo_cpu = (img.cpu().numpy() +
                            IMG_MEAN.reshape((1, 3, 1, 1))) / 255.0
            images_zoo_cpu -= zoo_mean
            images_zoo_cpu /= zoo_var

            images_zoo_cpu = torch.from_numpy(images_zoo_cpu)
            images_zoo = images_zoo_cpu.cuda(cfg.gpu)

            with torch.no_grad():
                zoo_feats = zoo_feat_net(images_zoo)
                zoo_feat = torch.cat([interp(zoo_feat)
                                      for zoo_feat in zoo_feats], dim=1)
                # saliency masking
                zoo_feat = zoo_feat * \
                    saliency_map.unsqueeze(dim=1).expand_as(zoo_feat).cuda(cfg.gpu)
            
            loss_sc = loss_scops.semantic_consistency_loss(
                features=zoo_feat, pred=pred, basis=part_basis_generator())
            
            # orthonomal_loss
            loss_orthonamal = loss_scops.orthonomal_loss(part_basis_generator())

            # Concentratin Loss
            loss_con = loss_scops.concentration_loss(pred)

            # Equivariance Loss
            images_cj = torch.from_numpy(
                ((img.cpu().numpy() + IMG_MEAN.reshape((1, 3, 1, 1))) / 255.0).clip(0, 1.0))
            for b in range(images_cj.shape[0]):
                images_cj[b] = torch.from_numpy(cj_transform(
                    images_cj[b]).numpy() * 255.0 - IMG_MEAN.reshape((1, 3, 1, 1)))
            images_cj = images_cj.cuda()

            tps.reset_control_points()

            # sum all loss terms
            loss_seg = cfg.lambda_con * loss_con \
                + cfg.lambda_sc * loss_sc \
                + cfg.lambda_orthonormal * loss_orthonamal
            # ---------------------------
            
            sum_of_parameters = sum(p.sum() for p in backbone.parameters())
            zero_sum = sum_of_parameters * 0.0

            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels, opt) + zero_sum

            total_loss = lambda_scheduler(loss, loss_seg, epoch)
            
            if cfg.fp16:
                amp.scale(total_loss).backward()
                amp.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                amp.step(opt)
                amp.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                opt.step()

            opt.zero_grad()
            lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                if global_step % cfg.verbose == 0 and global_step > 0:
                    callback_verification(global_step, backbone)

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint_gpu_{rank}.pt"))

        if rank == 0:
            path_module = os.path.join(cfg.output, "model.pt")
            torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_loader.reset()

    if rank == 0:
        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        from torch2onnx import convert_onnx
        convert_onnx(backbone.module.cpu().eval(), path_module, os.path.join(cfg.output, "model.onnx"))

    distributed.destroy_process_group()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, help="py config file")
    parser.add_argument("--local_rank", type=int, default=0, help="local_rank")
    main(parser.parse_args())
