from __future__ import division
import argparse
import copy
import os
import os.path as osp
import time
import numpy as np
import mmcv
import torch
from mmcv import Config
from mmcv.runner import init_dist

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector, init_detector, inference_detector, show_result
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    # parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", help="the dir to save logs and models")
    parser.add_argument("--resume_from", help="the checkpoint file to resume from")
    parser.add_argument("--pretrain_model", help="the checkpoint file to resume from")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="whether to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--autoscale-lr",
        action="store_true",
        help="automatically scale lr with the number of gpus",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def build_model(config_path, pretrain_path):
    # args = parse_args()

    cfg = Config.fromfile(config_path)
    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    # update configs according to CLI args
    
    cfg.pretrain_model = pretrain_path    
    
    cfg.gpu_ids = range(1)
    distributed = False
    
    # set random seeds
    # if args.seed is not None:
    #     logger.info(
    #         "Set random seed to {}, deterministic: {}".format(
    #             args.seed, args.deterministic
    #         )
    #     )
    #     set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = None
    model = init_detector(cfg, pretrain_path)
    return model


def obj_detect(model, img, N):
    img = mmcv.rgb2bgr(img)

    bbox_result, feature_maps = inference_detector(model, img)
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    confidents = bboxes[:,-1]
    keep_id = confidents.argsort()[-N:]
    labels = labels[keep_id]
    bboxes = bboxes[keep_id]

    final_rois = torch.from_numpy(bboxes[:, [4, 0, 1, 2, 3]])
    box_feats = model.bbox_roi_extractor(feature_maps[: len(model.bbox_roi_extractor.featmap_strides)], final_rois.cuda())
    box_feats = box_feats.reshape(N, 256, -1).mean(dim=-1)
    
    ## for visualization
    # img = mmcv.imread(img).copy()
    # mmcv.imshow_det_bboxes(img, bboxes, labels, class_names=[str(i) for i in range(1,501)], score_thr=0.5, show=False, out_file=None,)
    # img = mmcv.bgr2rgb(img)
    # import torchvision
    # torchvision.utils.save_image(torch.from_numpy(img/255).permute(2,0,1),'aa.jpg')
    # import pdb;pdb.set_trace()
    bboxes = torch.from_numpy(bboxes)
    labels = torch.from_numpy(labels).unsqueeze(-1)
    ret = torch.cat([labels.cuda().float(), bboxes.cuda() ,box_feats.cuda()],dim=-1)
    return ret

if __name__ == "__main__":
    config_path = '/mnt/proj56/sjhuang/TSD/configs/OpenImages_configs/r50-FPN-1x_classsampling_TSD/r50-FPN-1x_classsampling_TSD.py'
    pretrain_path = '/mnt/proj56/sjhuang/TSD/r50-FPN-1x_classsampling_TSD.pth'
    model = build_model(config_path, pretrain_path)
    bboxes, labels, box_feats = obj_detect(model, "/mnt/proj56/sjhuang/TSD/many.png", 5)
