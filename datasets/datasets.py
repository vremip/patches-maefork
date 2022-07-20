# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import numpy as np
import os
import PIL

import torch as ch
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from typing import List

from ffcv.pipeline.operation import Operation

def build_dataset(is_train, args, augmentations="full"):
    root = os.path.join(args.data_path, 'train' if is_train else 'val')

    if getattr(args, "dataset"):
        name = args.dataset.lower()
        if name == "mnist":
            dataset = datasets.MNIST(
                root=root,
                train=is_train,
                download=True,  # TODO
                transform=transforms.Compose([transforms.ToTensor()]),
            )
        else:
            raise NotImplementedError(f"Dataset named {name!r} not implemented.")

    else:
        if augmentations == "full":
            transform = build_transform(is_train, args)
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)
    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def ffcv_loader(dataset, train=True, batch_size=64, num_workers=8, in_memory=True, distributed=False):

    from ffcv.pipeline.operation import Operation
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
        RandomHorizontalFlip, ToTorchImage
    from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
        RandomResizedCropRGBImageDecoder
    from ffcv.fields.basics import IntDecoder

    IMAGE_SIZE = 224
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
    IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
    DEFAULT_CROP_RATIO = 224/256

    # Will need to set things up for multi gpu training
    this_device = f'cuda:{os.environ.get("LOCAL_RANK", 0)}'

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(ch.device(this_device), non_blocking=True)
    ]

    if train:
        decoder = RandomResizedCropRGBImageDecoder((IMAGE_SIZE, IMAGE_SIZE))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        order = OrderOption.QUASI_RANDOM if not distributed else OrderOption.RANDOM
        loader = Loader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=train,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
    else:
        cropper = CenterCropRGBImageDecoder((IMAGE_SIZE, IMAGE_SIZE), ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        loader = Loader(dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        })

    return loader