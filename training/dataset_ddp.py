"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

# Dataset code for the DDP training setting.

from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torch.utils import data
import numpy as np
import random
import re, os
from torchvision import transforms
import torch


class MultiResolutionDataset(Dataset):
    def __init__(self, cloth_path, person_path, pose_path, mask_path, transform, resolution=256):
        self.cloth_path = cloth_path
        self.person_path = person_path
        self.pose_path = pose_path
        self.mask_path = mask_path

        self.resolution = resolution
        self.transform = transform
        self.length = None

    def _open(self):
        self.cloth_env = lmdb.open(
            self.cloth_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.person_env = lmdb.open(
            self.person_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.pose_env = lmdb.open(
            self.pose_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.mask_env = lmdb.open(
            self.mask_path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError(f"Cannot open lmdb dataset {self.path}")

        with self.cloth_env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("utf-8")).decode("utf-8"))

    def _close(self):
        if self.cloth_env is not None:
            self.cloth_env.close()
            self.cloth_env = None
            self.person_env.close()
            self.person_env = None
            self.pose_env.close()
            self.pose_env = None
            self.mask_env.close()
            self.mask_env = None

    def __len__(self):
        if self.cloth_env is None:
            self._open()
            self._close()

        return self.length

    def __getitem__(self, index):
        if self.cloth_env is None:
            self._open()

        with self.cloth_env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        cloth = self.transform(img)

        with self.person_env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        person = self.transform(img)

        with self.pose_env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        pose = self.transform(img)

        with self.mask_env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes = txn.get(key)
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        mask = self.transform(img)

        return cloth,person,pose,mask
