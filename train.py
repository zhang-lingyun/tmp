import argparse
import os
import cv2
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from training import lpips
from training.model import Generator, Discriminator, Encoder
from training.dataset_ddp import MultiResolutionDataset
from tqdm import tqdm

from torchvision import utils
import numpy as np
from PIL import Image

from utils import *
import random
import torchvision.transforms.functional as tf
import random


if __name__ == '__main__':
  torch.multiprocessing.set_start_method('spawn')

torch.backends.cudnn.benchmark = True

class DDPModel(nn.Module):
    def __init__(self, device, args):
        super(DDPModel, self).__init__()

        # 3+3+3+5=14
        self.generator = Generator(
            args.size,
            args.mapping_layer_num,
            args.latent_channel_size,
            args.latent_spatial_size1,
            args.latent_spatial_size2,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
            last_channels=12,
        )
        self.g_ema = Generator(
            args.size,
            args.mapping_layer_num,
            args.latent_channel_size,
            args.latent_spatial_size1,
            args.latent_spatial_size2,
            lr_mul=args.lr_mul,
            channel_multiplier=args.channel_multiplier,
            normalize_mode=args.normalize_mode,
            small_generator=args.small_generator,
            last_channels=12,
        )

        self.discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,input_channel=12,
        )
        self.dis_p = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,input_channel=3,
        )
        self.dis_c = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,input_channel=3,
        )
        self.dis_pose = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,input_channel=3,
        )
        self.dis_seg = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier,input_channel=3,
        )

        self.ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.mse_loss = nn.MSELoss(reduction="mean")
        self.percept = lpips.exportPerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
        )
        self.device = device
        self.args = args
    

    def forward(self, real_img, mode):
        if mode == "G":
            z = make_noise(
                self.args.batch_per_gpu,
                self.args.latent_channel_size,
                self.device,
            )

            fake_cloth,fake_person,fake_pose,fake_mask,_ = self.generator(z, return_stylecode=False)

            fake_pred = self.discriminator(torch.cat([fake_cloth,fake_person,fake_pose,fake_mask],dim=1))
            adv_loss = g_nonsaturating_loss(fake_pred)

            fake_pred = self.dis_p(fake_person)
            adv_p_loss = g_nonsaturating_loss(fake_pred)

            fake_pred = self.dis_c(fake_cloth)
            adv_c_loss = g_nonsaturating_loss(fake_pred)

            fake_pred = self.dis_pose(fake_pose)
            adv_pose_loss = g_nonsaturating_loss(fake_pred)

            fake_pred = self.dis_seg(fake_mask)
            adv_seg_loss = g_nonsaturating_loss(fake_pred)

            return adv_loss, adv_p_loss, adv_c_loss, adv_pose_loss, adv_seg_loss

        elif mode == "D":
            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_cloth,fake_person,fake_pose,fake_mask,_ = self.generator(z, return_stylecode=False)

            real_pred = self.discriminator(torch.cat(real_img,dim=1))
            fake_pred = self.discriminator(torch.cat([fake_cloth,fake_person,fake_pose,fake_mask],dim=1))
            d_loss = d_logistic_loss(real_pred, fake_pred)

            return d_loss

        elif mode == "D_reg":
            real_cloth,real_person,real_pose,real_mask = real_img
            real_person.requires_grad,real_cloth.requires_grad,real_pose.requires_grad,real_mask.requires_grad = True,True,True,True
            real_img = torch.cat([real_cloth,real_person,real_pose,real_mask],dim=1)
            real_pred = self.discriminator(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss
        
        elif mode == "D_P":
            real_person = real_img
            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_cloth,fake_person,fake_pose,fake_mask,_ = self.generator(z, return_stylecode=False)

            real_pred = self.dis_p(real_person)
            fake_pred = self.dis_p(fake_person)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            return d_loss

        elif mode == "D_P_reg":
            real_img.requires_grad = True
            real_pred = self.dis_p(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss
        
        elif mode == "D_C":
            real_cloth = real_img

            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_cloth,fake_person,fake_pose,fake_mask,_ = self.generator(z, return_stylecode=False)

            real_pred = self.dis_c(real_cloth)

            fake_pred = self.dis_c(fake_cloth)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            return d_loss

        elif mode == "D_C_reg":
            real_img.requires_grad = True
            real_pred = self.dis_c(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss
        
        elif mode == "D_Pose":
            real_pose = real_img

            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_cloth,fake_person,fake_pose,fake_mask,_ = self.generator(z, return_stylecode=False)

            real_pred = self.dis_pose(real_pose)
            fake_pred = self.dis_pose(fake_pose)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            return d_loss
        
        elif mode == "D_Pose_reg":
            real_img.requires_grad = True
            real_pred = self.dis_pose(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss
        
        elif mode == "D_Seg":
            real_mask = real_img

            with torch.no_grad():
                z = make_noise(
                    self.args.batch_per_gpu,
                    self.args.latent_channel_size,
                    self.device,
                )
                fake_cloth,fake_person,fake_pose,fake_mask,_ = self.generator(z, return_stylecode=False)

            real_pred = self.dis_seg(real_mask)
            fake_pred = self.dis_seg(fake_mask)
            d_loss = d_logistic_loss(real_pred, fake_pred)

            return d_loss

        elif mode == "D_Seg_reg":
            real_img.requires_grad = True
            real_pred = self.dis_seg(real_img)
            r1_loss = d_r1_loss(real_pred, real_img)
            d_reg_loss = (
                self.args.r1 / 2 * r1_loss * self.args.d_reg_every + 0 * real_pred[0]
            )

            return d_reg_loss, r1_loss


def run(ddp_fn, world_size, args):
    print("world size", world_size)
    mp.spawn(ddp_fn, args=(world_size, args), nprocs=world_size, join=True)


def ddp_main(rank, world_size, args):
    print(f"Running DDP model on rank {rank}.")
    # setup(rank, world_size)
    map_location = f"cuda:{rank}"
    device = map_location
    torch.cuda.set_device(map_location)

    if args.ckpt:  # ignore current arguments
        ckpt = torch.load(args.ckpt, map_location=map_location)
        train_args = ckpt["train_args"]
        print("load model:", args.ckpt)
        train_args.start_iter = int(args.ckpt.split("/")[-1].replace(".pt", ""))
        print(f"continue training from {train_args.start_iter} iter")

        train_args.lr_t = args.lr_t

        args = train_args
        args.ckpt = True
    else:
        args.start_iter = 0
    

    # create model and move it to GPU with id rank
    model = DDPModel(device=map_location, args=args).to(map_location)
    # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    model.train()

    g_module = model.generator
    g_ema_module = model.g_ema
    g_ema_module.eval()
    accumulate(g_ema_module, g_module, 0)

    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    g_optim = optim.Adam(
        g_module.parameters(),
        lr=args.lr,
        betas=(0, 0.99),
    )
    
    d_optim = optim.Adam(
        model.discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    d_p_optim = optim.Adam(
        model.dis_p.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    d_c_optim = optim.Adam(
        model.dis_c.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    d_pose_optim = optim.Adam(
        model.dis_pose.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    d_seg_optim = optim.Adam(
        model.dis_seg.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    accum = 0.999

    if args.ckpt:
        model.generator.load_state_dict(ckpt["generator"])
        model.discriminator.load_state_dict(ckpt["discriminator"])
        model.dis_p.load_state_dict(ckpt["dis_p"])
        model.dis_c.load_state_dict(ckpt["dis_c"])
        model.dis_pose.load_state_dict(ckpt["dis_pose"])
        model.dis_seg.load_state_dict(ckpt["dis_seg"])
        model.g_ema.load_state_dict(ckpt["g_ema"])
        
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        d_p_optim.load_state_dict(ckpt["d_p_optim"])
        d_c_optim.load_state_dict(ckpt["d_c_optim"])
        d_pose_optim.load_state_dict(ckpt["d_pose_optim"])
        d_seg_optim.load_state_dict(ckpt["d_seg_optim"])

        del ckpt  # free GPU memory
    
    # 水平翻转后cloth和person仍然是匹配的
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    save_dir = "expr"
    os.makedirs(save_dir, 0o777, exist_ok=True)
    os.makedirs(save_dir + "/checkpoints", 0o777, exist_ok=True)
    os.makedirs(save_dir + "/sample", 0o777, exist_ok=True)

    dataset = MultiResolutionDataset(args.cloth_lmdb, args.person_lmdb, args.pose_lmdb, args.mask_lmdb, transform, args.size)

    print(f"dataset: {len(dataset)}")

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_per_gpu,
        drop_last=True,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    loader = sample_data(loader)

    pbar = range(args.start_iter, args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, mininterval=1)

    epoch = -1
    # gpu_group = dist.new_group(list(range(args.ngpus)))

    requires_grad(model.discriminator, False)
    requires_grad(model.dis_p, False)
    requires_grad(model.dis_c, False)
    requires_grad(model.dis_pose, False)
    requires_grad(model.dis_seg, False)
    for i in pbar:
        if i > args.iter:
            print("Done!")
            break
        elif i % (len(dataset) // args.batch) == 0:
            epoch += 1
            sampler.set_epoch(epoch)
            print("epoch: ", epoch)
        
        cloth,person,pose,mask = next(loader)


        if random.random() > 0.5:
            # data augment
            cloth = tf.hflip(cloth)

        cloth,person,pose,mask = cloth.to(map_location),person.to(map_location),pose.to(map_location),mask.to(map_location)

        # G
        adv_loss, adv_p_loss, adv_c_loss, adv_pose_loss, adv_seg_loss = model(None, "G")

        g_optim.zero_grad()
        (adv_loss + adv_p_loss + adv_c_loss + adv_pose_loss + adv_seg_loss).backward()
        g_optim.step()


        requires_grad(model.discriminator, True)
        requires_grad(model.dis_p, True)
        requires_grad(model.dis_c, True)
        requires_grad(model.dis_pose, True)
        requires_grad(model.dis_seg, True)
        # D
        d_loss = model([cloth,person,pose,mask], "D")
        d_optim.zero_grad()
        (d_loss).backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0
        if d_regularize:
            d_reg_loss, r1_loss = model([cloth,person,pose,mask], "D_reg")
            d_reg_loss = d_reg_loss.mean()
            d_optim.zero_grad()
            d_reg_loss.backward()
            d_optim.step()

        # DP
        d_p_loss = model(person, "D_P")
        d_p_optim.zero_grad()
        (d_p_loss).backward()
        d_p_optim.step()

        if d_regularize:
            d_p_reg_loss, r1_loss = model(person, "D_P_reg")
            d_p_reg_loss = d_p_reg_loss.mean()
            d_p_optim.zero_grad()
            d_p_reg_loss.backward()
            d_p_optim.step()

        # DC
        d_c_loss = model(cloth, "D_C")
        d_c_optim.zero_grad()
        (d_c_loss).backward()
        d_c_optim.step()

        if d_regularize:
            d_c_reg_loss, r1_loss = model(cloth, "D_C_reg")
            d_c_reg_loss = d_c_reg_loss.mean()
            d_c_optim.zero_grad()
            d_c_reg_loss.backward()
            d_c_optim.step()
        
        # D Pose
        d_pose_loss = model(pose, "D_Pose")
        d_pose_optim.zero_grad()
        (d_pose_loss).backward()
        d_pose_optim.step()

        if d_regularize:
            d_pose_reg_loss, r1_loss = model(pose, "D_Pose_reg")
            d_pose_reg_loss = d_pose_reg_loss.mean()
            d_pose_optim.zero_grad()
            d_pose_reg_loss.backward()
            d_pose_optim.step()
        
        # D Seg
        d_seg_loss = model(mask, "D_Seg")
        d_seg_optim.zero_grad()
        (d_seg_loss).backward()
        d_seg_optim.step()

        if d_regularize:
            d_seg_reg_loss, r1_loss = model(mask, "D_Seg_reg")
            d_seg_reg_loss = d_seg_reg_loss.mean()
            d_seg_optim.zero_grad()
            d_seg_reg_loss.backward()
            d_seg_optim.step()


        pbar.set_description(
            (f"adv_loss: {adv_loss:.4f}; adv_p_loss: {adv_p_loss:.4f}; adv_c_loss: {adv_c_loss:.4f};  adv_pose_loss: {adv_pose_loss:.4f}; adv_seg_loss: {adv_seg_loss:.4f}; \
               d_loss: {d_loss:.4f}; d_p_loss: {d_p_loss:.4f}; d_c_loss: {d_c_loss:.4f};  d_pose_loss: {d_pose_loss:.4f}; d_seg_loss: {d_seg_loss:.4f};")
        )

        with torch.no_grad():
            accumulate(g_ema_module, g_module, accum)

            if i % args.save_network_interval == 0:
                copy_norm_params(g_ema_module, g_module)
                
                # G
                z = make_noise(
                    args.batch_per_gpu,
                    args.latent_channel_size,
                    device,
                )
                fake_cloth,fake_person,fake_pose,fake_mask,_ = g_ema_module(z, return_stylecode=False)

                sample = torch.cat((fake_cloth,fake_person,fake_pose,fake_mask), dim=0)
                utils.save_image(
                        sample,
                        f"expr/sample/{str(i).zfill(6)}_generation.png",
                        nrow=args.batch,
                        normalize=True,
                        value_range=(-1, 1),
                    )
                

                torch.save(
                        {
                            "generator": model.generator.state_dict(),
                            "discriminator": model.discriminator.state_dict(),
                            "dis_p": model.dis_p.state_dict(),
                            "dis_c": model.dis_c.state_dict(),
                            "dis_pose": model.dis_pose.state_dict(),
                            "dis_seg": model.dis_seg.state_dict(),
                            "g_ema": g_ema_module.state_dict(),

                            "train_args": args,
                            "g_optim": g_optim.state_dict(),
                            "d_optim": d_optim.state_dict(),
                            "d_p_optim": d_p_optim.state_dict(),
                            "d_c_optim": d_c_optim.state_dict(), 
                            "d_pose_optim": d_pose_optim.state_dict(), 
                            "d_seg_optim": d_seg_optim.state_dict(), 
                        },
                        f"{save_dir}/checkpoints/{str(i).zfill(6)}.pt",
                    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_lmdb", type=str,default="/home/mist/data/256_train/image")
    parser.add_argument("--cloth_lmdb", type=str,default="/home/mist/data/256_train/cloth")
    parser.add_argument("--pose_lmdb", type=str,default="/home/mist/data/256_train/pose")
    parser.add_argument("--seg_lmdb", type=str,default="/home/mist/data/256_train/parse")

    parser.add_argument("--ckpt", type=str)
    parser.add_argument(
        "--dataset",
        type=str,
        default="viton",
        choices=[
            "viton",
        ],
    )
    parser.add_argument("--iter", type=int, default=1400000)
    parser.add_argument("--save_network_interval", type=int, default=1000)
    parser.add_argument("--small_generator", action="store_true")

    parser.add_argument("--batch", type=int, default=4, help="total batch sizes")
    parser.add_argument("--size", type=int, choices=[128, 256, 512, 1024], default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--lr_t", type=float, default=0.002)

    parser.add_argument("--lr_mul", type=float, default=0.01)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--latent_channel_size", type=int, default=64)
    parser.add_argument("--latent_spatial_size1", type=int, default=8)
    parser.add_argument("--latent_spatial_size2", type=int, default=6)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--normalize_mode",
        type=str,
        choices=["LayerNorm", "InstanceNorm2d", "BatchNorm2d", "GroupNorm"],
        default="LayerNorm",
    )
    parser.add_argument("--mapping_layer_num", type=int, default=8)

    parser.add_argument("--lambda_x_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_adv_loss", type=float, default=1)
    parser.add_argument("--lambda_w_rec_loss", type=float, default=1)
    parser.add_argument("--lambda_d_loss", type=float, default=1)
    parser.add_argument("--lambda_perceptual_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_D_loss", type=float, default=1)
    parser.add_argument("--lambda_indomainGAN_E_loss", type=float, default=1)

    input_args = parser.parse_args()

    ngpus = torch.cuda.device_count()
    print("{} GPUS!".format(ngpus))

    assert input_args.batch % ngpus == 0
    input_args.batch_per_gpu = input_args.batch // ngpus
    input_args.ngpus = ngpus
    print("{} batch per gpu!".format(input_args.batch_per_gpu))

    # run(ddp_main, ngpus, input_args)
    ddp_main(0, ngpus, input_args)