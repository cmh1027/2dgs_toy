import torch
import sys

import torchmetrics.image
sys.path.append(".")
import misc
from misc import b_extract, b_assign
import numpy as np
from tqdm import trange, tqdm
import random
from PIL import Image
import copy
from torch import matmul as mm
from icecream import ic
import sys
import time
import argparse
import torchvision, torchmetrics
import os
class Gaussian2DModel:
    def __init__(self, N, iteration=None, lr=None, gt=False, fixed=[], gt_model=None):
        assert N == 25 or N == 4
        device = self.device = torch.device('cuda')
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = misc.inverse_sigmoid
        self.rgb_activation = torch.sigmoid
        self.rgb_inverse_activation = misc.inverse_sigmoid
        # self.rgb_activation = lambda x: x
        # self.rgb_inverse_activation = lambda x: x
        self.rotation_activation = torch.tanh
        if N == 25:
            x = torch.tensor([-5, -2.5, 0.1, 2.5, 5.0])
            self._xy = torch.nn.Parameter((torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=2).reshape(-1, 2) + torch.rand(N, 2)).to(device)) # mean
        else:
            self._xy = torch.nn.Parameter((torch.rand(N, 2) + 1).to(device)) # mean
        self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(misc.generate_random_color(N)[torch.randperm(N)].to(device)))
        self._scale = torch.nn.Parameter(torch.rand(N, 2).to(device) + 0.5)
        self._rotation = torch.nn.Parameter((torch.rand(N, 1).to(device) - 0.5) * 2 * torch.pi) # unlike quaternion in 3D, theta is enough
        self._opacity = torch.nn.Parameter((torch.rand(N, 1).to(device) - 0.5) * 2)
        self.fixed = fixed
        if gt:
            if N == 25:
                x = torch.tensor([-20, -10, 0, 10, 20.])
                self._xy = torch.nn.Parameter(torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=2).reshape(-1, 2).to(device))
                # mat = torch.tensor([
                #     [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.7, 0.2, 0.3], [0.9, 0.2, 0.8],
                #     [0.9, 0.5, 0.2], [0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.9, 0.1, 0.2], [0.6, 0.1, 0.1],
                #     [0.2, 0.3, 0.8], [0.3, 0.2, 0.8], [0.8, 0.2, 0.3], [0.1, 0.9, 0.9], [0.1, 0.9, 0.3],
                #     [0.1, 0.9, 0.2], [0.7, 0.1, 0.3], [0.6, 0.1, 0.5], [0.9, 0.1, 0.9], [0.4, 0.9, 0.1],
                #     [0.3, 0.7, 0.2], [0.5, 0.5, 0.1], [0.7, 0.5, 0.3], [0.6, 0.1, 0.6], [0.9, 0.1, 0.9]
                # ])
                mat = torch.tensor([
                    [0.90, 0.01, 0.01], [0.90, 0.01, 0.20], [0.90, 0.01, 0.30], [0.90, 0.01, 0.40], [0.90, 0.01, 0.50],
                    [0.90, 0.50, 0.01], [0.90, 0.40, 0.01], [0.90, 0.30, 0.01], [0.90, 0.20, 0.20], [0.90, 0.01, 0.01],
                    [0.01, 0.90, 0.01], [0.30, 0.90, 0.01], [0.01, 0.90, 0.30], [0.50, 0.90, 0.01], [0.01, 0.90, 0.50],
                    [0.01, 0.01, 0.90], [0.20, 0.01, 0.90], [0.30, 0.01, 0.90], [0.40, 0.01, 0.90], [0.50, 0.01, 0.90],
                    [0.01, 0.50, 0.90], [0.01, 0.40, 0.90], [0.01, 0.30, 0.90], [0.01, 0.20, 0.90], [0.01, 0.01, 0.90]
                ])
                mat = mat.reshape(5,5,3).permute(1,0,2).reshape(25,3)
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(mat).to(device))
                x = torch.tensor([1.1, 0.8, 1.3, 0.5, 1.0])
                self._scale = torch.nn.Parameter(torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=2).reshape(-1, 2).to(device))
                self._rotation = torch.nn.Parameter(torch.linspace(-torch.pi, torch.pi, N)[..., None].to(device)) 
                self._opacity = torch.nn.Parameter((5 * torch.ones(N, 1)).to(device))
                # self._opacity = torch.nn.Parameter(torch.linspace(-1, 2, N)[..., None].to(device))
            elif N == 4:
                self._xy = torch.nn.Parameter(torch.tensor([[25., 25.], [-25., 25.], [25, -25.], [-25, -25.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1, 0.9]])).to(device))
                self._scale = torch.nn.Parameter(torch.ones(N, 2).to(device))
                self._rotation = torch.nn.Parameter(torch.zeros(N, 1).to(device)) 
                self._opacity = torch.nn.Parameter((torch.ones(N, 1) * 3).to(device))
            else:
                raise NotImplementedError
        else:
            if fixed != []:
                assert gt_model is not None
            if "rgb" in fixed:
                self._rgb = copy.deepcopy(gt_model._rgb)
            if "xy" in fixed:
                self._xy = copy.deepcopy(gt_model._xy)
            if "scale" in fixed:
                self._scale = copy.deepcopy(gt_model._scale)
            if "rotation" in fixed:
                self._rotation = copy.deepcopy(gt_model._rotation)
            if "opacity" in fixed:
                self._opacity = copy.deepcopy(gt_model._opacity)
        if lr is not None and iteration is not None:
            self.training_setup(lr, iteration)

    @property
    def get_rgb(self):
        return self.rgb_activation(self._rgb)

    def set_rgb(self, rgb, eps=1e-4):
        assert rgb.shape[-1] == 3
        self._rgb = self.rgb_inverse_activation(torch.clamp(rgb, eps, 1-eps))

    @property
    def get_scale(self):
        return self.scale_activation(self._scale)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) * torch.pi
    
    @property
    def get_xy(self):
        return self._xy
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def set_opacity(self, opacity, eps=1e-4):
        assert opacity.shape[-1] == 1
        self._opacity = self.opacity_inverse_activation(torch.clamp(opacity, eps, 1-eps))

    def training_setup(self, lr, iteration):
        lr_xy = lr['xy'] if 'xy' not in self.fixed else 0.
        lr_rgb = lr['rgb'] if 'rgb' not in self.fixed else 0.
        lr_opacity = lr['opacity'] if 'opacity' not in self.fixed else 0.
        lr_scale = lr['scale'] if 'scale' not in self.fixed else 0.
        lr_rotation = lr['rotation'] if 'rotation' not in self.fixed else 0.
        l = [
            {'params': [self._xy], 'lr': lr_xy, "name": "xy"},
            {'params': [self._rgb], 'lr': lr_rgb, "name": "rgb"},
            {'params': [self._opacity], 'lr': lr_opacity, "name": "opacity"},
            {'params': [self._scale], 'lr': lr_scale, "name": "scale"},
            {'params': [self._rotation], 'lr': lr_rotation, "name": "rotation"},
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.1, eps=1e-15)
        self.xy_scheduler_args = misc.get_expon_lr_func(lr_init=0.1001,
                                                         lr_final=0.1000016,
                                                         lr_delay_mult=0.11,
                                                         max_steps=iteration)

    @property
    def get_covariance(self):
        theta = self.get_rotation
        N = theta.shape[0]
        R = torch.cat([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)], dim=-1).reshape(N, 2, 2).to(self.device)
        S = torch.diag_embed(self.get_scale).to(self.device)
        RS = torch.bmm(R, S)
        return torch.bmm(RS, RS.permute(0, 2, 1))

    def get_proj_gaussians(self, n, b):
        """
        n : (2,) unit vector
        b : scalar
        n = (p, q)
        px + qy + b = 0
        """
        line = misc.normal2dir(n)
        xy = self.get_xy
        bias = misc.get_bias(n, b).cuda()
        proj_xy = mm(xy, line[..., None]) * line[None, ...] + bias[None, ...]
        distance = torch.sqrt(((xy - proj_xy) ** 2).sum(dim=-1))
        N = proj_xy.shape[0]
        cov = self.get_covariance
        line = line.view(1, 2).repeat(N, 1) # (N, 2)
        n = n.view(1,2).repeat(N, 1)
        proj_var = torch.bmm(line.view(N, 1, 2), torch.bmm(cov, line.view(N, 2, 1)))
        near_cull_mask = distance >= 0.11
        back_cull_mask = (torch.bmm(xy.view(N, 1, 2), n.view(N, 2, 1)).squeeze() + b) < 0
        mask = near_cull_mask & back_cull_mask
        return proj_xy, proj_var, distance, mask
    
    def render(self, x, n, b):
        """
        x : (k,) coefficients 
        """
        y = torch.zeros_like(x)[..., None] # (k, 1)
        T = torch.ones_like(x)[..., None]
        mu_, var_, distance, mask = self.get_proj_gaussians(n, b)
        sort_idx = torch.argsort(distance[mask])
        mu_, var_ = mu_[mask][sort_idx], var_[mask][sort_idx]
        rgb_ = self.get_rgb
        opacity_ = self.get_opacity
        opacity_, rgb_ = opacity_[mask][sort_idx], rgb_[mask][sort_idx]
        
        line = misc.normal2dir(n)
        bias = misc.get_bias(n, b)
        xs = line[None, ...] * x[..., None] + bias

        params = {}
        sorted_weights = []
        for mu, var, opacity, rgb in zip(mu_, var_, opacity_, rgb_):
            g = misc.eval_normal_1d(xs, mu, var)
            alpha = opacity * g
            y = y + T * alpha * rgb[None, ...]
            weight = alpha * T # (k, 1)
            sorted_weights.append(weight)
            T = T * (1-alpha)
        sorted_weights = torch.stack(sorted_weights, dim=0)

        N, k = self.get_opacity.shape[0], len(x)
        weights = torch.zeros(N, k, 1).to(x.device)
        weights[mask][sort_idx] = sorted_weights # (N, k, 1)
        params["w"] = weights
        params["sort_idx"] = sort_idx
        return y, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=2024) 
    parser.add_argument('--method', '-m', type=str, required=True) 
    parser.add_argument('--iteration', '-i', type=int, default=1000) 
    parser.add_argument('--exp_name', '-e', type=str, default="temp") 
    parser.add_argument('--bias', '-b', type=int, default=20) 
    args = parser.parse_args() 
    misc.set_seed(args.seed)
    METHOD = args.method.upper()
    save_dir = f"fig/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)


    N = 25
    xmin, xmax = -15, 15
    slope_N = 400
    theta = torch.linspace(-torch.pi, torch.pi, slope_N)[..., None] # (slope_N, 1)
    slope_list = torch.cat([torch.cos(theta), torch.sin(theta), torch.zeros(slope_N, 1).fill_(-args.bias)], dim=-1) # (slope_N, 3)
    li = list(range(0, slope_N, 4))
    train_mask = torch.zeros(slope_N).bool()
    train_mask[li] = True
    eval_mask = ~train_mask
    train_slope, eval_slope = slope_list[train_mask], slope_list[eval_mask]
    print(f"Number of train slopes : {len(train_slope)}")
    print(f"Number of eval slopes : {len(eval_slope)}")


    data = torch.arange(xmin, xmax, 0.1).cuda()
    iteration = args.iteration
    lr = {
        'xy': 0.1,
        'rgb': 0.1,
        'opacity': 0.15,
        'scale': 0.1,
        'rotation': 0.5
    }
    
    ### fixed ['rgb', 'xy', 'scale', 'rotation' 'opacity']
    if METHOD == "GD":
        fixed = []
        gt = True
    elif METHOD == "Ours":
        fixed = []
        gt = True
    else:
        raise NotImplementedError
    gt_model = Gaussian2DModel(N, gt=gt)
    model = Gaussian2DModel(N, iteration=iteration, lr=lr, fixed=fixed, gt_model=gt_model)
    p_init = misc.draw_model(model, None, None, [xmin, xmax], "init")
    images = []
    ### Data preparation
    with torch.no_grad():
        for i, (n_x, n_y, bias) in enumerate(train_slope):
            normal = torch.tensor([n_x,n_y]).cuda().float()
            normal = normal / torch.norm(normal)
            y, _ = gt_model.render(data, normal, bias)
            images.append((normal, bias, y))
    #####################

    if METHOD == "GD":
        stack = list(range(len(images)))
        random.shuffle(images)
        pbar = trange(iteration)
        for i in pbar:
            if stack == []:
                stack = list(range(len(images)))
                random.shuffle(images)
            normal, bias, y = images[stack.pop()]
            y_pred, _ = model.render(data, normal, bias)
            loss = ((y - y_pred)**2).mean()
            pbar.set_postfix({"loss":"%.3f" % loss.item()})
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
            
    else:
        raise NotImplementedError
    #####################

    ### Visualization
    for mode, slope in zip(('train', 'eval'), (train_slope, eval_slope)):
        with torch.no_grad():
            pred_vecs, gt_vecs = [], []
            for i, (n_x, n_y, bias) in enumerate(slope):
                normal = torch.tensor([n_x,n_y]).cuda().float()
                normal = normal / torch.norm(normal)
                p_pred = misc.draw_model(model, normal, bias, [xmin, xmax], "predict")
                p_gt = misc.draw_model(gt_model, normal, bias, [xmin, xmax], "GT")
                pred_vec, _ = model.render(data, normal, bias)
                gt_vec, _ = gt_model.render(data, normal, bias)
                pred_vecs.append(pred_vec[None, ...])
                gt_vecs.append(gt_vec[None, ...])
                if i % 20 == 0 : 
                    r_pred = misc.line2image(pred_vec)
                    r_gt = misc.line2image(gt_vec)
                    h, w, _ = r_pred.shape
                    r = np.zeros((h, 2*w+10, 3), dtype=np.uint8)
                    r[:, 0:w], r[:, w+10:] = r_pred, r_gt
                    Image.fromarray(r).save(f"{save_dir}/plot_{mode}_{'%03d' % i}_render.png")
                    
                    h, w, _ = p_pred.shape
                    p = np.zeros((h, 3*w+20, 3), dtype=np.uint8)
                    p[:, 0:w], p[:, w+10:2*w+10], p[:, 2*w+20:] = p_init, p_pred, p_gt
                    Image.fromarray(p).save(f"{save_dir}/plot_{mode}_{'%03d' % i}.png")

            pred_vec = torch.cat(pred_vecs, dim=0).permute(2, 0, 1) 
            gt_vec = torch.cat(gt_vecs, dim=0).permute(2, 0, 1)

            pred_img = np.array(torchvision.transforms.functional.to_pil_image(pred_vec)) 
            gt_img = np.array(torchvision.transforms.functional.to_pil_image(gt_vec))
            h, w, _ = pred_img.shape
            r = np.zeros((h, 2*w+10, 3), dtype=np.uint8)
            r[:, 0:w], r[:, w+10:] = pred_img, gt_img
            Image.fromarray(r).save(f"{save_dir}/render_{mode}.png")
            metric = torchmetrics.image.PeakSignalNoiseRatio().cuda()
            metric.update(pred_vec, gt_vec)
            print(f"{mode} PSNR : ", metric.compute().item())

