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
import torch.nn as nn
import argparse
import torchvision, torchmetrics
import os

class MeanApproxNet(nn.Module):
    def __init__(self):
        super(MeanApproxNet, self).__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5) 
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  
        x = torch.relu(self.fc2(x))
        x = self.dropout(x) 
        x = torch.relu(self.fc3(x))
        x = self.dropout(x) 
        x = self.fc4(x)
        return x

class Gaussian2DModel:
    def __init__(self, N, iteration=None, gt=False, fixed=[], noise=[], gt_model=None):
        device = self.device = torch.device('cuda')
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = misc.inverse_sigmoid
        self.rgb_activation = torch.sigmoid
        self.rgb_inverse_activation = misc.inverse_sigmoid
        # self.rgb_activation = lambda x: x
        # self.rgb_inverse_activation = lambda x: x
        self.rotation_activation = torch.tanh
        # x = torch.tensor([-5, -2.5, 0.1, 2.5, 5.0])
        # self._xy = torch.nn.Parameter((torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=2).reshape(-1, 2) + torch.rand(N, 2)).to(device)) # mean
        self._xy = torch.nn.Parameter(((torch.rand(N, 2) - 0.5) * 30).to(device)) # mean
        self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(misc.generate_random_color(N)[torch.randperm(N)].to(device)))
        self._scale = torch.nn.Parameter(torch.rand(N, 2).to(device) + 0.5)
        self._rotation = torch.nn.Parameter((torch.rand(N, 1).to(device) - 0.5) * 2 * torch.pi) # unlike quaternion in 3D, theta is enough
        self._opacity = torch.nn.Parameter((torch.rand(N, 1).to(device) - 0.5) * 2)
        self.fixed = fixed
        if gt:
            if N != 4:
                # x = torch.tensor([-20, -10, 0, 10, 20.])
                # mat = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=2).reshape(-1, 2)
                # mat = torch.tensor([
                #     [-20, 20],  [-10, 20],  [0, 20],  [10, 20],  [20, 20],
                #     [-20, 10],  [-10, 10],  [0, 10],  [10, 10],  [20, 10],
                #     [-20, 0],   [-10, 0],   [0, 0],   [10, 0],  [20, 0],
                #     [-20, -10], [-10, -10], [0, -10], [10, -10], [20, -10],
                #     [-20, -20], [-10, -20], [0, -20], [10, -20], [20, -20]
                # ]).float()
                M = 40
                C = 21
                background = torch.cat([
                    torch.cat((torch.linspace(-M, M, C)[..., None], torch.tensor([[M]]).repeat(C, 1)), dim=-1),
                    torch.cat((torch.linspace(-M, M, C)[..., None], torch.tensor([[-M]]).repeat(C, 1)), dim=-1),
                    torch.cat((torch.tensor([[M]]).repeat(C-2, 1), torch.linspace(-M-5, M-5, C-2)[..., None]), dim=-1),
                    torch.cat((torch.tensor([[-M]]).repeat(C-2, 1), torch.linspace(-M-5, M-5, C-2)[..., None]), dim=-1),
                ], dim=0).float()
                center = torch.tensor([
                    [-5, 5], [0, 5], [5, 5],    
                    [-5, 0], [0, 0], [5, 0],    
                    [-5, -5],[0, -5],[5, -5],  
                ]).float()
                mat = torch.cat([background, center], dim=0)
                self._xy = torch.nn.Parameter(mat.to(device))
                # mat = torch.tensor([
                #     [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.7, 0.2, 0.3], [0.9, 0.2, 0.8],
                #     [0.9, 0.5, 0.2], [0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.9, 0.1, 0.2], [0.6, 0.1, 0.1],
                #     [0.2, 0.3, 0.8], [0.3, 0.2, 0.8], [0.8, 0.2, 0.3], [0.1, 0.9, 0.9], [0.1, 0.9, 0.3],
                #     [0.1, 0.9, 0.2], [0.7, 0.1, 0.3], [0.6, 0.1, 0.5], [0.9, 0.1, 0.9], [0.4, 0.9, 0.1],
                #     [0.3, 0.7, 0.2], [0.5, 0.5, 0.1], [0.7, 0.5, 0.3], [0.6, 0.1, 0.6], [0.9, 0.1, 0.9]
                # ])
                # mat = torch.tensor([
                #     [0.90, 0.01, 0.01], [0.90, 0.01, 0.20], [0.90, 0.01, 0.30], [0.90, 0.01, 0.40], [0.90, 0.01, 0.50],
                #     [0.90, 0.50, 0.01], [0.90, 0.40, 0.01], [0.90, 0.30, 0.01], [0.90, 0.20, 0.20], [0.90, 0.01, 0.01],
                #     [0.01, 0.90, 0.01], [0.30, 0.90, 0.01], [0.01, 0.90, 0.30], [0.50, 0.90, 0.01], [0.01, 0.90, 0.50],
                #     [0.01, 0.01, 0.90], [0.20, 0.01, 0.90], [0.30, 0.01, 0.90], [0.40, 0.01, 0.90], [0.50, 0.01, 0.90],
                #     [0.01, 0.50, 0.90], [0.01, 0.40, 0.90], [0.01, 0.30, 0.90], [0.01, 0.20, 0.90], [0.01, 0.01, 0.90]
                # ])
                # mat = mat.reshape(5,5,3).permute(1,0,2).reshape(25,3)
                mat = misc.generate_random_color(N)[torch.randperm(N)]
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(mat).to(device))
                # x = torch.tensor([1.1, 0.8, 1.3, 0.5, 1.0])
                # mat = torch.stack(torch.meshgrid(x, x, indexing='ij'), dim=2).reshape(-1, 2)
                mat = torch.rand(N, 2) + 0.5
                self._scale = torch.nn.Parameter(mat.to(device))
                self._rotation = torch.nn.Parameter(torch.linspace(-torch.pi, torch.pi, N)[..., None].to(device)) 
                self._opacity = torch.nn.Parameter(((torch.ones(N, 1) + 0.5) * 2).to(device))
                # self._opacity = torch.nn.Parameter(torch.linspace(-1, 2, N)[..., None].to(device))
            elif N == 4:
                self._xy = torch.nn.Parameter(torch.tensor([[5., 5.], [-5., 5.], [5, -5.], [-5, -5.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1, 0.9]])).to(device))
                self._scale = torch.nn.Parameter(torch.ones(N, 2).to(device))
                self._rotation = torch.nn.Parameter(torch.zeros(N, 1).to(device)) 
                self._opacity = torch.nn.Parameter((torch.ones(N, 1) * 3).to(device))
            else:
                raise NotImplementedError
        else:
            if fixed != [] or noise != []:
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
            with torch.no_grad():
                if "xy" in noise:
                    self._xy = copy.deepcopy(gt_model._xy) + (torch.rand_like(gt_model._xy))

    @property
    def get_rgb(self):
        return self.rgb_activation(self._rgb)

    @torch.no_grad()
    def set_rgb(self, rgb, eps=1e-4, mask=None, sort_idx=None):
        assert rgb.shape[-1] == 3
        if mask is None: mask = torch.ones_like(model.get_rgb[..., 0]).bool()
        self._rgb.index_put_((mask.nonzero(as_tuple=True)[0][sort_idx],), self.rgb_inverse_activation(torch.clamp(rgb, eps, 1-eps)))

    @property
    def get_scale(self):
        return self.scale_activation(self._scale)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) * torch.pi
    
    @property
    def get_xy(self):
        return self._xy

    @torch.no_grad()
    def set_xy(self, xy, mask=None, sort_idx=None):
        assert xy.shape[-1] == 2
        if mask is None: mask = torch.ones_like(model.get_xy[..., 0]).bool()
        self._xy.index_put_((mask.nonzero(as_tuple=True)[0][sort_idx],), xy)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @torch.no_grad()
    def set_opacity(self, opacity, eps=1e-4, mask=None, sort_idx=None):
        assert opacity.shape[-1] == 1
        if mask is None: mask = torch.ones_like(model.get_rgb[..., 0]).bool()
        self._opacity.index_put_((mask.nonzero(as_tuple=True)[0][sort_idx],), self.opacity_inverse_activation(torch.clamp(opacity, eps, 1-eps)))

    @property
    def get_covariance(self):
        theta = self.get_rotation
        N = theta.shape[0]
        R = torch.cat([torch.cos(theta), -torch.sin(theta), torch.sin(theta), torch.cos(theta)], dim=-1).reshape(N, 2, 2).to(self.device)
        S = torch.diag_embed(self.get_scale).to(self.device)
        RS = torch.bmm(R, S)
        return torch.bmm(RS, RS.permute(0, 2, 1))

    def get_proj_gaussians(self, n, b, x):
        """
        n : (2,) unit vector
        b : scalar
        n = (p, q)
        px + qy + b = 0
        """
        line = misc.normal2dir(n)
        bias = misc.get_bias(n, b)
        xy = self.get_xy
        proj_xy = mm(xy, line[..., None]) * line[None, ...] + bias[None, ...]
        distance = torch.sqrt(((xy - proj_xy) ** 2).sum(dim=-1))
        N = proj_xy.shape[0]
        cov = self.get_covariance
        line = line.view(1, 2).repeat(N, 1) # (N, 2)
        n = n.view(1,2).repeat(N, 1)
        proj_var = torch.bmm(line.view(N, 1, 2), torch.bmm(cov, line.view(N, 2, 1)))
        near_cull_mask = distance >= 0.11
        back_cull_mask = (torch.bmm(xy.view(N, 1, 2), n.view(N, 2, 1)).squeeze() + b) < 0
        side_cull_mask = torch.sqrt(((proj_xy - bias[None, ...]) ** 2).sum(dim=-1)) < x.max() + 1.5
        mask = near_cull_mask & back_cull_mask & side_cull_mask
        return proj_xy, proj_var, distance, mask
    
    def render(self, x, n, b):
        """
        x : (k,) coefficients 
        """
        line = misc.normal2dir(n)
        bias = misc.get_bias(n, b)
        xs = line[None, ...] * x[..., None] + bias
        y = torch.zeros_like(x)[..., None] # (k, 1)
        T = torch.ones_like(x)[..., None]
        mu_, var_, distance, mask = self.get_proj_gaussians(n, b, x)
        sort_idx = torch.argsort(distance[mask])
        mu_, var_ = mu_[mask][sort_idx], var_[mask][sort_idx]
        rgb_ = self.get_rgb
        opacity_ = self.get_opacity
        opacity_, rgb_ = opacity_[mask][sort_idx], rgb_[mask][sort_idx]
        params = {'sort_idx': sort_idx, 'mask': mask, "proj_mu": mu_, "proj_var": var_, "xs" : xs}
        srt_params = {"w": [], "T": [], "g": [], "alpha": []}

        for mu, var, opacity, rgb in zip(mu_, var_, opacity_, rgb_):
            g = misc.eval_normal_1d(xs, mu, var)
            alpha = opacity * g
            y = y + T * alpha * rgb[None, ...]
            weight = alpha * T 
            srt_params["w"] += [weight]
            srt_params["T"] += [T]
            srt_params["g"] += [g]
            srt_params["alpha"] += [alpha]
            T = T * (1-alpha)
        
        N, k = self.get_opacity.shape[0], len(x)
        for key, srt_param in srt_params.items():
            stacked = torch.stack(srt_param, dim=0)
            params[key] = torch.zeros(N, k, stacked.shape[-1]).to(stacked.device).index_put_((mask.nonzero(as_tuple=True)[0][sort_idx],), stacked)
        
        return y, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=2024) 
    parser.add_argument('--method', '-m', type=str, required=True) 
    parser.add_argument('--iteration', '-i', type=int, default=1000) 
    parser.add_argument('--exp_name', '-e', type=str, default="temp") 
    parser.add_argument('--bias', '-b', type=int, default=20)
    parser.add_argument('--xmin', '-x', type=int, default=15)
    parser.add_argument('--viz_train', action='store_true')
    parser.add_argument('--plot_border', '-pb', type=int, default=35)
    args = parser.parse_args() 
    misc.set_seed(args.seed)
    METHOD = args.method.upper()
    save_dir = f"fig/{args.exp_name}"
    os.makedirs(save_dir, exist_ok=True)

    N = 89
    xmin, xmax = -args.xmin, args.xmin
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
        fixed = ['opacity', 'rotation', 'scale']
        noise = []
        gt = True
    elif METHOD == "OURS":
        fixed = ['xy', 'opacity', 'scale']
        gt = True
    else:
        raise NotImplementedError
    gt_model = Gaussian2DModel(N, gt=gt)
    model = Gaussian2DModel(N, iteration=iteration, fixed=fixed, noise=noise, gt_model=gt_model)
    p_init = misc.draw_model(model, None, None, [xmin, xmax], "init", plot_border=args.plot_border)
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
        ### optimizer
        lr_xy = lr['xy'] if 'xy' not in fixed else 0.
        
        lr_opacity = lr['opacity'] if 'opacity' not in fixed else 0.
        lr_scale = lr['scale'] if 'scale' not in fixed else 0.
        lr_rotation = lr['rotation'] if 'rotation' not in fixed else 0.
        # lr_rgb = lr['rgb'] if 'rgb' not in fixed else 0.
        l = [
            {'params': [model._xy], 'lr': lr_xy},
            {'params': [model._opacity], 'lr': lr_opacity},
            {'params': [model._scale], 'lr': lr_scale},
            {'params': [model._rotation], 'lr': lr_rotation},
            # {'params': [model._rgb], 'lr': lr_rgb},
        ]
        optimizer = torch.optim.Adam(l, eps=1e-15)
        #####################
        stack = list(range(len(images)))
        random.shuffle(images)
        sub_iter = 50
        pbar = trange(iteration // sub_iter)
        for i in pbar:
            for _ in range(sub_iter):
                if stack == []:
                    stack = list(range(len(images)))
                    random.shuffle(images)
                normal, bias, y = images[stack.pop()]
                y_pred, params = model.render(data, normal, bias)
                loss = ((y - y_pred)**2).mean()
                pbar.set_postfix({"loss":"%.3f" % loss.item()})
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    

            y_list, y_pred_list, w_list = [], [], []
            for i in trange(len(images), leave=False):
                normal, bias, y = images[i] # y : (k, 3)
                y_pred, params = model.render(data, normal, bias) 
                y_list.append(y)
                y_pred_list.append(y_pred)
                w_list.append(params['w'])
            y = torch.cat(y_list, dim=0) # (k, 3)
            y_pred = torch.cat(y_pred_list, dim=0)

            w = torch.cat(w_list, dim=1).permute(2, 0, 1) # (1, N, k)

            c, o = model.get_rgb.permute(1, 0)[..., None], model.get_opacity.permute(1, 0)[..., None] # (3, N, 1)
            r = (y_pred - y).permute(1, 0).unsqueeze(1) # (3, 1, k)
            ######## rgb
            A_rgb = mm(w.permute(2, 1, 0), w.permute(2, 0, 1)).sum(dim=0) # (N, N)
            b_rgb = mm(w.repeat(3, 1, 1), y.permute(1, 0)[..., None]).squeeze(-1).permute(1, 0) # (N, 3)
            result_rgb = mm(torch.inverse(A_rgb), b_rgb) 
            model.set_rgb(result_rgb)
            ############

    
    else:
        g_mean_approx = MeanApproxNet().cuda().eval()
        g_mean_approx.load_state_dict(torch.load("mean.pt"))
        with torch.no_grad():
            if args.iteration == -1:
                iteration = 5
                for it in trange(iteration, leave=False):
                    y_list, y_pred_list, params_list = [], [], []
                    for i in trange(len(images), leave=False):
                        normal, bias, y = images[i] # y : (k, 3)
                        y_pred, params = model.render(data, normal, bias) 
                        y_list.append(y)
                        y_pred_list.append(y_pred)
                        params_list.append(params)
                    y = torch.cat(y_list, dim=0) # (k, 3)
                    y_pred = torch.cat(y_pred_list, dim=0)
                    params = misc.concat_dictlist(params_list)

                    
                    c, o = model.get_rgb.permute(1, 0)[..., None], model.get_opacity.permute(1, 0)[..., None] # (3, N, 1)
                    r = (y_pred - y).permute(1, 0).unsqueeze(1) # (3, 1, k)
                    ######## rgb
                    A_rgb = mm(w.permute(2, 1, 0), w.permute(2, 0, 1)).sum(dim=0) # (N, N)
                    b_rgb = mm(w.repeat(3, 1, 1), y.permute(1, 0)[..., None]).squeeze(-1).permute(1, 0) # (N, 3)
                    result_rgb = mm(torch.inverse(A_rgb), b_rgb) 
                    model.set_rgb(result_rgb)
                    ############
            else:
                stack = list(range(len(images)))
                random.shuffle(images)
                pbar = trange(iteration)
                for i in pbar:
                    if stack == []:
                        stack = list(range(len(images)))
                        random.shuffle(images)
                    normal, bias, y = images[stack.pop()]
                    y_pred, params = model.render(data, normal, bias) # (k, 3)
                    sort_idx, mask = params['sort_idx'], params['mask']
                    params = misc.mask_params(params, mask, sort_idx, (2,0,1))

                    (w, T, g, alpha) = (params['w'], params['T'], params['g'], params['alpha']) # (1, N, k)
                    c = model.get_rgb[mask][sort_idx].permute(1, 0)[..., None] # (3, N, 1)
                    o = model.get_opacity[mask][sort_idx].permute(1, 0)[..., None] # (3, N, 1)
                    mu = model.get_xy[mask][sort_idx].permute(1, 0)[..., None] # (2, N, 1)

                    proj_mu, proj_var = params['proj_mu'].permute(1, 0)[..., None], params['proj_var'].squeeze(-1).permute(1, 0)[..., None] # (2, N, 1), (1, N, 1)
                    xs = params['xs'].permute(1, 0).unsqueeze(1) # (2, 1, k)
                    
                    r = (y_pred - y).permute(1, 0).unsqueeze(1) # (3, 1, k)

                    ######## rgb
                    A_rgb = mm(w.permute(2, 1, 0), w.permute(2, 0, 1)).sum(dim=0) # (N, N)
                    b_rgb = mm(w.repeat(3, 1, 1), y.permute(1, 0)[..., None]).squeeze(-1).permute(1, 0) # (N, 3)
                    result_rgb = mm(torch.inverse(A_rgb), b_rgb) 
                    model.set_rgb(result_rgb, mask=mask, sort_idx=sort_idx)
                    ##########################

                    ######## opacity
                    # dT_do = -T.unsqueeze(1) * (g / (1 - alpha)).unsqueeze(2) # (1, Ni, Nj, k)
                    # dT_do = dT_do.permute(0, 3, 1, 2).triu(diagonal=1).permute(0, 2, 3, 1) # (1, Ni, Nj, k)
                    # dhatC_do = (T * g * c) + ((g * c * o).unsqueeze(1) * dT_do).sum(dim=2) # (3, Ni, k)
                    # A_opacity = (dhatC_do.unsqueeze(2) * (T * g * c).unsqueeze(1)).sum(dim=-1) # (3, Ni, Nj)
                    # b_opacity = (y.permute(1, 0)[:, None, :] * dhatC_do.squeeze(2)).sum(dim=-1, keepdim=True) # (3, Ni, 1)
                    # result_opacity = mm(torch.inverse(A_opacity), b_opacity).mean(dim=0) # (N, 1)
                    # print(result_opacity.squeeze())
                    # model.set_opacity(result_opacity)
                    ##########################

                    ######## mean
                    # d = misc.normal2dir(normal) # (2, )
                    # F = g_mean_approx(
                    #         torch.cat([
                    #             d[:, None, None].repeat(1, proj_var.shape[1], xs.shape[-1]), 
                    #             xs.repeat(1, proj_var.shape[1], 1),
                    #             proj_var.repeat(1, 1, xs.shape[-1])
                    #         ], dim=0).permute(1, 2, 0).reshape(-1, 5)
                    #     ).permute(1, 0).reshape(2, proj_var.shape[1], xs.shape[-1])  # (2, N, k)
                    # mu_bar = proj_mu - xs # (2, N, k)
                    # dT_dmu = ((o * g / (1 - o * g)) * (mu_bar / proj_var)).unsqueeze(2) * T.unsqueeze(1)  # (2, Ni, Nj, k)
                    # p = (T * c * o) # (3, N, k)
                    # q = (proj_var.unsqueeze(2) * dT_dmu * o.unsqueeze(1)).unsqueeze(1) * c.unsqueeze(0).unsqueeze(2) # (2, 3, Ni, Nj, k)
                    # a = r * p # (3, N, k)
                    # b = r[None, :, :, None] * q # (2, 3, Ni, Nj, k)

                    # U = mm(
                    #     (xs.unsqueeze(0) * a.unsqueeze(1)).permute(0, 2, 3, 1).unsqueeze(-1), # (3, N, k, 2, 1)
                    #     F.unsqueeze(0).repeat(3, 1, 1, 1).permute(0, 2, 3, 1).unsqueeze(-2) # (3, N, k, 1, 2)
                    # ).sum(dim=2) # (3, N, 2, 2)
                    # # (3, 2N, 2N)
                    # # (3, k, Ni, Nj, 2, 1)
                    # V = mm(
                    #     b.permute(1, 4, 2, 3, 0).unsqueeze(-1), # (3, k, Ni, Nj, 2, 1)
                    #     F.unsqueeze(0).repeat(3, 1, 1, 1).permute(0, 3, 2, 1).unsqueeze(2).unsqueeze(-2).repeat(1, 1, F.shape[1], 1, 1, 1) # (3, k, Ni, Nj, 1, 2)
                    # ).sum(dim=1).permute(0, 1, 3, 2, 4).reshape(3, 2*F.shape[1], 2*F.shape[1]) # (3, 2N, 2N)
                    # U_diag = torch.stack([
                    #     torch.block_diag(*list(map(lambda d:d.squeeze(), U[0].split(1, dim=0)))),
                    #     torch.block_diag(*list(map(lambda d:d.squeeze(), U[1].split(1, dim=0)))),
                    #     torch.block_diag(*list(map(lambda d:d.squeeze(), U[2].split(1, dim=0))))
                    # ]) # (3, 2N, 2N)
                    # A_xy = V + U_diag # (3, 2N, 2N)
                    # e = ((a * g).unsqueeze(0) * proj_mu.unsqueeze(1)).sum(-1).permute(1, 0, 2).reshape(3, -1).unsqueeze(-1) # (3, 2N, 1)
                    # result_xy = mm(A_xy, e).mean(dim=0).reshape(-1, 2) # (N, 2)
                    # model.set_xy(result_xy, mask=mask, sort_idx=sort_idx)
                    
                    ##########################        

    #####################

    ### Visualization
    for mode, slope in zip(('train', 'eval'), (train_slope, eval_slope)):
        if mode == 'train' and not args.viz_train: continue
        with torch.no_grad():
            pred_vecs, gt_vecs = [], []
            for i, (n_x, n_y, bias) in tqdm(list(enumerate(slope)), desc=f"{mode} vis..."):
                normal = torch.tensor([n_x,n_y]).cuda().float()
                normal = normal / torch.norm(normal)
                p_pred = misc.draw_model(model, normal, bias, [xmin, xmax], "predict", plot_border=args.plot_border)
                p_gt = misc.draw_model(gt_model, normal, bias, [xmin, xmax], "GT", plot_border=args.plot_border)
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

