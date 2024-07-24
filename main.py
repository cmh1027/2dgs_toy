import torch
import sys
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
import minhyuk
import sys
import time
import argparse

class Gaussian2DModel:
    def __init__(self, N, iteration=None, lr=None, gt=False, fixed=[], gt_model=None):
        device = self.device = torch.device('cuda')
        self.scale_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.opacity_inverse_activation = misc.inverse_sigmoid
        self.rgb_activation = torch.sigmoid
        self.rgb_inverse_activation = misc.inverse_sigmoid
        # self.rgb_activation = lambda x: x
        # self.rgb_inverse_activation = lambda x: x
        self.rotation_activation = torch.tanh
        self._xy = torch.nn.Parameter(((torch.rand(N, 2) - 0.5) * 10 + 3).to(device)) # mean
        self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(misc.generate_random_color(N)[torch.randperm(N)].to(device)))
        self._scale = torch.nn.Parameter(torch.rand(N, 2).to(device) + 0.2)
        self._rotation = torch.nn.Parameter((torch.rand(N, 1).to(device) - 0.5) * 2 * torch.pi) # unlike quaternion in 3D, theta is enough
        self._opacity = torch.nn.Parameter((torch.rand(N, 1).to(device) - 0.5) * 2)
        self.fixed = fixed
        if gt:
            if N == 2:
                self._xy = torch.nn.Parameter(torch.tensor([[2., 5.], [5., 2.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.1, 0.9, 0.9], [0.9, 0.9, 0.1]])).to(device))
                self._scale = torch.nn.Parameter(torch.tensor([[0.4, 1.2], [0.9, 1.1]]).to(device))
                self._rotation = torch.nn.Parameter(torch.tensor([[1.3], [-1.6]]).to(device)) 
                self._opacity = torch.nn.Parameter(torch.tensor([[3.0], [-1.0]]).to(device))
            elif N == 3:
                self._xy = torch.nn.Parameter(torch.tensor([[3., 3.], [6., 3.], [0.2, 6.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])).to(device))
                self._scale = torch.nn.Parameter(torch.tensor([[0.5, 0.9], [0.4, 1.2], [0.9, 1.1]]).to(device))
                self._rotation = torch.nn.Parameter(torch.tensor([[1.3], [-1.6], [2.2]]).to(device)) 
                self._opacity = torch.nn.Parameter(torch.tensor([[5.0], [3.0], [-1.0]]).to(device))
            elif N == 4:
                self._xy = torch.nn.Parameter(torch.tensor([[2., 2.], [8., 3.], [2, 6.], [6, 2.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9], [0.9, 0.1, 0.9]])).to(device))
                self._scale = torch.nn.Parameter(torch.tensor([[0.5, 0.9], [0.4, 1.2], [0.9, 1.1], [0.7, 0.3]]).to(device))
                self._rotation = torch.nn.Parameter(torch.tensor([[1.3], [-1.6], [2.2], [-0.3]]).to(device)) 
                self._opacity = torch.nn.Parameter(torch.tensor([[3.0], [1.0], [-1.0], [2.0]]).to(device))   
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

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xy_scheduler_args = misc.get_expon_lr_func(lr_init=0.0001,
                                                         lr_final=0.0000016,
                                                         lr_delay_mult=0.01,
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
        bias = misc.get_bias(n, b)
        proj_xy = mm(xy, line[..., None]) * line[None, ...] + bias[None, ...]
        distance = torch.sqrt(((xy - proj_xy) ** 2).sum(dim=-1))
        N = proj_xy.shape[0]
        cov = self.get_covariance
        line = line.view(1, 2).repeat(N, 1) # (N, 2)
        n = n.view(1,2).repeat(N, 1)
        proj_var = torch.bmm(line.view(N, 1, 2), torch.bmm(cov, line.view(N, 2, 1)))
        near_cull_mask = distance >= 0.001
        back_cull_mask = (torch.bmm(xy.view(N, 1, 2), n.view(N, 2, 1)).squeeze() + b) > 0
        mask = torch.logical_and(near_cull_mask, back_cull_mask)

        return proj_xy, proj_var, distance, mask
    
    def render(self, x, n, b, precomp_opacity=None, precomp_rgb=None):
        """
        x : (k,) coefficients 
        """
        y = torch.zeros_like(x)[..., None] # (k, 1)
        T = torch.ones_like(x)[..., None]
        mu_, var_, distance, _ = self.get_proj_gaussians(n, b)
        sort_idx = torch.argsort(distance)
        mu_, var_ = mu_[sort_idx], var_[sort_idx]
        opacity_ = self.get_opacity if precomp_opacity is None else precomp_opacity
        rgb_ = self.get_rgb if precomp_rgb is None else precomp_rgb
        opacity_, rgb_ = opacity_[sort_idx], rgb_[sort_idx]
        line = misc.normal2dir(n)
        bias = misc.get_bias(n, b)
        xs = line[None, ...] * x[..., None] + bias
        params = {}
        sorted_weights = []
        sorted_Ts = []
        sorted_gs = []
        sorted_alphas = []
        for mu, var, opacity, rgb in zip(mu_, var_, opacity_, rgb_):
            g = misc.eval_normal_1d(xs, mu, var)
            alpha = opacity * g
            y = y + T * alpha * rgb[None, ...]
            weight = alpha * T # (k, 1)
            sorted_weights.append(weight)
            sorted_Ts.append(T)
            sorted_gs.append(g)
            sorted_alphas.append(alpha)
            T = T * (1-alpha)
        sorted_weights = torch.stack(sorted_weights, dim=0)
        sorted_Ts = torch.stack(sorted_Ts, dim=0)
        sorted_gs = torch.stack(sorted_gs, dim=0)
        sorted_alphas = torch.stack(sorted_alphas, dim=0)

        weights = torch.zeros_like(sorted_weights)
        Ts = torch.zeros_like(sorted_Ts)
        gs = torch.zeros_like(sorted_gs)
        alphas = torch.zeros_like(sorted_alphas)

        weights[sort_idx] = sorted_weights # (N, k, 1)
        Ts[sort_idx] = sorted_Ts # (N, k, 1)
        gs[sort_idx] = sorted_gs # (N, k, 1)
        alphas[sort_idx] = sorted_alphas # (N, k, 1)

        params["w"] = weights
        params["T"] = Ts
        params["g"] = gs
        params["alpha"] = alphas
        params["sort_idx"] = sort_idx
        return y, params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=2024) 
    parser.add_argument('--method', '-m', type=str, required=True) 
    parser.add_argument('--iteration', '-i', type=int, default=1000) 
    args = parser.parse_args() 
    misc.set_seed(args.seed)
    METHOD = args.method.upper()
    N = 9
    xmin, xmax = -6, 6
    slope_min, slope_max, slope_N = 1, 3, 25
    slope_list = list(zip(np.linspace(slope_min, slope_max, slope_N), np.linspace(slope_max, slope_min, slope_N), np.random.rand(slope_N) * 4 + 1))
    # slope_list.extend(list(zip(np.linspace(slope_min, slope_max, slope_N), np.linspace(slope_max, slope_min, slope_N), -np.random.rand(slope_N) * 2 - 10)))
    data = torch.arange(xmin, xmax, 0.05).cuda()
    iteration = args.iteration
    lr = {
        'xy': 0.1,
        'rgb': 0.01,
        'opacity': 0.05,
        'scale': 0.1,
        'rotation': 0.5
    }
    
    ### fixed ['rgb', 'xy', 'scale', 'rotation' 'opacity']
    if METHOD == "GD":
        fixed = []
        gt = False
    elif METHOD == "BFGS":
        fixed = ['xy', 'scale', 'rotation']
        gt = False
    elif METHOD == "EM":
        fixed = ['xy', 'scale', 'rotation']
        gt = True
    else:
        raise NotImplementedError
    gt_model = Gaussian2DModel(N, gt=gt)
    model = Gaussian2DModel(N, iteration=iteration, lr=lr, fixed=fixed, gt_model=gt_model)
    p_init = misc.draw_model(model, None, None, [xmin, xmax], "init")
    Image.fromarray(misc.draw_model(model, None, None, [xmin, xmax])).save("fig/plot_init.png")
    Image.fromarray(misc.draw_model(gt_model, None, None, [xmin, xmax])).save("fig/plot_init_gt.png")
    images = []
    ### Data preparation
    with torch.no_grad():
        for i, (n_x, n_y, bias) in enumerate(slope_list):
            normal = torch.tensor([n_x,n_y]).cuda().float()
            normal = normal / torch.norm(normal)
            y, _ = gt_model.render(data, normal, bias)
            images.append((normal, bias, y))
    if METHOD == "GD":
        stack = list(range(len(images)))
        random.shuffle(images)
        pbar = trange(iteration)
        start = time.time()
        for i in pbar:
            if stack == []:
                stack = list(range(len(images)))
                random.shuffle(images)
            normal, bias, y = images[stack.pop()]
            y_pred, _ = model.render(data, normal, bias)
            loss = ((y - y_pred)**2).mean()
            pbar.set_postfix({"loss":loss.item()})
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
        print(time.time() - start)
            
    elif METHOD == "BFGS":
        optimizers = {}
        for param in ['xy', 'scale', 'rotation', 'opacity']:
            if param not in fixed:
                optimizer = torch.optim.LBFGS([getattr(model, "_" + param)], max_iter=10, tolerance_grad=1e-4, tolerance_change=1e-4)
                optimizers[param] = optimizer
        start = time.time()
        for i in trange(10):
            if 'rgb' not in fixed:
                with torch.no_grad():
                    ys, ws = [], []
                    for i in range(len(images)):
                        normal, bias, y = images[i] # y : (k, 3)
                        ys.append(y)
                        _, params = model.render(data, normal, bias) 
                        ws.append(params['w']) # (N, k, 1) 
                    y = torch.cat(ys, dim=0)
                    w = torch.cat(ws, dim=1)
                    if 'rgb' not in fixed:
                        Y_c = (w * y[None, ...].repeat(N, 1, 1)).permute(1, 0, 2).mean(dim=0) # (k, N, 3)
                        X_c = (w.permute(1, 0, 2) * w.permute(1, 2, 0)).mean(dim=0) # (k, N, N)
                        result_rgb = mm(torch.inverse(X_c), Y_c) 
                        model.set_rgb(result_rgb)

            for param in ['xy', 'scale', 'rotation', 'opacity']:
                if param in fixed: continue
                optimizers[param].zero_grad()
                def closure():
                    ys, y_preds, = [], []
                    for i in range(len(images)):
                        normal, bias, y = images[i]
                        y_pred, _ = model.render(data, normal, bias)
                        ys.append(y)
                        y_preds.append(y_pred)
                    y = torch.cat(ys, dim=0)
                    y_pred = torch.cat(y_preds, dim=0)
                    loss = ((y - y_pred)**2).mean()
                    loss.backward()
                    return loss
                optimizers[param].step(closure)

        print(time.time() - start)

    elif METHOD == "EM":
        iteration = 10 if len(fixed) <= 3 else 1
        optim_params = list(set(lr.keys()).difference(set(fixed)))
        for it in trange(iteration, leave=False):
            ys, y_preds, ws, Ts, gs, sort_idx = [], [], [], [], [], []
            for i in trange(len(images), leave=False):
                normal, bias, y = images[i] # y : (k, 3)
                ys.append(y)
                y_pred, params = model.render(data, normal, bias) 
                y_preds.append(y_pred)
                ws.append(params['w']) # (N, k, 1) 
                Ts.append(params['T']) # (N, k, 1) 
                gs.append(params['g']) # (N, k, 1) 
                sort_idx.append(params['sort_idx']) # (I, N)
            y = torch.cat(ys, dim=0)
            y_pred = torch.cat(y_preds, dim=0)
            w = torch.cat(ws, dim=1)
            # T = torch.stack(Ts, dim=0) # (I, N, k//I, 1)
            g = torch.stack(gs, dim=0) # (I, N, k//I, 1)
            sort_idx = torch.stack(sort_idx, dim=0) # (I, N)

            if 'rgb' not in fixed and it % len(optim_params) == optim_params.index('rgb'):
                Y_c = (w * y[None, ...].repeat(N, 1, 1)).permute(1, 0, 2).mean(dim=0) # (k, N, 3)
                X_c = (w.permute(1, 0, 2) * w.permute(1, 2, 0)).mean(dim=0) # (k, N, N)
                result_rgb = mm(torch.inverse(X_c), Y_c) 
                model.set_rgb(result_rgb)

            if 'opacity' not in fixed and it % len(optim_params) == optim_params.index('opacity'):
                y_ = y.permute(1, 0) # (3, k)
                c_ = model.get_rgb.permute(1, 0) # (3, N)
                # g_ = g.permute(2, 0, 1) # (1, N, k)

                x = (torch.ones(N, 1) * 0.5).cuda()
                # x = (torch.rand(N, 1) * 0.7 + 0.3).cuda() # initialize
                # x = model.get_opacity
                tolerance = 1e-3
                damp = 0.01
                damp_factor = 2
                for i in trange(100, desc="LM Method", leave=False):
                    ############
                    # g_ = g.permute(3, 1, 0, 2).reshape(1, N, -1)
                    # T_ = torch.ones_like(g_) 
                    # T_[0, 1:] = torch.cumprod(1 - x[..., None] * g_, dim=1)[0, :-1] # (1, N, k)
                    I, _, k_I, _ = g.shape
                    bprod = 1 - b_extract(x[None, ..., None].repeat(I, 1, k_I, 1), sort_idx) * b_extract(g, sort_idx)                
                    prod = torch.cumprod(bprod, dim=1) # (I, N, k//I, 1)
                    prod = torch.cat((torch.ones_like(prod[:, 0:1, :, :]), prod[:, :-1, ...]), dim=1)
                    T_ = b_assign(prod, sort_idx).permute(3, 1, 0, 2).reshape(1, N, -1) # (1, N, k)
                    g_ = g.permute(3, 1, 0, 2).reshape(1, N, -1) # (1, N, k)
                    ############

                    ############
                    f_x_r = minhyuk.cost_opacity(y[0], x[..., 0], c_[0].contiguous(), T_[0], g_[0]) # (N,)
                    f_x_g = minhyuk.cost_opacity(y[1], x[..., 0], c_[1].contiguous(), T_[0], g_[0])
                    f_x_b = minhyuk.cost_opacity(y[2], x[..., 0], c_[2].contiguous(), T_[0], g_[0])
                    f_x = torch.stack([f_x_r, f_x_g, f_x_b], dim=0) # (3, N)
                    F_x = torch.stack([(f_x_r**2).sum(), (f_x_g**2).sum(), (f_x_b**2).sum()])[..., None] # (3, 1)
                    # P = T_ * g_ * c_.unsqueeze(2) # (3, Nj, k)
                    # dT_do = -T_.unsqueeze(1) * (g_ / (1 - g_ * x[None, ...])).unsqueeze(2) # (1, Ni, Nj, k)
                    # dT_do = dT_do.permute(0, 3, 1, 2).triu(diagonal=1).permute(0, 2, 3, 1) # (1, Ni, Nj, k)
                    # Q = dT_do * g_.unsqueeze(2) * c_[..., None, None] # (3, Ni, Nj, k)
                    # P_repeat = P.unsqueeze(1).repeat(1, N, 1, 1) # (3, Ni, Nj, k)
                    # A = 0.5 * (mm(P_repeat, Q.permute(0, 1, 3, 2)) + mm(Q, P_repeat.permute(0, 1, 3, 2))) # (3, Ni, Nj, Nj)
                    # b = mm(P, P.permute(0, 2, 1)) - (y_[:, None, None, :] * Q).sum(dim=-1) # (3, Ni, Nj)
                    # z = -(y_.unsqueeze(1) * T_ * g_ * c_.unsqueeze(2)).sum(-1) # (3, Ni)
                    # f_x = misc.F(x, A, b, z).squeeze(-1) # (3, N, 1)
                    # F_x = f_x.sum(dim=1, keepdim=True) # (3, 1)
                    ############

                    ############
                    # J_x = misc.J(x, A, b, z).squeeze(-1) # (3, Nj)
                    df_do_r = minhyuk.derv_opacity(y[0], x[..., 0], c_[0].contiguous(), T_[0], g_[0]) # (N, N)
                    df_do_g = minhyuk.derv_opacity(y[1], x[..., 0], c_[1].contiguous(), T_[0], g_[0]) # (N, N)
                    df_do_b = minhyuk.derv_opacity(y[2], x[..., 0], c_[2].contiguous(), T_[0], g_[0]) # (N, N)
                    df_do = torch.stack([df_do_r, df_do_g, df_do_b], dim=0) # (3, N, N)
                    J_x = 2 * torch.matmul(f_x.unsqueeze(1), df_do).squeeze(1) # (3, 1, N) * (3, N, N) = (3, N)
                    ############
                    
                    JTJ = (J_x * J_x).sum(dim=-1, keepdim=True) # (3, 1)
                    M = (damp+1) * JTJ # (3, 1)
                    gradient = J_x * F_x # (3, Nj)
                    step = (-gradient / M).mean(dim=0)[..., None] # (N, 1)
                    x_new = x + step

                    ############
                    # f_x_new = misc.F(x_new, A, b, z).squeeze(-1) # (3, N, 1)
                    # F_x_new = f_x_new.sum(dim=1, keepdim=True) # (3, 1)
                    F_x_new = torch.stack([
                                (minhyuk.cost_opacity(y[0], x_new[..., 0], c_[0].contiguous(), T_[0], g_[0])**2).sum(),
                                (minhyuk.cost_opacity(y[1], x_new[..., 0], c_[1].contiguous(), T_[0], g_[0])**2).sum(),
                                (minhyuk.cost_opacity(y[2], x_new[..., 0], c_[2].contiguous(), T_[0], g_[0])**2).sum()
                            ])[..., None] # (3, 1)
                    ############

                    norm = F_x.mean(dim=0)
                    norm_new = F_x_new.mean(dim=0)
                    ic(x, x_new, norm, norm_new, damp)
                    ic(torch.norm(gradient).item())

                    if norm > norm_new:
                        x = x_new
                        damp = damp / damp_factor
                    else:
                        damp = damp * damp_factor
                    if norm < tolerance:
                        ic(norm.item())
                        break
                    if torch.norm(step) < tolerance:
                        ic(torch.norm(step).item())
                        break
                    if torch.norm(gradient) < tolerance:
                        ic(torch.norm(gradient).item())
                        break
                    

                result_opacity = x
                model.set_opacity(result_opacity)

                # with torch.no_grad():
                #     ys, y_preds = [], []
                #     for i in trange(len(images), leave=False):
                #         normal, bias, y = images[i] # y : (k, 3)
                #         ys.append(y)
                #         y_pred, params = model.render(data, normal, bias, precomp_opacity=x)
                #         y_preds.append(y_pred)
                #     y = torch.cat(ys, dim=0)
                #     y_pred = torch.cat(y_preds, dim=0)
                #     ic(((y - y_pred) ** 2).mean().item())

    # eval
    with torch.no_grad():
        for i, (n_x, n_y, bias) in enumerate(slope_list):
            normal = torch.tensor([n_x,n_y]).cuda().float()
            normal = normal / torch.norm(normal)
            p_pred = misc.draw_model(model, normal, bias, [xmin, xmax], "predict")
            p_gt = misc.draw_model(gt_model, normal, bias, [xmin, xmax], "GT")
            h, w, _ = p_pred.shape
            p = np.zeros((h, 3*w+20, 3), dtype=np.uint8)
            p[:, 0:w], p[:, w+10:2*w+10], p[:, 2*w+20:] = p_init, p_pred, p_gt
            Image.fromarray(p).save(f"fig/plot{'%03d' % i}.png")
            r_pred = misc.line2image(model.render(data, normal, bias)[0], f"fig/render_{'%03d' % i}.png")
            r_gt = misc.line2image(gt_model.render(data, normal, bias)[0], f"fig/render_gt_{'%03d' % i}.png")
            h, w, _ = r_pred.shape
            r = np.zeros((h, 2*w+10, 3), dtype=np.uint8)
            r[:, 0:w], r[:, w+10:] = r_pred, r_gt
            Image.fromarray(r).save(f"fig/render{'%03d' % i}.png")


