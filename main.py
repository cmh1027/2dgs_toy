import torch
import sys
sys.path.append(".")
import misc
import numpy as np
from tqdm import trange, tqdm
import random
from PIL import Image
import copy
class Gaussian2DModel:
    def __init__(self, N, iteration=None, lr=None, gt=False, fixed=[], gt_model=None):
        device = self.device = torch.device('cuda')
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        # self.rgb_activation = torch.sigmoid
        # self.rgb_inverse_activation = misc.inverse_sigmoid
        self.rgb_activation = lambda x: x
        self.rgb_inverse_activation = lambda x: x
        self.rotation_activation = torch.tanh
        self._xy = torch.nn.Parameter(((torch.rand(N, 2) - 0.5) * 5 + torch.rand(N, 1) * 5 + 2).to(device)) # mean
        self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(misc.generate_random_color(N)[torch.randperm(N)].to(device)))
        self._scaling = torch.nn.Parameter(torch.rand(N, 2).to(device) + 0.2)
        self._rotation = torch.nn.Parameter(torch.zeros(N, 1).to(device)) # unlike quaternion in 3D, theta is enough
        self._opacity = torch.nn.Parameter(torch.rand(N, 1).to(device) + 1.5)
        self.fixed = fixed
        if gt:
            if N == 2:
                self._xy = torch.nn.Parameter(torch.tensor([[2., 5.], [5., 2.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.1, 0.9, 0.9], [0.9, 0.9, 0.1]])).to(device))
                self._scaling = torch.nn.Parameter(torch.tensor([[0.4, 1.2], [0.9, 1.1]]).to(device))
                self._rotation = torch.nn.Parameter(torch.tensor([[1.3], [-1.6]]).to(device)) 
                self._opacity = torch.nn.Parameter(torch.tensor([[2.3], [1.6]]).to(device))
            if N == 3:
                self._xy = torch.nn.Parameter(torch.tensor([[3., 3.], [6., 3.], [2., 6.]]).to(device))
                self._rgb = torch.nn.Parameter(self.rgb_inverse_activation(torch.tensor([[0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]])).to(device))
                self._scaling = torch.nn.Parameter(torch.tensor([[0.5, 0.9], [0.4, 1.2], [0.9, 1.1]]).to(device))
                self._rotation = torch.nn.Parameter(torch.tensor([[1.3], [-1.6], [2.2]]).to(device)) 
                self._opacity = torch.nn.Parameter(torch.tensor([[2.3], [1.6], [0.7]]).to(device))
        else:
            if fixed != []:
                assert gt_model is not None
            if "rgb" in fixed:
                self._rgb = copy.deepcopy(gt_model._rgb)
            if "xy" in fixed:
                self._xy = copy.deepcopy(gt_model._xy)
            if "scale" in fixed:
                self._scaling = copy.deepcopy(gt_model._scaling)
            if "rotation" in fixed:
                self._rotation = copy.deepcopy(gt_model._rotation)
            if "opacity" in fixed:
                self._opacity = copy.deepcopy(gt_model._opacity)
        if lr is not None and iteration is not None:
            self.training_setup(lr, iteration)

    @property
    def get_rgb(self):
        return self.rgb_activation(self._rgb)

    def set_rgb(self, rgb):
        self._rgb = self.rgb_inverse_activation(rgb)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation) * torch.pi
    
    @property
    def get_xy(self):
        return self._xy
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

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
            {'params': [self._scaling], 'lr': lr_scale, "name": "scaling"},
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
        S = torch.diag_embed(self.get_scaling).to(self.device)
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
        proj_xy = torch.matmul(xy, line[..., None]) * line[None, ...] + bias[None, ...]
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
    
    def render(self, x, n, b):
        """
        x : (k,) coefficients 
        """
        y = torch.zeros_like(x)[..., None] # (k, 1)
        T = torch.ones_like(x)[..., None]
        mu_, var_, distance, _ = self.get_proj_gaussians(n, b)
        sort_idx = torch.argsort(distance)
        mu_, var_, opacity_, rgb_ = mu_[sort_idx], var_[sort_idx], self.get_opacity[sort_idx], self.get_rgb[sort_idx]
        line = misc.normal2dir(n)
        bias = misc.get_bias(n, b)
        xs = line[None, ...] * x[..., None] + bias
        sorted_weights = []
        for mu, var, opacity, rgb in zip(mu_, var_, opacity_, rgb_):
            alpha = opacity * misc.eval_normal_1d(xs, mu, var)
            y = y + T * alpha * rgb[None, ...]
            weight = alpha * T # (k, 1)
            sorted_weights.append(weight)
            T = T * (1-alpha)
        sorted_weights = torch.stack(sorted_weights, dim=0)
        weights = torch.zeros_like(sorted_weights)
        weights[sort_idx] = sorted_weights # (N, k, 1)
        return y, weights


if __name__ == "__main__":
    METHOD = "EM"
    N = 10
    xmin, xmax = -6, 6
    slope_min, slope_max, slope_N = 1, 3, 25
    slope_list = list(zip(np.linspace(slope_min, slope_max, slope_N), np.linspace(slope_max, slope_min, slope_N), np.random.rand(slope_N) * 4 + 1))
    # slope_list.extend(list(zip(np.linspace(slope_min, slope_max, slope_N), np.linspace(slope_max, slope_min, slope_N), -np.random.rand(slope_N) * 2 - 10)))
    x = torch.arange(xmin, xmax, 0.05).cuda()
    iteration = 1000
    lr = {
        'xy': 0.1,
        'rgb': 0.025,
        'opacity': 0.05,
        'scale': 0.01,
        'rotation': 0.1
    }
    
    ### fixed ['rgb', 'xy', 'scale', 'rotation' 'opacity']
    # fixed = ['xy', 'scale', 'rotation', 'opacity']
    if METHOD == "GD":
        fixed = []
        gt = False
    else:
        fixed = ['xy', 'scale', 'rotation', 'opacity']
        gt = True
    gt_model = Gaussian2DModel(N, gt=gt)
    model = Gaussian2DModel(N, iteration=iteration, lr=lr, fixed=fixed, gt_model=gt_model)
    p_init = Image.fromarray(misc.draw_model(model, None, None, [xmin, xmax])).save("fig/plot_init.png")
    p_init = Image.fromarray(misc.draw_model(gt_model, None, None, [xmin, xmax])).save("fig/plot_init_gt.png")
    data = []
    ### Data preparation
    with torch.no_grad():
        for i, (n_x, n_y, bias) in enumerate(slope_list):
            normal = torch.tensor([n_x,n_y]).cuda().float()
            normal = normal / torch.norm(normal)
            y, _ = gt_model.render(x, normal, bias)
            data.append((normal, bias, y))
    if METHOD == "GD":
        stack = list(range(len(data)))
        random.shuffle(data)
        pbar = trange(iteration)
        for i in pbar:
            if stack == []:
                stack = list(range(len(data)))
                random.shuffle(data)
            normal, bias, y = data[stack.pop()]
            y_pred, _ = model.render(x, normal, bias)
            loss = ((y - y_pred)**2).mean()
            pbar.set_postfix({"loss":loss.item()})
            loss.backward()
            model.optimizer.step()
            model.optimizer.zero_grad()
    elif METHOD == "EM":
        with torch.no_grad():
            assert set(fixed) == set(['xy', 'scale', 'rotation', 'opacity'])
            numer, denom = 0, 0
            for _ in trange(100):
                for i in trange(len(data), leave=False):
                    normal, bias, y = data[i] # y : (k, 3)
                    y_pred, weight = model.render(x, normal, bias) # weight : (N, k, 1)
                    rgb = model.get_rgb # (N, 3)
                    N = rgb.shape[0]
                    denom += (weight * weight).sum(dim=1) # (N, 1)
                    wc = weight * rgb.unsqueeze(1).repeat(1, weight.shape[1], 1)  # (N, k, 3)
                    numer += (weight * y[None, ...] + (wc - wc.sum(dim=0, keepdim=True)))).sum(dim=1) # (N, 3)
                rgb = numer / denom
                model.set_rgb(torch.clamp(model.get_rgb, 0, 1))

            # for _ in trange(10):
            #     X = torch.zeros(N, N).cuda()
            #     Y = torch.zeros(N, 3).cuda()
            #     for i in trange(len(data), leave=False):
            #         normal, bias, y = data[i] # y : (k, 3)
            #         y_pred, weight = model.render(x, normal, bias) # weight : (N, k, 1)
            #         Y += (weight * y[None, ...].repeat(N, 1, 1)).sum(dim=1)
            #         X += (weight.permute(1, 0, 2) * weight.permute(1, 2, 0)).sum(dim=0) # (k, N, 1) * (k, 1, N) = (k, N, N) => (N, N)
            #     result = torch.matmul(torch.inverse(X), Y)
            #     # if not torch.logical_and(0<=result, result<=1).all():
            #     #     result = torch.clamp(result, 0, 1)
            #     model.set_rgb(result)
    # eval
    with torch.no_grad():
        for i, (n_x, n_y, bias) in enumerate(slope_list):
            normal = torch.tensor([n_x,n_y]).cuda().float()
            normal = normal / torch.norm(normal)
            p_pred = misc.draw_model(model, normal, bias, [xmin, xmax])
            p_gt = misc.draw_model(gt_model, normal, bias, [xmin, xmax])
            h, w, _ = p_pred.shape
            p = np.zeros((h, 2*w+10, 3), dtype=np.uint8)
            p[:, 0:w], p[:, w+10:] = p_pred, p_gt
            Image.fromarray(p).save(f"fig/plot{'%03d' % i}.png")
            r_pred = misc.line2image(gt_model.render(x, normal, bias)[0], f"fig/render_gt_{'%03d' % i}.png")
            r_gt = misc.line2image(model.render(x, normal, bias)[0], f"fig/render_{'%03d' % i}.png")
            h, w, _ = r_pred.shape
            r = np.zeros((h, 2*w+10, 3), dtype=np.uint8)
            r[:, 0:w], r[:, w+10:] = r_pred, r_gt
            Image.fromarray(r).save(f"fig/render{'%03d' % i}.png")

