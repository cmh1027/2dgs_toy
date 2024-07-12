import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch
import colorsys
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def normal2dir(n):
    line = torch.tensor([n[1], -n[0]]).to(n.device)
    return line

def get_bias(n, b):
    bias = torch.tensor([0., -b/n[1]]).to(n.device)
    return bias

def line2image(vec, name):
    N = len(vec)
    vec = vec.view(1, N, 3).repeat(N, 1, 1).permute(2, 0, 1)
    return np.array(torchvision.transforms.functional.to_pil_image(vec))

def eval_normal_1d(x, mu, var):
    """
    x : (M, 2)
    mu : (2,)
    var : (1,)
    """
    c = 2.5066282 # sqrt(2pi)
    # return torch.exp(-0.5 * ((x - mu) / sigma)**2) / (sigma * c)
    return torch.exp(-0.5 * (((x - mu[None, ...])**2).sum(dim=-1) / var.item()))[..., None]

def draw_model(model, normal, bias, plot_xlim, title=None):
    xy_ = model.get_xy
    cov_ = model.get_covariance
    rgb_ = model.get_rgb
    opacity_ = model.get_opacity
    xmin, xmax = -10, 15
    ymin, ymax = -10, 15
    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    if normal is not None:
        normal = normal.cpu().detach().numpy()
    for mu, cov, rgb, opacity in zip(xy_, cov_, rgb_, opacity_):
        mu = mu.cpu().detach().numpy()
        cov = cov.cpu().detach().numpy()
        rgb = rgb.cpu().detach().numpy()
        opacity = opacity.cpu().detach().numpy()
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # Ellipse parameters
        width, height = 2 * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))

        # Plotting the ellipse
        ellipse = Ellipse(xy=mu, width=width, height=height, angle=angle, edgecolor='None', fc=np.concatenate([rgb, opacity], axis=-1))
        ax.add_patch(ellipse)

        # Plot the contour plot
        ax.vlines(0, ymin=ymin, ymax=ymax, colors='black')
        ax.hlines(0, xmin=xmin, xmax=xmax, colors='black')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        if title is not None:
            ax.set_title(title)
        if normal is not None and bias is not None:
            X = np.linspace(plot_xlim[0], plot_xlim[1], 100)
            Y = (-normal[0] / normal[1]) * X - (bias / normal[1])
            plt.plot(X, Y, color='black')
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return image_from_plot

def generate_random_color(N):
    h = torch.clamp(torch.linspace(0.2, 0.8, N) + (torch.rand(N) - 0.5) * 0.1, 0, 1)[..., None]
    s = torch.rand(N, 1) * 0.1 + 0.9  # Saturation: 0.5 to 1.0 for vivid colors
    v = torch.rand(N, 1) * 0.1 + 0.9  # Brightness: 0.5 to 1.0 for brighter colors
    i = (h*6.0).to(torch.uint8)
    f = (h*6.0) - i
    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))
    i = i%6
    i = i.squeeze()
    rgb = torch.zeros(N, 3)
    rgb[i == 0] = torch.cat((v[i == 0], t[i == 0], p[i == 0]), dim=-1)
    rgb[i == 1] = torch.cat((q[i == 1], v[i == 1], p[i == 1]), dim=-1)
    rgb[i == 2] = torch.cat((p[i == 2], v[i == 2], t[i == 2]), dim=-1)
    rgb[i == 3] = torch.cat((p[i == 3], q[i == 3], v[i == 3]), dim=-1)
    rgb[i == 4] = torch.cat((t[i == 4], p[i == 4], v[i == 4]), dim=-1)
    rgb[i == 5] = torch.cat((v[i == 5], p[i == 5], q[i == 5]), dim=-1)
    return rgb

def inverse_sigmoid(x):
    return torch.log(x/(1-x))