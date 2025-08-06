#coding-utf-8
import torch
import numpy as np
import math

def fill_boxes_to_fixed_num(gt_boxes, fixed_num=2):
    """
    填补 gt_boxes 为指定数量的 box，形状保持 [B, fixed_num, 10]

    Args:
        gt_boxes: [B, N, 10]  输入的 box
        fixed_num: int        要填补到的目标数量，默认为 2

    Returns:
        padded_boxes: [B, fixed_num, 10]
    """
    B, N, D = gt_boxes.shape
    if N >= fixed_num:
        return gt_boxes[:, :fixed_num, :]  # 裁剪多余的 box

    # 填补部分
    pad = torch.zeros((B, fixed_num - N, D), dtype=gt_boxes.dtype, device=gt_boxes.device)
    padded_boxes = torch.cat([gt_boxes, pad], dim=1).to(gt_boxes.device)
    return padded_boxes

def print_size(net):
    """
    Print the number of parameters of a network
    """

    if net is not None and isinstance(net, torch.nn.Module):
        module_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params = sum([np.prod(p.size()) for p in module_parameters])
        print(f"{net.__class__.__name__} Parameters: {params}")
        # print("{} Parameters: {:.6f}M".format(
        #     net.__class__.__name__, params / 1e6), flush=True)

# ---- ScaleLong ----
def universal_scalling(s_feat,s_factor=2**(-0.5)):
    return s_feat * s_factor

def exponentially_scalling(s_feat,k=0.8,i=1):
    return s_feat * k**(i - 1)
# ---- ScaleLong ----


def get_diffusion_betas(type='linear', start=0.0001, stop=0.02, T=1000):
    """Get betas from the hyperparameters."""
    if type == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        scale = 1000 / T
        beta_start = scale * start
        beta_end = scale * stop
        return torch.linspace(beta_start, beta_end, T, dtype=torch.float64)

    elif type == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = T + 1
        s = 0.008
        # t = torch.linspace(0, T, steps, dtype=torch.float64) / T
        t = torch.linspace(start, stop, steps, dtype=torch.float64) / T
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)


    elif type == 'sigmoid':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        start = -3
        end = 3
        tau = 1
        steps = T + 1
        t = torch.linspace(0, T, steps, dtype=torch.float64) / T
        v_start = torch.tensor(start / tau).sigmoid()
        v_end = torch.tensor(end / tau).sigmoid()
        alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    elif type == "laplace":
        mu = 0.0
        b = 0.5
        lmb = lambda t: mu - b * torch.sign(0.5 - t) * torch.log(1 - 2 * torch.abs(0.5 - t))

        snr_func = lambda t: torch.exp(lmb(t))
        alpha_func = lambda t: torch.sqrt(snr_func(t) / (1 + snr_func(t)))
        # sigma_func = lambda t: torch.sqrt(1 / (1 + snr_func(t)))

        timesteps = torch.linspace(0, 1, 1002)[1:-1]
        alphas_cumprod = []
        for t in timesteps:
            a = alpha_func(t) ** 2
            alphas_cumprod.append(a)
        alphas_cumprod = torch.cat(alphas_cumprod, dim=0)
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    else:
        raise NotImplementedError(type)

def get_diffusion_hyperparams(
        noise_schedule,
        beta_start,
        beta_end,
        T
):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value,
                                where any beta_t in the middle is linearly interpolated

    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
        These cpu tensors are changed to cuda tensors on each individual gpu
    """

    # Beta = torch.linspace(noise_schedule,beta_start, beta_end, T)
    Beta = get_diffusion_betas(
        type=noise_schedule,
        start=beta_start,
        stop=beta_end,
        T=T
    )
    # at = 1 - bt
    Alpha = 1 - Beta
    # at_
    Alpha_bar = Alpha + 0
    # 方差
    Beta_tilde = Beta + 0
    for t in range(1, T):
        # \bar{\alpha}_t = \prod_{s=1}^t \alpha_s
        Alpha_bar[t] *= Alpha_bar[t - 1]
        # \tilde{\beta}_t = (1-\bar{\alpha}_{t-1}) / (1-\bar{\alpha}_t) * \beta_t
        Beta_tilde[t] *= (1 - Alpha_bar[t - 1]) / (1 - Alpha_bar[t])
    # 标准差
    Sigma = torch.sqrt(Beta_tilde)  # \sigma_t^2  = \tilde{\beta}_t
    Sigma[0] = 0.0

    '''
        SNR = at ** 2 / sigma ** 2
        at = sqrt(at_), sigma = sqrt(1 - at_)
        q(xt|x0) = sqrt(at_) * x0 + sqrt(1 - at_) * noise
    '''
    SNR = Alpha_bar / (1 - Alpha_bar)

    return Beta, Alpha, Alpha_bar, Sigma, SNR

def continuous_p_ddim_sample(x_t, t, noise,dm_target, Alpha_bar):

    if(dm_target == "noise"):
        # x0 = (xt - sqrt(1-at_) * noise) / sqrt(at_)
        x0 = (x_t - torch.sqrt(1 - Alpha_bar[t]) * noise) / torch.sqrt(Alpha_bar[t])
    else:
        x0 = noise
        # noise = (xt - sqrt(1-at_) * x0) / sqrt(1-at_)
        noise = (x_t - torch.sqrt(Alpha_bar[t]) * x0) / torch.sqrt(1 - Alpha_bar[t])

    if(t[0] == 0):
        return x0

    # sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_)
    xs_1 = torch.sqrt(Alpha_bar[t-1]) * x0

    # sqrt(1 - at-1_) * noise
    xs_2 = torch.sqrt(1 - Alpha_bar[t-1]) * noise

    # xt-1 = sqrt(at-1_) * (xt - sqrt(1-at_) * noise) / sqrt(at_) + sqrt(1 - at-1_) * noise
    xs = xs_1 + xs_2

    return xs

def continuous_q_sample(x_0, t, Alpha_bar, noise=None):
    if(noise is None):
        # sampling from Gaussian distribution
        noise = torch.normal(0, 1, size=x_0.shape, dtype=torch.float32).cuda()
    # xt = sqrt(at_) * x0 + sqrt(1-at_) * noise
    x_t = torch.sqrt(Alpha_bar[t]) * x_0 + torch.sqrt(1 - Alpha_bar[t]) * noise
    return x_t

def get_time_schedule(self, T=1000, step=5):
    times = np.linspace(-1, T - 1, num = step + 1, dtype=int)[::-1]
    return times

"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""
def calc_t_emb(ts, t_emb_dim):
    """
    Embed time steps into a higher dimension space
    """
    assert t_emb_dim % 2 == 0

    # input is of shape (B) of integer time steps
    # output is of shape (B, t_emb_dim)
    if(ts.shape == 1):
        ts = ts.unsqueeze(1)
    half_dim = t_emb_dim // 2
    t_emb = np.log(10000) / (half_dim - 1)
    t_emb = torch.exp(torch.arange(half_dim) * -t_emb)
    t_emb = t_emb.to(ts.device)  # shape (half_dim)
    # ts is of shape (B,1)
    t_emb = ts * t_emb
    t_emb = torch.cat((torch.sin(t_emb), torch.cos(t_emb)), 1)

    return t_emb

def get_data_perturbations(start=1.0, stop=1.5, T=1000):
    return torch.linspace(start=start, end=stop, steps=T)

def givens_rotation(
    features,
    angle,
    dim_pairs=[(0, 1), (2, 3), (4, 5), (6, 7)],
):
    """
    对 features [N, D] 应用 Givens 旋转，每个样本使用自己的角度。
    angle: [N, 1]，每个样本一个旋转角度
    dim_pairs: 需要进行 Givens 旋转的维度对
    """
    assert features.ndim == 2
    assert angle.ndim == 2 and angle.shape[1] == 1

    N, D = features.shape
    assert all(0 <= i < D and 0 <= j < D for i, j in dim_pairs)

    angle = angle.squeeze(-1)  # [N]
    rotated = features

    for i, j in dim_pairs:
        c = torch.cos(angle).unsqueeze(1)  # [N, 1]
        s = torch.sin(angle).unsqueeze(1)  # [N, 1]

        xi = rotated[:, i:i+1]  # [N, 1]
        xj = rotated[:, j:j+1]  # [N, 1]

        rotated[:, i:i+1] = c * xi - s * xj
        rotated[:, j:j+1] = s * xi + c * xj

    return rotated

def add_gaussian_noise(pts, sigma=0.1, clamp=0.03):

    assert (clamp > 0)
    # jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp)
    jittered_data = sigma * torch.randn_like(pts).cuda() # e~N(0,I) ==> e * sigma ~ N(0, sigmaI)
    jittered_data = jittered_data + pts
    print(f"----gaussian Noise Level : {sigma}----")
    return jittered_data

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(x, angle):
    """
    Args:
        x: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    x_dim = x.dim()
    angle = angle.squeeze()
    if x_dim == 2:
        x = x.unsqueeze(0)  # -> [1, N, 7]

    x, is_numpy = check_numpy_to_torch(x)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(x.shape[0])
    ones = angle.new_ones(x.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    x_rot = torch.matmul(x[:, :, 0:3], rot_matrix)
    x_rot = torch.cat((x_rot, x[:, :, 3:]), dim=-1)
    if(x_dim == 2):
        x_rot = x_rot.squeeze()

    return x_rot.numpy() if is_numpy else x_rot, angle