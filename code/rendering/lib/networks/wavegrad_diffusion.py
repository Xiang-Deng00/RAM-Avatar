"""
This code is based on:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
but I simplify the implementation by removing a lot of unused features, and implement the training scheme in WaveGrad.

Specifically, in this implementation, I only consider diffusion models that:
* predicts epsilon (the noise injected)
* uses noise level sqrt(alpha_bar) randomly sampled from a segment
* condions on continuous noise level sqrt(alpha_bar)
* uses L1 loss
"""

import enum
import math

import numpy as np
import torch
import torch.nn.functional as F


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        rescale_timesteps=False,
    ):
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        print("===== betas =====")
        print(betas)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        print("===== sqrt_alphas_cumprod =====")
        print(self.sqrt_alphas_cumprod)
        # print(self.sqrt_alphas_cumprod[:int(0.3*len(self.betas))])

        # calculations for hierarchical sampling of noise level (See Algo.1 in WaveGrad)
        self.sqrt_alphas_cumprod_seg_start = np.sqrt(0.5 * (self.alphas_cumprod + self.alphas_cumprod_prev))
        self.sqrt_alphas_cumprod_seg_end = np.sqrt(0.5 * (self.alphas_cumprod_next + self.alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        print("===== posterior_variance =====")
        print(self.posterior_variance)
        # print(self.posterior_variance[:int(0.3*len(self.betas))])

        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x_start, t, noise=None, noise_mask=None, noise_fn=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :param noise_mask: if specified, the noise will be multiplied with it
        :param noise_fn: if specified, it will be called to obtain noise
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise_fn is not None:
            noise = noise_fn(x_start)
        if noise_mask is not None:
            noise = noise * noise_mask

        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_sample_continuous(self, x_start, sqrt_alphas_cumprod, noise=None, noise_mask=None, noise_fn=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param sqrt_alphas_cumprod: the noise level, or sqrt(alpha_bar) in the WaveGrad paper
        :param noise: if specified, the split-out normal noise.
        :param noise_mask: if specified, the noise will be multiplied with it
        :param noise_fn: if specified, it will be called to obtain noise
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise_fn is not None:
            noise = noise_fn(x_start)
        if noise_mask is not None:
            noise = noise * noise_mask
        assert noise.shape == x_start.shape
        if isinstance(sqrt_alphas_cumprod, np.ndarray):
            sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - sqrt_alphas_cumprod**2)
        else:
            sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - sqrt_alphas_cumprod**2)
        return (
            _expand_dims_into_tensor(sqrt_alphas_cumprod, x_start.device, x_start.shape) * x_start
            + _expand_dims_into_tensor(sqrt_one_minus_alphas_cumprod, x_start.device, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        sqrt_alphas_cumprod = _select_into_tensor(self.sqrt_alphas_cumprod, t)
        model_output = model(x, sqrt_alphas_cumprod*1e3, **model_kwargs)

        model_variance = self.posterior_variance
        model_log_variance = self.posterior_log_variance_clipped

        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(
            self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
        )
        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        noise_mask=None,
        noise_fn=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        model_kwargs = {k: self.q_posterior_mean_variance(v, v, t-1)[0]
                        for k, v in model_kwargs.items()}

        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = torch.randn_like(x)

        if noise_fn is not None:
            noise = noise_fn(x)
        if noise_mask is not None:
            noise = noise * noise_mask
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        noise_mask=None,
        noise_fn=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            noise_mask=noise_mask,
            noise_fn=noise_fn,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        noise_mask=None,
        noise_fn=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = torch.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, ascii=True, desc='p-sampling')

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_mask=noise_mask,
                    noise_fn=noise_fn,
                )
                yield out
                img = out["sample"]

    def p_sample_loop_from_coarse(
        self,
        model,
        shape,
        img,
        start_step,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        noise_mask=None,
        noise_fn=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param img: the start image to denoise.
        :param start_step: the start step
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive_from_coarse(
            model,
            shape,
            img,
            start_step,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            noise_mask=noise_mask,
            noise_fn=noise_fn,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive_from_coarse(
        self,
        model,
        shape,
        img,
        start_step,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        noise_mask=None,
        noise_fn=None,
        device=None,
        progress=False,
    ):
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        indices = list(range(self.num_timesteps))[0:start_step][::-1]

        noise = torch.randn_like(img)
        if noise_fn is not None:
            noise = noise_fn(img)
        t = torch.LongTensor([start_step]).to(device)
        img = self.q_sample(img, t, noise=noise, noise_mask=noise_mask, noise_fn=noise_fn)

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, ascii=True, desc='p-sampling')

        for i in indices:
            t = torch.tensor([i] * shape[0], device=device)
            with torch.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    noise_mask=noise_mask,
                    noise_fn=noise_fn,
                )
                yield out
                img = out["sample"]

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None, noise_mask=None, noise_fn=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise_fn is not None:
            noise = noise_fn(x_start)
        if noise_mask is not None:
            noise = noise * noise_mask

        sqrt_alphas_cumprod_seg_start = _select_into_tensor(self.sqrt_alphas_cumprod_seg_start, t)
        sqrt_alphas_cumprod_seg_end = _select_into_tensor(self.sqrt_alphas_cumprod_seg_end, t)
        w = torch.rand_like(sqrt_alphas_cumprod_seg_start)
        sqrt_alphas_cumprod = w * sqrt_alphas_cumprod_seg_start + (1 - w) * sqrt_alphas_cumprod_seg_end

        x_t = self.q_sample_continuous(x_start, sqrt_alphas_cumprod, noise=noise)

        model_kwargs = {k: self.q_sample_continuous(v, sqrt_alphas_cumprod, noise=torch.zeros_like(v))
                        for k, v in model_kwargs.items()}
        model_output = model(x_t, sqrt_alphas_cumprod*1e3, **model_kwargs)

        terms = dict()
        terms["mse"] = mean_flat(torch.abs(noise - model_output))
        terms["loss"] = terms["mse"]

        return terms


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    elif isinstance(arr, torch.Tensor):
        res = arr.to(device=timesteps.device)[timesteps].float()
    else:
        raise RuntimeError('Unsupported array type: ' + str(type(arr)))

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _select_into_tensor(arr, timesteps):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :return: a tensor of shape [batch_size, ] where the shape has K dims.
    """
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    elif isinstance(arr, torch.Tensor):
        res = arr.to(device=timesteps.device)[timesteps].float()
    else:
        raise RuntimeError('Unsupported array type: ' + str(type(arr)))
    return res


def _expand_dims_into_tensor(arr, device, broadcast_shape):
    if isinstance(arr, np.ndarray):
        res = torch.from_numpy(arr.astype(np.float32)).to(device)
    elif isinstance(arr, torch.Tensor):
        res = arr.to(device)
    else:
        raise RuntimeError('Unsupported array type: ' + str(type(arr)))

    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
