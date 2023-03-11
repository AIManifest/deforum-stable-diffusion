import torch
from PIL import Image
import requests
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
import os
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.ddim import DDIMSampler
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from torch import autocast
from contextlib import nullcontext
from einops import rearrange, repeat
from types import SimpleNamespace

from helpers.prompt import get_uc_and_c
from helpers.k_samplers import sampler_fn, make_inject_timing_fn
from scipy.ndimage import gaussian_filter

from helpers.callback import SamplerCallback
from helpers.animation import DeformAnimKeys

from helpers.conditioning import exposure_loss, make_mse_loss, get_color_palette, make_clip_loss_fn
from helpers.conditioning import make_rgb_color_match_loss, blue_loss_fn, threshold_by, make_aesthetics_loss_fn, mean_loss_fn, var_loss_fn, exposure_loss
from helpers.model_wrap import CFGDenoiserWithGrad
from helpers.load_images import load_img, load_mask_latent, prepare_mask, prepare_overlay_mask

def generate(embedding_name, embedding_dir, cond_prompts, uncond_prompt, training_width, training_height, arg_seed, root, return_latent=False, return_sample=False, return_c=False):
    W = training_width 
    H = training_height
    W, H = map(lambda x: x - x % 64, (W, H))
    bit_depth_output = 8
    render_video_every = 1000
    
    seed = arg_seed 
    sampler = 'euler_ancestral'
    steps = 20
    scale = 7
    ddim_eta = 0.0
    dynamic_threshold = None
    static_threshold = None   

    
    save_samples = True
    save_settings = True
    display_samples = True
    save_sample_per_step = False
    show_sample_per_step = False

    n_batch = 1
    n_samples = 1
    batch_name = embedding_name
    filename_format = "{timestring}_{index}_{prompt}.png"
    seed_behavior = "iter"
    seed_iter_N = 1
    make_grid = False 
    grid_rows = 2
    outdir = embedding_dir

    
    use_init = False
    strength = 0.65
    strength_0_no_init = True
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
    add_init_noise = True
    init_noise = 0.015
    
    use_mask = False 
    use_alpha_as_mask = False 
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" 
    invert_mask = False 
    
    mask_brightness_adjust = 1.0
    mask_contrast_adjust = 1.0
    
    overlay_mask = True 
    
    mask_overlay_blur = 5 

    
    mean_scale = 0 
    var_scale = 0 
    exposure_scale = 0
    exposure_target = 0.5 

    
    colormatch_scale = 0 
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"
    colormatch_n_colors = 4
    ignore_sat_weight = 0

    
    clip_name = 'ViT-L/14'
    clip_scale = 0
    aesthetics_scale = 0
    cutn = 1
    cut_pow = 0.0001

    
    init_mse_scale = 0
    init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
    blue_scale = 0
    
    gradient_wrt = 'x0_pred'
    gradient_add_to = 'both'
    decode_method = 'linear'
    grad_threshold_type = 'dynamic'
    clamp_grad_threshold = 0.2
    clamp_start = 0.2
    clamp_stop = 0.01
    grad_inject_timing = list(range(1,10))

    cond_uncond_sync = False
    precision = 'autocast' 
    C = 4
    f = 8

    timestring = ""
    init_latent = None
    init_sample = None
    init_sample_raw = None
    mask_sample = None
    init_c = None
    seed_internal = 0
    def Args():
        W = training_width 
        H = training_height
        W, H = map(lambda x: x - x % 64, (W, H))
        bit_depth_output = 8
        render_video_every = 1000
        
        seed = arg_seed 
        sampler = 'euler_ancestral'
        steps = 20
        scale = 7
        ddim_eta = 0.0
        dynamic_threshold = None
        static_threshold = None   

        
        save_samples = True
        save_settings = True
        display_samples = True
        save_sample_per_step = False
        show_sample_per_step = False

        n_batch = 1
        n_samples = 1
        batch_name = embedding_name
        filename_format = "{timestring}_{index}_{prompt}.png"
        seed_behavior = "iter"
        seed_iter_N = 1
        make_grid = False 
        grid_rows = 2
        outdir = embedding_dir

        
        use_init = False
        strength = 0.65
        strength_0_no_init = True
        init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
        add_init_noise = True
        init_noise = 0.015
        
        use_mask = False 
        use_alpha_as_mask = False 
        mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg" 
        invert_mask = False 
        
        mask_brightness_adjust = 1.0
        mask_contrast_adjust = 1.0
        
        overlay_mask = True 
        
        mask_overlay_blur = 5 

        
        mean_scale = 0 
        var_scale = 0 
        exposure_scale = 0
        exposure_target = 0.5 

        
        colormatch_scale = 0 
        colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"
        colormatch_n_colors = 4
        ignore_sat_weight = 0

        
        clip_name = 'ViT-L/14'
        clip_scale = 0
        aesthetics_scale = 0
        cutn = 1
        cut_pow = 0.0001

        
        init_mse_scale = 0
        init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"
        blue_scale = 0
        
        gradient_wrt = 'x0_pred'
        gradient_add_to = 'both'
        decode_method = 'linear'
        grad_threshold_type = 'dynamic'
        clamp_grad_threshold = 0.2
        clamp_start = 0.2
        clamp_stop = 0.01
        grad_inject_timing = list(range(1,10))

        cond_uncond_sync = False
        precision = 'autocast' 
        C = 4
        f = 8

        timestring = ""
        init_latent = None
        init_sample = None
        init_sample_raw = None
        mask_sample = None
        init_c = None
        seed_internal = 0
        return locals()

    args = Args()
    args = SimpleNamespace(**args)
    seed_everything(seed)
    os.makedirs(outdir, exist_ok=True)
    
    sampler = PLMSSampler(root.model) if sampler == 'plms' else DDIMSampler(root.model)
    if root.model.parameterization == "v":
        model_wrap = CompVisVDenoiser(root.model)
    else:
        model_wrap = CompVisDenoiser(root.model)
    batch_size = n_samples
    # cond prompts
    cond_prompts = cond_prompts
    assert cond_prompts is not None
    cond_data = [batch_size * [cond_prompts]]
    # uncond prompts
    uncond_prompt = uncond_prompt
    assert uncond_prompt is not None
    uncond_data = [batch_size * [uncond_prompt]]
    
    precision_scope = autocast if precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if init_latent is not None:
        init_latent = init_latent
    elif init_sample is not None:
        with precision_scope("cuda"):
            init_latent = root.model.get_first_stage_encoding(root.model.encode_first_stage(init_sample))
    elif use_init and init_image != None and init_image != '':
        init_image, mask_image = load_img(init_image, 
                                          shape=(W, H),  
                                          use_alpha_as_mask=use_alpha_as_mask)
        if add_init_noise:
            init_image = add_noise(init_image,init_noise)
        init_image = init_image.to(root.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = root.model.get_first_stage_encoding(root.model.encode_first_stage(init_image))  # move to latent space        

    if not use_init and strength > 0 and strength_0_no_init:
        #print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        #print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        strength = 0

    # Mask functions
    if use_mask:
        assert mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(mask_file if mask_image is None else mask_image, 
                            init_latent.shape,
                            mask_contrast_adjust, 
                            mask_brightness_adjust,
                            invert_mask)
        
        if (torch.all(mask == 0) or torch.all(mask == 1)) and use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
        
        mask = mask.to(root.device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (use_mask and overlay_mask) and (init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

    # Init MSE loss image
    init_mse_image = None
    if init_mse_scale and init_mse_image != None and init_mse_image != '':
        init_mse_image, mask_image = load_img(init_mse_image,
                                          shape=(W, H),
                                          use_alpha_as_mask=use_alpha_as_mask)
        init_mse_image = init_mse_image.to(root.device)
        init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=batch_size)

    assert not ( init_mse_scale != 0 and (init_mse_image is None or init_mse_image == '') ), "Need an init image when init_mse_scale != 0"

    t_enc = int((1.0-strength) * steps)
    print(f"tenc: {t_enc}")

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(steps)
    clamp_schedule = dict(zip(k_sigmas.tolist(), np.linspace(clamp_start,clamp_stop,steps+1)))
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if sampler in ['plms','ddim']:
        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)

    if colormatch_scale != 0:
        assert colormatch_image is not None, "If using color match loss, colormatch_image is needed"
        colormatch_image, _ = load_img(colormatch_image)
        colormatch_image = colormatch_image.to('cpu')
        del(_)
    else:
        colormatch_image = None

    # Loss functions
    if init_mse_scale != 0:
        if decode_method == "linear":
            mse_loss_fn = make_mse_loss(root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(init_mse_image.to(root.device)))))
        else:
            mse_loss_fn = make_mse_loss(init_mse_image)
    else:
        mse_loss_fn = None

    if colormatch_scale != 0:
        _,_ = get_color_palette(root, colormatch_n_colors, colormatch_image, verbose=True) # display target color palette outside the latent space
        if decode_method == "linear":
            grad_img_shape = (int(W/f), int(H/f))
            colormatch_image = root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(colormatch_image.to(root.device))))
            colormatch_image = colormatch_image.to('cpu')
        else:
            grad_img_shape = (W, H)
        color_loss_fn = make_rgb_color_match_loss(root,
                                                  colormatch_image, 
                                                  n_colors=colormatch_n_colors, 
                                                  img_shape=grad_img_shape,
                                                  ignore_sat_weight=ignore_sat_weight)
    else:
        color_loss_fn = None

    if clip_scale != 0:
        clip_loss_fn = make_clip_loss_fn(root, args)
    else:
        clip_loss_fn = None

    if aesthetics_scale != 0:
        aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
    else:
        aesthetics_loss_fn = None

    if exposure_scale != 0:
        exposure_loss_fn = exposure_loss(exposure_target)
    else:
        exposure_loss_fn = None

    loss_fns_scales = [
        [clip_loss_fn,              clip_scale],
        [blue_loss_fn,              blue_scale],
        [mean_loss_fn,              mean_scale],
        [exposure_loss_fn,          exposure_scale],
        [var_loss_fn,               var_scale],
        [mse_loss_fn,               init_mse_scale],
        [color_loss_fn,             colormatch_scale],
        [aesthetics_loss_fn,        aesthetics_scale]
    ]

    # Conditioning gradients not implemented for ddim or PLMS
    assert not( any([cond_fs[1]!=0 for cond_fs in loss_fns_scales]) and (sampler in ["ddim","plms"]) ), "Conditioning gradients not implemented for ddim or plms. Please use a different sampler."

    callback = SamplerCallback(args=args,
                            root=root,
                            mask=mask, 
                            init_latent=init_latent,
                            sigmas=k_sigmas,
                            sampler=sampler,
                            verbose=False).callback 

    clamp_fn = threshold_by(threshold=clamp_grad_threshold, threshold_type=grad_threshold_type, clamp_schedule=clamp_schedule)

    grad_inject_timing_fn = make_inject_timing_fn(grad_inject_timing, model_wrap, steps)

    cfg_model = CFGDenoiserWithGrad(model_wrap, 
                                    loss_fns_scales, 
                                    clamp_fn, 
                                    gradient_wrt, 
                                    gradient_add_to, 
                                    cond_uncond_sync,
                                    decode_method=decode_method,
                                    grad_inject_timing_fn=grad_inject_timing_fn, # option to use grad in only a few of the steps
                                    grad_consolidate_fn=None, # function to add grad to image fn(img, grad, sigma)
                                    verbose=False)
    
    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with root.model.ema_scope():
                for cond_prompts, uncond_prompts in zip(cond_data,uncond_data):

                    # if isinstance(cond_prompts, tuple):
                    #     cond_prompts = list(cond_prompts)
                    # if isinstance(uncond_prompts, tuple):
                    #     uncond_prompts = list(uncond_prompts)

                    uc = root.model.get_learned_conditioning(uncond_prompts)
                    c = root.model.get_learned_conditioning(cond_prompts)

                    if scale == 1.0:
                        uc = None
                    else:
                        scale = scale
                    if init_c != None:
                        c = init_c
                    
                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]:
                        samples = sampler_fn(
                            c=c, 
                            uc=uc, 
                            args=args, 
                            model_wrap=cfg_model, 
                            init_latent=init_latent, 
                            t_enc=t_enc, 
                            device=root.device, 
                            cb=callback,
                            verbose=False)
                    elif args.sampler in ['plms','ddim']:
                        if init_latent is not None and strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(root.device))
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc)
                        else:
                            z_enc = torch.randn([n_samples, C, H // f, W // f], device=root.device)
                            shape = [C, H // f, W // f]
                            samples, _ = sampler.sample(S=steps,
                                                            conditioning=c,
                                                            batch_size=n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=scale,
                                                            unconditional_conditioning=uc,
                                                            eta=ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                    else:
                        raise Exception(f"Sampler {sampler} not recognised.")

                    
                    if return_latent:
                        results.append(samples.clone())

                    x_samples = root.model.decode_first_stage(samples)

                    if use_mask and overlay_mask:
                        # Overlay the masked image after the image is generated
                        if init_sample_raw is not None:
                            img_original = init_sample_raw
                        elif init_image is not None:
                            img_original = init_image
                        else:
                            raise Exception("Cannot overlay the masked image without an init image to overlay")

                        if mask_sample is None or using_vid_init:
                            mask_sample = prepare_overlay_mask(args, root, img_original.shape)

                        x_samples = img_original * mask_sample + x_samples * ((mask_sample * -1.0) + 1)

                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        def uint_number(datum, number):
                            if number == 8:
                                datum = Image.fromarray(datum.astype(np.uint8))
                            elif number == 32:
                                datum = datum.astype(np.float32)
                            else:
                                datum = datum.astype(np.uint16)
                            return datum
                        if bit_depth_output == 8:
                            exponent_for_rearrange = 1
                        elif bit_depth_output == 32:
                            exponent_for_rearrange = 0
                        else:
                            exponent_for_rearrange = 2
                        x_sample = 255.**exponent_for_rearrange * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = uint_number(x_sample, bit_depth_output)
                        results.append(image)
    return results
