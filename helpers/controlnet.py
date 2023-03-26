import os
import time
import subprocess
import itertools
import cv2
import einops
import numpy as np
import torch
import random
from torch import autocast
from pytorch_lightning import seed_everything

from PIL import Image
import requests
import torchvision.transforms.functional as TF
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser
from k_diffusion import sampling
from contextlib import nullcontext
from einops import rearrange, repeat

from .prompt import get_uc_and_c
from .k_samplers import sampler_fn, make_inject_timing_fn
from scipy.ndimage import gaussian_filter

from .callback import SamplerCallback
from .animation import DeformAnimKeys

from .conditioning import exposure_loss, make_mse_loss, get_color_palette, make_clip_loss_fn
from .conditioning import make_rgb_color_match_loss, blue_loss_fn, threshold_by, make_aesthetics_loss_fn, mean_loss_fn, var_loss_fn, exposure_loss
from .model_wrap import CFGDenoiserWithGrad
from .load_images import load_img, load_mask_latent, prepare_mask, prepare_overlay_mask

from annotator.hed import HEDdetector
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.openpose import OpenposeDetector
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config

from IPython import display

def vid2frames(video_path, frames_path, n=1, overwrite=True):      
    if not os.path.exists(frames_path) or overwrite: 
        try:
            for f in pathlib.Path(frames_path).glob('*.jpg'):
                f.unlink()
        except:
            pass
        assert os.path.exists(video_path), f"Video input {video_path} does not exist"
          
        vidcap = cv2.VideoCapture(video_path)
        success,image = vidcap.read()
        count = 0
        t=1
        success = True
        while success:
            if count % n == 0:
                cv2.imwrite(frames_path + os.path.sep + f"{t:05}.jpg" , image)     # save frame as JPEG file
                t += 1
            success,image = vidcap.read()
            count += 1
        print("Converted %d frames" % count)
    else: print("Frames already unpacked")

def create_first_video(frame_folder, output_filename, frame_rate=30, quality=17):
    os.chdir(frame_folder)
    pattern = '*.png'
    pix_fmt = 'yuv420p'
    process = subprocess.Popen(['ffmpeg',
                                '-framerate', 
                                f"{frame_rate}", 
                                '-pattern_type', 'glob', 
                                '-i', pattern, 
                                '-crf', str(quality), 
                                '-pix_fmt', pix_fmt, 
                                '-preset', 'veryfast', 
                                '-y', output_filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, error = process.communicate()

    display.display(output)
    display.display(error)
    os.chdir("..")

def load_controlnet(controlnet_config_path, controlnet_model_path):
    print_flag = False
    verbose = False
    model = create_model(controlnet_config_path).cpu()
    # model.load_state_dict(load_state_dict(controlnet_model_path, location='cuda'))
    pl_sd = torch.load(controlnet_model_path, map_location="cuda")
    try:
        sd = pl_sd["state_dict"]
    except:
        sd = pl_sd
    
    torch.set_default_dtype(torch.float32)
    m, u = model.load_state_dict(sd, strict=False)
    if print_flag:
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    save_memory = False
    return model

def process(root, control, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    model = root.model
    ddim_sampler = control.ddim_sampler
    save_memory = control.save_memory
    with torch.no_grad():
        if control.controlnet_model_type == "apply_hed":
            print("\033[32mApplying HED Detection\033[0m")
            input_image = HWC3(input_image)
            detected_map = control.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        elif control.controlnet_model_type == "open_pose":
            print("\033[33mApplying OpenPose Detection\033[0m")
            input_image = HWC3(input_image)
            detected_map, _ = control.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        elif control.controlnet_model_type == "apply_canny":
            print("\033[34mApplying Canny Detection\033[0m")
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = control.controlnet_apply_type(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
        
        else:
            print("\033[35mApplying Midas Detection\033[0m")
            input_image = HWC3(input_image)
            detected_map, _ = control.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        print("\033[31m..Input Image..\033[0m..")
        new_input_image = Image.fromarray(input_image)
        display.display(new_input_image)

        print("\033[31m..Detecting Map From Image..\033[0m..")
        map_detected = Image.fromarray(detected_map)
        display.display(map_detected)

        controlx = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        controlx = torch.stack([controlx for _ in range(num_samples)], dim=0)
        controlx = einops.rearrange(controlx, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [controlx], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [controlx], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [x_samples[i] for i in range(num_samples)]
        img_results = Image.fromarray(x_samples[0])
        display.display(img_results)
        
    return [detected_map] + results if control.controlnet_model_type == "apply_hed" or "apply_depth" or "open_pose" else [255 - detected_map] + results

def generate_control(root, control):
    if control.generate_frames:
        video_path = control.video_path
        frames_path = control.IN_dir
        vid2frames(video_path, frames_path, n=1, overwrite=True)
    else:
        print(f"Frames Path Exists Already at {control.IN_dir}, Skipping frame extraction.")
    input_image = control.IN_dir
    prompt = control.prompt
    a_prompt = control.a_prompt
    n_prompt = control.n_prompt
    num_samples = control.num_samples
    image_resolution = control.image_resolution
    detect_resolution = control.detect_resolution
    ddim_steps = control.ddim_steps
    guess_mode = control.guess_mode
    strength = control.strength
    scale = control.scale
    seed = control.seed
    eta = control.eta
    low_threshold = control.canny_low_threshold
    high_threshold = control.canny_high_threshold
    start_idx = 0
    max_frames = len(os.listdir(control.IN_dir))
    if not os.path.exists(control.OUT_dir):
        os.makedirs(control.OUT_dir)
    if control.resume_control:            
        start_idx = start_idx - 1
        IN_idx = control.img_idx
        last_idx = start_idx-1
    img_idx = 1
    if control.resume_control:
        img_idx = control.img_idx
    while img_idx < max_frames:
        input_img = os.path.join(control.IN_dir, f"{img_idx:05}.jpg")
        IN_img = input_img
        IN_img = Image.open(IN_img)
        print(f"\033[35mInput Image\033[0m: {input_img}")
        IN_array = np.asarray(IN_img, dtype=np.uint8)
        input_image = IN_array 
        output = ''
        colors = ['\033[31m', '\033[33m', '\033[34m', '\033[35m', '\033[32m']

        for i, letter in enumerate(control.prompt):
            color = colors[i % len(colors)]  # use modulo to cycle through the colors
            output += f'{color}{letter}'

        output += '\033[0m'
        # print the output
        print(output)
        a_output = ''

        for i, letter in enumerate(control.a_prompt):
            color = colors[i % len(colors)]  # use modulo to cycle through the colors
            a_output += f'{color}{letter}'

        a_output += '\033[0m'
        # print the output
        print(a_output)
        n_output = ''

        for i, letter in enumerate(control.n_prompt):
            color = colors[i % len(colors)]  # use modulo to cycle through the colors
            n_output += f'{color}{letter}'

        n_output += '\033[0m'
        # print the output
        print(n_output)
        # \033[31m', '\033[33m', '\033[34m', '\033[35m', '\033[32m
        print(f"\033[31mNumber of Images Rendering\033[0m: {control.num_samples}")
        print(f"\033[33mResolution of the Image\033[0m: {control.image_resolution} - \033[33mResolution of Detection\033[0m: {control.detect_resolution}")
        print(
            f"\033[31m'DDIM Steps\033[0m: {control.ddim_steps}",
            f"\033[33mScale\033[0m: {control.scale} - \033[32mStrength\033[0m: {control.strength} - \033[35mSeed\033[0m: {control.seed}"
        )

        image = process(root, control, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold)
        img = Image.fromarray(image[1])
        img.save(f"{control.OUT_dir}/{control.IN_batch}_{img_idx:05d}.png")
        print(f"Image Saved as: {control.OUT_dir}/{control.IN_batch}_{img_idx:05d}.png")
        img_idx+=1 
        # show the image
        # img.show()
        pil_image0 = IN_img
        pil_image1 = Image.fromarray(image[0])
        pil_image2 = Image.fromarray(image[1])

        pil_image2.show()

        height = max(pil_image1.height, pil_image2.height, pil_image0.height)
        width = pil_image1.width + pil_image2.width + pil_image0.width
        pil_image1 = pil_image1.resize((int(height * pil_image1.width / pil_image1.height), height))
        pil_image2 = pil_image2.resize((int(height * pil_image2.width / pil_image2.height), height))
        pil_image0 = pil_image0.resize((int(height * pil_image0.width / pil_image0.height), height))

        # Create a new image with the same height and triple the width
        new_image = Image.new('RGB', (width, height))

        # Paste the three images side by side
        new_image.paste(pil_image1, (0, 0))
        new_image.paste(pil_image2, (pil_image1.width, 0))
        new_image.paste(pil_image0, (pil_image1.width + pil_image2.width, 0))
        new_image_dir = os.path.join(control.OUT_dir, "merged_images")
        
        if not os.path.exists(new_image_dir):
          os.makedirs(new_image_dir)

        # # Show the resulting image
        new_image.show()
        new_image.save(os.path.join(new_image_dir, f"new_image_{img_idx:05d}.png"))
        # display.clear_output(wait=True)
        display.clear_output(wait=True)
        if img_idx % control.render_video_every == 0:
            print("..\033[33mRendering Video\033[0m..")
            time_start = time.time()
            frame_folder = control.OUT_dir
            output_filename = f"{control.OUT_dir}/{control.IN_batch}.mp4"
            create_first_video(frame_folder, output_filename, frame_rate=30, quality=17)
            time_end = time.time()
            time_elapsed = time_end - time_start
            print(f"Progress Animation Video Compiled, Saved to: {control.OUT_dir}, Filename: {output_filename}")
            print(f"Video Rendered in: {time_elapsed} seconds..")
    if img_idx == max_frames:
        frame_folder = control.OUT_dir
        output_filename_final = f"{control.OUT_dir}/{control.IN_batch}_final.mp4"
        create_first_video(frame_folder, output_filename_final, frame_rate=30, quality=17)
        print(f"Animation Video Compled, Saved to: {control.OUT_dir}, Filename: {output_filename_final}")

def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

#deforum render inference
def render_control_process(root, args, anim_args, keys, frame, input_image, prompt, a_prompt, n_prompt, num_samples, image_resolution, detect_resolution, ddim_steps, guess_mode, strength, scale, seed, eta, low_threshold, high_threshold):
    model = root.model
    ddim_sampler = DDIMSampler(model)
    sampler = "ddim"
    save_memory = args.save_memory
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    
    # sampler = DDIMSampler(root.model)
    if root.model.parameterization == "v":
        model_wrap = CompVisVDenoiser(root.model)
    else:
        model_wrap = CompVisDenoiser(root.model)
    batch_size = args.n_samples

    # cond prompts
    cond_prompt = args.cond_prompt
    assert cond_prompt is not None
    cond_data = [batch_size * [cond_prompt]]
    frame+=1
    # uncond prompts
    uncond_prompt = args.uncond_prompt
    assert uncond_prompt is not None
    uncond_data = [batch_size * [uncond_prompt]]
    
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = root.model.get_first_stage_encoding(root.model.encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image, 
                                          shape=(args.W, args.H),  
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        if args.add_init_noise:
            init_image = add_noise(init_image,args.init_noise)
        init_image = init_image.to(root.device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = root.model.get_first_stage_encoding(root.model.encode_first_stage(init_image))  # move to latent space        

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        #print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        #print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"


        mask = prepare_mask(args.mask_file if mask_image is None else mask_image, 
                            init_latent.shape,
                            args.mask_contrast_adjust, 
                            args.mask_brightness_adjust,
                            args.invert_mask)
        
        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")
        
        mask = mask.to(root.device)
        mask = repeat(mask, '1 ... -> b ...', b=batch_size)
    else:
        mask = None

    assert not ( (args.use_mask and args.overlay_mask) and (args.init_sample is None and init_image is None)), "Need an init image when use_mask == True and overlay_mask == True"

    # Init MSE loss image
    init_mse_image = None
    if args.init_mse_scale and args.init_mse_image != None and args.init_mse_image != '':
        init_mse_image, mask_image = load_img(args.init_mse_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_mse_image = init_mse_image.to(root.device)
        init_mse_image = repeat(init_mse_image, '1 ... -> b ...', b=batch_size)

    assert not ( args.init_mse_scale != 0 and (args.init_mse_image is None or args.init_mse_image == '') ), "Need an init image when init_mse_scale != 0"

    steps = int(keys.steps_schedule_series[frame]) if anim_args.animation_mode != "None" else args.steps

    t_enc = int((1.0-args.strength) * steps)
    print(f"\033[34mtenc\033[0m: {t_enc}")

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(steps)
    args.clamp_schedule = dict(zip(k_sigmas.tolist(), np.linspace(args.clamp_start,args.clamp_stop,steps+1)))
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if args.sampler in ['plms','ddim']:
        ddim_sampler.make_schedule(ddim_num_steps=steps, ddim_eta=args.ddim_eta, verbose=False)

    if args.colormatch_scale != 0:
        assert args.colormatch_image is not None, "If using color match loss, colormatch_image is needed"
        colormatch_image, _ = load_img(args.colormatch_image)
        colormatch_image = colormatch_image.to('cpu')
        del(_)
    else:
        colormatch_image = None

    # Loss functions
    if args.init_mse_scale != 0:
        if args.decode_method == "linear":
            mse_loss_fn = make_mse_loss(root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(init_mse_image.to(root.device)))))
        else:
            mse_loss_fn = make_mse_loss(init_mse_image)
    else:
        mse_loss_fn = None

    if args.colormatch_scale != 0:
        _,_ = get_color_palette(root, args.colormatch_n_colors, colormatch_image, verbose=True) # display target color palette outside the latent space
        if args.decode_method == "linear":
            grad_img_shape = (int(args.W/args.f), int(args.H/args.f))
            colormatch_image = root.model.linear_decode(root.model.get_first_stage_encoding(root.model.encode_first_stage(colormatch_image.to(root.device))))
            colormatch_image = colormatch_image.to('cpu')
        else:
            grad_img_shape = (args.W, args.H)
        color_loss_fn = make_rgb_color_match_loss(root,
                                                  colormatch_image, 
                                                  n_colors=args.colormatch_n_colors, 
                                                  img_shape=grad_img_shape,
                                                  ignore_sat_weight=args.ignore_sat_weight)
    else:
        color_loss_fn = None

    if args.clip_scale != 0:
        clip_loss_fn = make_clip_loss_fn(root, args)
    else:
        clip_loss_fn = None

    if args.aesthetics_scale != 0:
        aesthetics_loss_fn = make_aesthetics_loss_fn(root, args)
    else:
        aesthetics_loss_fn = None

    if args.exposure_scale != 0:
        exposure_loss_fn = exposure_loss(args.exposure_target)
    else:
        exposure_loss_fn = None

    loss_fns_scales = [
        [clip_loss_fn,              args.clip_scale],
        [blue_loss_fn,              args.blue_scale],
        [mean_loss_fn,              args.mean_scale],
        [exposure_loss_fn,          args.exposure_scale],
        [var_loss_fn,               args.var_scale],
        [mse_loss_fn,               args.init_mse_scale],
        [color_loss_fn,             args.colormatch_scale],
        [aesthetics_loss_fn,        args.aesthetics_scale]
    ]

    # Conditioning gradients not implemented for ddim or PLMS
    assert not( any([cond_fs[1]!=0 for cond_fs in loss_fns_scales]) and (args.sampler in ["ddim","plms"]) ), "Conditioning gradients not implemented for ddim or plms. Please use a different sampler."

    callback = SamplerCallback(args=args,
                            root=root,
                            mask=mask, 
                            init_latent=init_latent,
                            sigmas=k_sigmas,
                            sampler=sampler,
                            verbose=False).callback 

    clamp_fn = threshold_by(threshold=args.clamp_grad_threshold, threshold_type=args.grad_threshold_type, clamp_schedule=args.clamp_schedule)

    grad_inject_timing_fn = make_inject_timing_fn(root, args, args.grad_inject_timing, model_wrap, steps)

    cfg_model = CFGDenoiserWithGrad(model_wrap, 
                                    loss_fns_scales, 
                                    clamp_fn, 
                                    args.gradient_wrt, 
                                    args.gradient_add_to, 
                                    args.cond_uncond_sync,
                                    decode_method=args.decode_method,
                                    grad_inject_timing_fn=grad_inject_timing_fn, # option to use grad in only a few of the steps
                                    grad_consolidate_fn=None, # function to add grad to image fn(img, grad, sigma)
                                    verbose=False)
    keys = DeformAnimKeys(anim_args)
    with torch.no_grad():
        if root.controlnet_model_type == "apply_hed":
            print("\033[32mApplying HED Detection\033[0m")
            input_image = HWC3(input_image)
            detected_map = root.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        elif root.controlnet_model_type == "open_pose":
            print("\033[33mApplying OpenPose Detection\033[0m")
            input_image = HWC3(input_image)
            detected_map, _ = root.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        
        elif root.controlnet_model_type == "apply_canny":
            print("\033[34mApplying Canny Detection\033[0m")
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = root.controlnet_apply_type(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
        
        else:
            print("\033[35mApplying Midas Detection\033[0m")
            input_image = HWC3(input_image)
            detected_map, _ = root.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        print("\033[31m..Input Image..\033[0m..")
        new_input_image = Image.fromarray(input_image)
        display.display(new_input_image)

        print("\033[31m..Detecting Map From Image..\033[0m..")
        map_detected = Image.fromarray(detected_map)
        if args.display_detected_map:
            display.display(map_detected)

        controlx = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        controlx = torch.stack([controlx for _ in range(num_samples)], dim=0)
        controlx = einops.rearrange(controlx, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [controlx], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [controlx], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        # if init_latent is not None and args.strength > 0:
        #     z_enc = ddim_sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(root.device))
        #     samples = ddim_sampler.decode(z_enc, cond, t_enc, unconditional_guidance_scale=args.scale,
        #                           unconditional_conditioning=un_cond)
        # else:
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                    shape, cond, verbose=False, eta=eta,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=un_cond)

        if save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        if args.use_mask and args.overlay_mask:
            # Overlay the masked image after the image is generated
            if args.init_sample_raw is not None:
                img_original = args.init_sample_raw
            elif init_image is not None:
                img_original = init_image
            else:
                raise Exception("Cannot overlay the masked image without an init image to overlay")

            if args.mask_sample is None or args.using_vid_init:
                args.mask_sample = prepare_overlay_mask(args, root, img_original.shape)

            x_samples = img_original * args.mask_sample + x_samples * ((args.mask_sample * -1.0) + 1)
        results = [x_samples[i] for i in range(num_samples)]
        img_results = Image.fromarray(x_samples[0])
        display.display(img_results)
        
    return [detected_map] + results if root.controlnet_model_type == "apply_hed" or "apply_depth" or "open_pose" else [255 - detected_map] + results
