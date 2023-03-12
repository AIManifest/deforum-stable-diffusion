import os
import time
import itertools
import cv2
import einops
import numpy as np
import torch
import random
from pytorch_lightning import seed_everything
from annotator.hed import HEDdetector
from annotator.canny import CannyDetector
from annotator.midas import MidasDetector
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from ldm.util import instantiate_from_config
from IPython import display
from PIL import Image

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

# apply_hed = HEDdetector()
# apply_canny = CannyDetector()
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
    # torch.set_default_dtype(torch.float16)
    # model = instantiate_from_config("/content/ControlNet/models/cldm_v15.yaml")
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
            print("Applying HED Detection")
            input_image = HWC3(input_image)
            detected_map = control.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        elif control.controlnet_model_type == "apply_canny":
            print("Applying Canny Detection")
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            detected_map = control.controlnet_apply_type(img, low_threshold, high_threshold)
            detected_map = HWC3(detected_map)
        
        else:
            print("Applying Midas Detection")
            input_image = HWC3(input_image)
            detected_map, _ = control.controlnet_apply_type(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape
            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

        detected_map = detected_map
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
    return [detected_map] + results if control.controlnet_model_type == "apply_hed" or "apply_depth" else [255 - detected_map] + results

def generate_control(root, control):
    if control.generate_frames:
        video_path = control.video_path
        frames_path = control.IN_dir
        helpers.animation.vid2frames(video_path, frames_path, n=1, overwrite=True)
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
        display.clear_output(wait=True)
        # display.clear_output(wait=True)
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
 
