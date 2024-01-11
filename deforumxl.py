import os
import cv2
import xformers
import xformers.ops
from comfy import model_management
import folder_paths
from enum import Enum
import torch
import cuda_malloc
from comfy.model_management import VRAMState
from IPython.display import clear_output
from pytorch_lightning import seed_everything
from PIL import Image as pil_image
from PIL import ImageOps
import numpy as np
from einops import rearrange
from comfy import model_management, sd
from torchvision.transforms.functional import to_pil_image
from comfy import latent_formats
from comfy.latent_formats import SDXL
from IPython.display import display, clear_output
import io
import nodes
import comfy
import importlib
import latent_preview
import gc
import random
from helpers.animation import sample_from_cv2
from comfy_extras import nodes_clip_sdxl
import time
from ipywidgets import Image, Layout, VBox
from io import BytesIO
from PIL import Image as pilimage
from PIL.PngImagePlugin import PngInfo
import json
from comfy_extras.nodes_canny import canny
from custom_nodes.comfy_controlnet_preprocessors.nodes.util import common_annotator_call, img_np_to_tensor, skip_v1
from custom_nodes.comfy_controlnet_preprocessors.v1 import midas, leres
from custom_nodes.comfy_controlnet_preprocessors.v11 import zoe, normalbae
import numpy as np
from iprogress import iprogress
from natsort import natsorted

def get_device_memory():
    total_memory = torch.cuda.get_device_properties(0).total_memory
    total_memory_gb = total_memory / (1024 ** 3)
    reserved_memory = torch.cuda.memory_reserved(0)
    reserved_memory_gb = reserved_memory / (1024 ** 3)
    allocated_memory = torch.cuda.memory_allocated(0)
    allocated_memory_gb = allocated_memory / (1024 ** 3)
    free_memory = total_memory - allocated_memory
    free_memory_gb = free_memory / (1024 ** 3)

    print(f"Total memory: {total_memory_gb:.2f} GB")
    print(f"Reserved memory: {reserved_memory_gb:.2f} GB")
    print(f"Allocated memory: {allocated_memory_gb:.2f} GB")
    print(f"Free memory: {free_memory_gb:.2f} GB")

get_device_memory()

def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
        if strength == 0:
            return (positive, negative)

        control_hint = image.movedim(-1,1)
        cnets = {}

        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()

                prev_cnet = d.get('control', None)
                if prev_cnet in cnets:
                    c_net = cnets[prev_cnet]
                else:
                    c_net = control_net.copy().set_cond_hint(control_hint, strength, (1.0 - start_percent, 1.0 - end_percent))
                    c_net.set_previous_controlnet(prev_cnet)
                    cnets[prev_cnet] = c_net

                d['control'] = c_net
                d['control_apply_to_uncond'] = False
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1])

def load_image(image_path):
        # image_path = folder_paths.get_annotated_filepath(image)
        i = pil_image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

def load_lora(model, clip, lora_name, strength_model, strength_clip):
    loaded_lora = None
    if strength_model == 0 and strength_clip == 0:
        return (model, clip)

    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora = None
    if loaded_lora is not None:
            if loaded_lora[0] == lora_path:
                lora = loaded_lora[1]
            else:
                temp = loaded_lora
                loaded_lora = None
                del temp

    if lora is None:
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        loaded_lora = (lora_path, lora)

    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
    del model
    del clip
    return (model_lora, clip_lora)

def loadsdxl(root):
    start = time.time()
    loader = nodes.CheckpointLoaderSimple()
    out = loader.load_checkpoint(
            root.custom_checkpoint_path,
            output_vae=True,
            output_clip=True,
            )
    
    model, clip, vae, clipvision = out
    
    clear_output(wait=True)
    
    get_device_memory()
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    if root.lora_name is not None:
        if isinstance(root.lora_name, dict):
            # If it's a dictionary, iterate through the items and load each one
            for lora_item, strength_model_clip in root.lora_name.items():
                print(f'Multiple Loras detected, loading {lora_item}')
                # Get the strengths for the current lora_name or use defaults
                strength_model = strength_model_clip
                strength_clip = strength_model_clip    
                lora = load_lora(model, clip, lora_item, strength_model, strength_clip)
                old_model, old_clip, old_out = model, clip, out
                model, clip = lora
                del old_model
                del old_clip
                del old_out
            out = (model, clip, vae, clipvision)
        elif isinstance(root.lora_name, str):
            # If it's a string, load only that string
            lora = load_lora(model, clip, root.lora_name, root.strength_model, root.strength_clip)
            old_model, old_clip, old_out = model, clip, out
            model, clip = lora
            del old_model
            del old_clip
            del old_out
            out = (model, clip, vae, clipvision)

    end = time.time()
    print(f'model loaded in {end-start:.02f} seconds')
    return out

def create_video(image_folder, fps, video_name):
    ext = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
    images = [img for img in natsorted(os.listdir(image_folder)) if os.path.splitext(img)[1] in ext]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(os.path.join(image_folder, video_name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))
    for image in iprogress(images, desc="creating video", colour="sunset"):
        video.write(cv2.imread(os.path.join(image_folder, image)))
    cv2.destroyAllWindows()
    video.release()

def runsdxl(args, root, keys, frame=0, control_net=None):
    model, clip, vae, _ = root.model
    
    init_sample = None

    if args.stop_at_last_layer != None:
        clip = clip.clone()
        clip.clip_layer(args.stop_at_last_layer)
        
    tokens = clip.tokenize(args.cond_prompt)
    tokens["l"] = clip.tokenize(args.cond_prompt)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    pcond, ppooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    positive = [[pcond, {"pooled_output": ppooled, "width": args.clipwidth, "height": args.clipheight, "crop_w": args.crop_w, "crop_h": args.crop_h, "target_width": args.target_width, "target_height": args.target_height}]]
    tokens = clip.tokenize(args.uncond_prompt)
    tokens["l"] = clip.tokenize(args.uncond_prompt)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    ncond, npooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    negative = [[ncond, {"pooled_output": npooled, "width": args.clipwidth, "height": args.clipheight, "crop_w": args.crop_w, "crop_h": args.crop_h, "target_width": args.target_width, "target_height": args.target_height}]]

    if args.init_sample is None:
        latentempty = nodes.EmptyLatentImage()
        latent = latentempty.generate(args.imagewidth, args.imageheight, args.batch_size)
        latent = latent[0]
        if args.is_controlnet:
            image, image_mask = load_image(args.controlnet_image)
            if "canny" in args.controlnet_name:
                output = canny(image.movedim(-1, 1), args.controlnet_low_threshold, args.controlnet_high_threshold)
                img_out = output[1].repeat(1, 3, 1, 1).movedim(1, -1)
            elif "depth" in args.controlnet_name:
                np_detected_map = common_annotator_call(zoe.ZoeDetector(), image)
                img_out = img_np_to_tensor(np_detected_map)
            positive, negative = apply_controlnet(positive, negative, control_net[0], img_out, args.controlnet_strength, args.controlnet_start_percent, args.controlnet_end_percent)
    else:
        latent = args.init_sample

    if args.init_sample is not None:
        try:
            vae
        except:
            vae = comfy.sd.VAE(ckpt_path=args.vae_path)
        init_sample = args.init_sample
        init_sample = pilimage.fromarray(init_sample)
        i = ImageOps.exif_transpose(init_sample)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        latent = vae.encode(image)
        
    force_full_denoise = args.force_full_denoise
    disable_noise = args.disable_noise 
    
    device = comfy.model_management.get_torch_device()
    
    if args.init_sample is not None:
        latent_image = latent
    else:
        latent_image = latent["samples"]
    
    disable_noise = False if args.init_sample is None else args.disable_noise
    if disable_noise:
        noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device=args.noisedevice)
    else:
        try:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            print(f'BATCH INDS ACTIVATED: {batch_inds}')
        except:
            batch_inds = None
        noise = comfy.sample.prepare_noise(latent_image, args.seed, batch_inds)    
    noise_mask = None
        
    preview_format = "PNG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"
    
    class LatentFormat:
        def process_in(self, latent):
            return latent * self.scale_factor
    
        def process_out(self, latent):
            return latent / self.scale_factor
    latent_format = SDXL()
    use_preview = args.use_preview
    if use_preview:
        previewer = latent_preview.Latent2RGBPreviewer(latent_format.latent_rgb_factors)#get_previewer(device, model.model.latent_format)
    else:
        previewer = latent_preview.get_previewer(device, model.model.latent_format)
    pbar = comfy.utils.ProgressBar(args.steps)
    
    image_widget = Image()
    vbox = VBox([image_widget], layout=Layout(width="256px"))
    display(vbox)

    def callback(step, x0, x, total_steps):
        preview_bytes = None
        # idx = len(os.listdir(preview_save_path))
        if previewer:
            preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            if use_preview:
                new_bytes = preview_bytes[1]
                # preview_save = os.path.join(preview_save_path, f'preview_{idx+1:05d}.png')
                # new_bytes.save(preview_save)
                display_bytes = BytesIO()
                new_bytes.save(display_bytes, format='PNG')
                image_data = display_bytes.getvalue()
                image_widget.value = image_data
        pbar.update_absolute(step + 1, total_steps, preview_bytes)
    
    # start_step = 0 if args.init_latent_in is None else int(args.steps*args.denoise)
    denoise = 1.0 if init_sample is None else args.strength
    start_step = 0
    init_step = int((1.0-args.strength) * args.steps)
    start_step = start_step if init_sample is None else init_step
    
    print("args.steps:", args.steps)
    print("args.scale:", args.scale)
    print("args.sampler:", args.sampler)
    print("args.scheduler:", args.scheduler)
    print("denoise:", denoise)
    print("disable_noise:", disable_noise)
    print("start_step:", start_step)
    print("last_step:", args.steps)
    print("force_full_denoise:", True)
    print("seed:", args.seed)
    print("args.clip_skip", args.stop_at_last_layer)

    samples = comfy.sample.sample(args, 
                                  model, 
                                  noise, 
                                  args.steps, 
                                  args.scale, 
                                  args.sampler, 
                                  args.scheduler, 
                                  positive, 
                                  negative, 
                                  latent_image, 
                                  denoise=denoise, 
                                  disable_noise=disable_noise, 
                                  start_step=start_step, 
                                  last_step=args.steps, 
                                  force_full_denoise=True, 
                                  noise_mask=noise_mask, 
                                  callback=callback, 
                                  seed=args.seed)
    try:
        print('Copying latent')
        samplez = latent.copy()
    except:
        print('Cloning latent')
        samplez = latent.clone()
    # samplez["samples"] = samples
    # model_management.unload_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    get_device_memory()

    if args.use_refiner:
        loader = nodes.CheckpointLoaderSimple()
        refinerout = loader.load_checkpoint(
                root.custom_checkpoint_path,
                output_vae=True,
                output_clip=True,
                )
        clear_output(wait=True)
    
        refinermodel, refinerclip, refinervae, refinerclipvision = refinerout

        refinernoise = comfy.sample.prepare_noise(samples, args.seed, batch_inds)
    
        refinersamples = comfy.sample.sample(args, 
                                      refinermodel, 
                                      refinernoise, 
                                      args.refinersteps, 
                                      args.scale, 
                                      args.sampler, 
                                      args.scheduler, 
                                      positive, 
                                      negative, 
                                      samples, 
                                      denoise=denoise, 
                                      disable_noise=args.refinerdisable_noise, 
                                      start_step=args.refiner_start_step, 
                                      last_step=args.refiner_last_step, 
                                      force_full_denoise=args.refinerforce_full_denoise, 
                                      noise_mask=noise_mask, 
                                      callback=callback, 
                                      seed=args.seed)
        del refinermodel
        del refinerclip
        del refinervae
        del refinerclipvision
        del refinerout

        old_samples = samples
        samples = refinersamples
            
    samples=samples.cpu()

    if args.vae_path:
        try:
            vae
        except:
            print(f"Loading {args.vae_path}")
            vae = comfy.sd.VAE(ckpt_path=args.vae_path)
    
    vae_decode_method = args.vae_decode_method
    if vae_decode_method == "normal":
        image = vae.decode(samples)
    else:
        image = vae.decode_tiled(samples)
    vaeimage = rearrange(image, 'b h w c -> b c h w')
    vaeimage = vaeimage.squeeze(0)
    vaeimage = to_pil_image(vaeimage)

    return samples, vaeimage
