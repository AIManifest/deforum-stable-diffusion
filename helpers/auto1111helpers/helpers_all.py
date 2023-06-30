import os, sys, subprocess, gc, torch
import cv2
import random
import clip
import yaml
from IPython import display
from ipywidgets import widgets

import traceback
import inspect
from collections import namedtuple
import contextlib
import html
import datetime
import csv
import safetensors.torch
from helpers.auto1111helpers.dummy_generate import generate
import numpy as np
from PIL import Image, PngImagePlugin
from torch.utils.tensorboard import SummaryWriter
import datetime
import sys
import traceback
import pytz
import io
import math
import os
from collections import namedtuple
import re

import numpy as np
import piexif
import piexif.helper
from PIL import Image, ImageFont, ImageDraw, PngImagePlugin
try:
    from fonts.ttf import Roboto
except ImportError:
    subprocess.run(["pip", "install", "font-roboto"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
import string
import json
import hashlib
try:
    import google.colab
    models_path_gdrive = "/content/drive/MyDrive/sd/stable-diffusion-webui/models/Stable-diffusion"
    embeddings_dir = "/content/drive/MyDrive/sd/stable-diffusion-webui/embeddings/" #@param {type:"string"}
    hypernetwork_dir = "/content/drive/MyDrive/AI/hypernetworks" #@param {type:"string"}
    data_dir = "/content/drive/MyDrive/sd/stable-diffusion-webui/" #@param {type:"string"}
    use_xformers = True #@param{type:'boolean'}
    use_sub_quad_attention = False #@param{type:'boolean'}
    use_split_attention_v1 = False #@param{type:'boolean'}
    use_split_cross_attention_forward_invokeAI = False #@param{type:'boolean'}
    use_cross_attention_attnblock_forward = False #@param{type:'boolean'}
except:
    embeddings_dir = f"{os.getcwd()}/stable-diffusion-webui/embeddings/" #@param {type:"string"}
    hypernetwork_dir = f"{os.getcwd()}/stable-diffusion-webui/hypernetworks" #@param {type:"string"}
    data_dir = f"{os.getcwd()}/stable-diffusion-webui/" #@param {type:"string"}
    use_xformers = True #@param{type:'boolean'}
    use_sub_quad_attention = False #@param{type:'boolean'}
    use_split_attention_v1 = False #@param{type:'boolean'}
    use_split_cross_attention_forward_invokeAI = False #@param{type:'boolean'}
    use_cross_attention_attnblock_forward = False #@param{type:'boolean'}

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float32
dtype_unet = dtype
device_cuda = device
t_autocast = torch.autocast("cuda")
opts_upcast_attn = False
invalid_filename_chars = '<>:"/\\|?*\n'
invalid_filename_prefix = ' '
invalid_filename_postfix = ' .'
re_nonletters = re.compile(r'[\s' + string.punctuation + ']+')
re_pattern = re.compile(r"(.*?)(?:\[([^\[\]]+)\]|$)")
re_pattern_arg = re.compile(r"(.*)<([^>]*)>$")
max_filename_part_length = 128

#images
def sanitize_filename_part(text, replace_spaces=True):
    if text is None:
        return None

    if replace_spaces:
        text = text.replace(' ', '_')

    text = text.translate({ord(x): '_' for x in invalid_filename_chars})
    text = text.lstrip(invalid_filename_prefix)[:max_filename_part_length]
    text = text.rstrip(invalid_filename_postfix)
    return text

class FilenameGenerator:
    replacements = {
        'seed': lambda self: self.seed if self.seed is not None else '',
        'steps': lambda self: self.steps,
        'cfg': lambda self: self.scale,
        'width': lambda self: self.image.width,
        'height': lambda self: self.image.height,
        'styles': lambda self: sanitize_filename_part(", ".join([style for style in self.p.styles if not style == "None"]) or "None", replace_spaces=False),
        'sampler': lambda self: sanitize_filename_part(self.p.sampler_name, replace_spaces=False),
        'model_hash': lambda self: ckpt_hash,
        'model_name': lambda self: sanitize_filename_part(os.path.basename(custom_checkpoint_path), replace_spaces=False),
        'date': lambda self: datetime.datetime.now().strftime('%Y-%m-%d'),
        'datetime': lambda self, *args: self.datetime(*args),  # accepts formats: [datetime], [datetime<Format>], [datetime<Format><Time Zone>]
        # 'job_timestamp': lambda self: getattr(self.p, "job_timestamp", shared.state.job_timestamp),
        'prompt_hash': lambda self: hashlib.sha256(self.prompt.encode()).hexdigest()[0:8],
        'prompt': lambda self: sanitize_filename_part(self.prompt),
        'prompt_no_styles': lambda self: "no_style",
        'prompt_spaces': lambda self: sanitize_filename_part(self.prompt, replace_spaces=False),
        'prompt_words': lambda self: "no words",
    }
    default_time_format = '%Y%m%d%H%M%S'

    def __init__(self, p, seed, prompt, image):
        # self.p = p
        self.seed = seed
        self.prompt = prompt
        self.image = image

    def prompt_words(self):
        words = [x for x in re_nonletters.split(self.prompt or "") if len(x) > 0]
        if len(words) == 0:
            words = ["empty"]
        return sanitize_filename_part(" ".join(words[0:30]), replace_spaces=False)

    def datetime(self, *args):
        time_datetime = datetime.datetime.now()

        time_format = args[0] if len(args) > 0 and args[0] != "" else self.default_time_format
        try:
            time_zone = pytz.timezone(args[1]) if len(args) > 1 else None
        except pytz.exceptions.UnknownTimeZoneError as _:
            time_zone = None

        time_zone_time = time_datetime.astimezone(time_zone)
        try:
            formatted_time = time_zone_time.strftime(time_format)
        except (ValueError, TypeError) as _:
            formatted_time = time_zone_time.strftime(self.default_time_format)

        return sanitize_filename_part(formatted_time, replace_spaces=False)

    def apply(self, x):
        res = ''

        for m in re_pattern.finditer(x):
            text, pattern = m.groups()
            res += text

            if pattern is None:
                continue

            pattern_args = []
            while True:
                m = re_pattern_arg.match(pattern)
                if m is None:
                    break

                pattern, arg = m.groups()
                pattern_args.insert(0, arg)

            fun = self.replacements.get(pattern.lower())
            if fun is not None:
                try:
                    replacement = fun(self, *pattern_args)
                except Exception:
                    replacement = None
                    print(f"Error adding [{pattern}] to filename", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)

                if replacement is not None:
                    res += str(replacement)
                    continue

            res += f'[{pattern}]'

        return res

def get_next_sequence_number(path, basename):
    """
    Determines and returns the next sequence number to use when saving an image in the specified directory.
    The sequence starts at 0.
    """
    result = -1
    if basename != '':
        basename = basename + "-"

    prefix_length = len(basename)
    for p in os.listdir(path):
        if p.startswith(basename):
            l = os.path.splitext(p[prefix_length:])[0].split('-')  # splits the filename (removing the basename first if one is defined, so the sequence number is always the first element)
            try:
                result = max(int(l[0]), result)
            except ValueError:
                pass

    return result + 1

def save_image(image, path, basename, seed=None, prompt=None, extension='png', info=None, short_filename=False, no_prompt=False, grid=False, pnginfo_section_name='parameters', p=None, existing_info=None, forced_filename=None, suffix="", save_to_dirs=None):
    """Save an image.
    Args:
        image (`PIL.Image`):
            The image to be saved.
        path (`str`):
            The directory to save the image. Note, the option `save_to_dirs` will make the image to be saved into a sub directory.
        basename (`str`):
            The base filename which will be applied to `filename pattern`.
        seed, prompt, short_filename,
        extension (`str`):
            Image file extension, default is `png`.
        pngsectionname (`str`):
            Specify the name of the section which `info` will be saved in.
        info (`str` or `PngImagePlugin.iTXt`):
            PNG info chunks.
        existing_info (`dict`):
            Additional PNG info. `existing_info == {pngsectionname: info, ...}`
        no_prompt:
            TODO I don't know its meaning.
        p (`StableDiffusionProcessing`)
        forced_filename (`str`):
            If specified, `basename` and filename pattern will be ignored.
        save_to_dirs (bool):
            If true, the image will be saved into a subdirectory of `path`.
    Returns: (fullfn, txt_fullfn)
        fullfn (`str`):
            The full path of the saved imaged.
        txt_fullfn (`str` or None):
            If a text file is saved for this image, this will be its full path. Otherwise None.
    """
    namegen = FilenameGenerator(p, seed, prompt, image)

    if save_to_dirs is None:
        save_to_dirs = (grid and opts.grid_save_to_dirs) or (not grid and opts.save_to_dirs and not no_prompt)

    if save_to_dirs:
        dirname = namegen.apply(opts.directories_filename_pattern or "[prompt_words]").lstrip(' ').rstrip('\\ /')
        path = os.path.join(path, dirname)

    os.makedirs(path, exist_ok=True)

    if forced_filename is None:
        if short_filename or seed is None:
            file_decoration = ""
        elif opts.save_to_dirs:
            file_decoration = opts.samples_filename_pattern or "[seed]"
        else:
            file_decoration = opts.samples_filename_pattern or "[seed]-[prompt_spaces]"

        add_number = file_decoration == ''

        if file_decoration != "" and add_number:
            file_decoration = "-" + file_decoration

        file_decoration = namegen.apply(file_decoration) + suffix

        if add_number:
            basecount = get_next_sequence_number(path, basename)
            fullfn = None
            for i in range(500):
                fn = f"{basecount + i:05}" if basename == '' else f"{basename}-{basecount + i:04}"
                fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
                if not os.path.exists(fullfn):
                    break
        else:
            fullfn = os.path.join(path, f"{file_decoration}.{extension}")
    else:
        fullfn = os.path.join(path, f"{forced_filename}.{extension}")

    pnginfo = existing_info or {}
    if info is not None:
        pnginfo[pnginfo_section_name] = info

    # params = script_callbacks.ImageSaveParams(image, p, fullfn, pnginfo)
    # script_callbacks.before_image_saved_callback(params)

    # image = params.image
    # fullfn = params.filename
    # info = params.pnginfo.get(pnginfo_section_name, None)

#image_embedding

import base64
import json
import numpy as np
import zlib
from PIL import Image, PngImagePlugin, ImageDraw, ImageFont
import torch



class EmbeddingEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return {'TORCHTENSOR': obj.cpu().detach().numpy().tolist()}
        return json.JSONEncoder.default(self, obj)


class EmbeddingDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, d):
        if 'TORCHTENSOR' in d:
            return torch.from_numpy(np.array(d['TORCHTENSOR']))
        return d


def embedding_to_b64(data):
    d = json.dumps(data, cls=EmbeddingEncoder)
    return base64.b64encode(d.encode())


def embedding_from_b64(data):
    d = base64.b64decode(data)
    return json.loads(d, cls=EmbeddingDecoder)


def lcg(m=2**32, a=1664525, c=1013904223, seed=0):
    while True:
        seed = (a * seed + c) % m
        yield seed % 255


def xor_block(block):
    g = lcg()
    randblock = np.array([next(g) for _ in range(np.product(block.shape))]).astype(np.uint8).reshape(block.shape)
    return np.bitwise_xor(block.astype(np.uint8), randblock & 0x0F)


def style_block(block, sequence):
    im = Image.new('RGB', (block.shape[1], block.shape[0]))
    draw = ImageDraw.Draw(im)
    i = 0
    for x in range(-6, im.size[0], 8):
        for yi, y in enumerate(range(-6, im.size[1], 8)):
            offset = 0
            if yi % 2 == 0:
                offset = 4
            shade = sequence[i % len(sequence)]
            i += 1
            draw.ellipse((x+offset, y, x+6+offset, y+6), fill=(shade, shade, shade))

    fg = np.array(im).astype(np.uint8) & 0xF0

    return block ^ fg


def insert_image_data_embed(image, data):
    d = 3
    data_compressed = zlib.compress(json.dumps(data, cls=EmbeddingEncoder).encode(), level=9)
    data_np_ = np.frombuffer(data_compressed, np.uint8).copy()
    data_np_high = data_np_ >> 4
    data_np_low = data_np_ & 0x0F

    h = image.size[1]
    next_size = data_np_low.shape[0] + (h-(data_np_low.shape[0] % h))
    next_size = next_size + ((h*d)-(next_size % (h*d)))

    data_np_low = np.resize(data_np_low, next_size)
    data_np_low = data_np_low.reshape((h, -1, d))

    data_np_high = np.resize(data_np_high, next_size)
    data_np_high = data_np_high.reshape((h, -1, d))

    edge_style = list(data['string_to_param'].values())[0].cpu().detach().numpy().tolist()[0][:1024]
    edge_style = (np.abs(edge_style)/np.max(np.abs(edge_style))*255).astype(np.uint8)

    data_np_low = style_block(data_np_low, sequence=edge_style)
    data_np_low = xor_block(data_np_low)
    data_np_high = style_block(data_np_high, sequence=edge_style[::-1])
    data_np_high = xor_block(data_np_high)

    im_low = Image.fromarray(data_np_low, mode='RGB')
    im_high = Image.fromarray(data_np_high, mode='RGB')

    background = Image.new('RGB', (image.size[0]+im_low.size[0]+im_high.size[0]+2, image.size[1]), (0, 0, 0))
    background.paste(im_low, (0, 0))
    background.paste(image, (im_low.size[0]+1, 0))
    background.paste(im_high, (im_low.size[0]+1+image.size[0]+1, 0))

    return background


def crop_black(img, tol=0):
    mask = (img > tol).all(2)
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), mask.shape[1]-mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), mask.shape[0]-mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def extract_image_data_embed(image):
    d = 3
    outarr = crop_black(np.array(image.convert('RGB').getdata()).reshape(image.size[1], image.size[0], d).astype(np.uint8)) & 0x0F
    black_cols = np.where(np.sum(outarr, axis=(0, 2)) == 0)
    if black_cols[0].shape[0] < 2:
        print('No Image data blocks found.')
        return None

    data_block_lower = outarr[:, :black_cols[0].min(), :].astype(np.uint8)
    data_block_upper = outarr[:, black_cols[0].max()+1:, :].astype(np.uint8)

    data_block_lower = xor_block(data_block_lower)
    data_block_upper = xor_block(data_block_upper)

    data_block = (data_block_upper << 4) | (data_block_lower)
    data_block = data_block.flatten().tobytes()

    data = zlib.decompress(data_block)
    return json.loads(data, cls=EmbeddingDecoder)


def caption_image_overlay(srcimage, title, footerLeft, footerMid, footerRight, textfont=None):
    from math import cos

    image = srcimage.copy()
    fontsize = 32
    if textfont is None:
        try:
            textfont = ImageFont.truetype(Roboto, fontsize)
            textfont = Roboto
        except Exception:
            textfont = Roboto

    factor = 1.5
    gradient = Image.new('RGBA', (1, image.size[1]), color=(0, 0, 0, 0))
    for y in range(image.size[1]):
        mag = 1-cos(y/image.size[1]*factor)
        mag = max(mag, 1-cos((image.size[1]-y)/image.size[1]*factor*1.1))
        gradient.putpixel((0, y), (0, 0, 0, int(mag*255)))
    image = Image.alpha_composite(image.convert('RGBA'), gradient.resize(image.size))

    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(textfont, fontsize)
    padding = 10

    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    fontsize = min(int(fontsize * (((image.size[0]*0.75)-(padding*4))/w)), 72)
    font = ImageFont.truetype(textfont, fontsize)
    _, _, w, h = draw.textbbox((0, 0), title, font=font)
    draw.text((padding, padding), title, anchor='lt', font=font, fill=(255, 255, 255, 230))

    _, _, w, h = draw.textbbox((0, 0), footerLeft, font=font)
    fontsize_left = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72)
    _, _, w, h = draw.textbbox((0, 0), footerMid, font=font)
    fontsize_mid = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72)
    _, _, w, h = draw.textbbox((0, 0), footerRight, font=font)
    fontsize_right = min(int(fontsize * (((image.size[0]/3)-(padding))/w)), 72)

    font = ImageFont.truetype(textfont, min(fontsize_left, fontsize_mid, fontsize_right))

    draw.text((padding, image.size[1]-padding),               footerLeft, anchor='ls', font=font, fill=(255, 255, 255, 230))
    draw.text((image.size[0]/2, image.size[1]-padding),       footerMid, anchor='ms', font=font, fill=(255, 255, 255, 230))
    draw.text((image.size[0]-padding, image.size[1]-padding), footerRight, anchor='rs', font=font, fill=(255, 255, 255, 230))

    return image


# if __name__ == '__main__':

#     testEmbed = Image.open('test_embedding.png')
#     data = extract_image_data_embed(testEmbed)
#     assert data is not None

#     data = embedding_from_b64(testEmbed.text['sd-ti-embedding'])
#     assert data is not None

#     image = Image.new('RGBA', (512, 512), (255, 255, 200, 255))
#     cap_image = caption_image_overlay(image, 'title', 'footerLeft', 'footerMid', 'footerRight')

#     test_embed = {'string_to_param': {'*': torch.from_numpy(np.random.random((2, 4096)))}}

#     embedded_image = insert_image_data_embed(cap_image, test_embed)

#     retrived_embed = extract_image_data_embed(embedded_image)

#     assert str(retrived_embed) == str(test_embed)

#     embedded_image2 = insert_image_data_embed(cap_image, retrived_embed)

#     assert embedded_image == embedded_image2

#     g = lcg()
#     shared_random = np.array([next(g) for _ in range(100)]).astype(np.uint8).tolist()

#     reference_random = [253, 242, 127,  44, 157,  27, 239, 133,  38,  79, 167,   4, 177,
#                          95, 130,  79,  78,  14,  52, 215, 220, 194, 126,  28, 240, 179,
#                         160, 153, 149,  50, 105,  14,  21, 218, 199,  18,  54, 198, 193,
#                          38, 128,  19,  53, 195, 124,  75, 205,  12,   6, 145,   0,  28,
#                          30, 148,   8,  45, 218, 171,  55, 249,  97, 166,  12,  35,   0,
#                          41, 221, 122, 215, 170,  31, 113, 186,  97, 119,  31,  23, 185,
#                          66, 140,  30,  41,  37,  63, 137, 109, 216,  55, 159, 145,  82,
#                          204, 86,  73, 222,  44, 198, 118, 240,  97]

#     assert shared_random == reference_random

#     hunna_kay_random_sum = sum(np.array([next(g) for _ in range(100000)]).astype(np.uint8).tolist())

#     assert 12731374 == hunna_kay_random_sum

#@title hijack_checkpoint

from torch.utils.checkpoint import checkpoint

import ldm.modules.attention
import ldm.modules.diffusionmodules.openaimodel


def BasicTransformerBlock_forward(self, x, context=None):
    return checkpoint(self._forward, x, context)


def AttentionBlock_forward(self, x):
    return checkpoint(self._forward, x)


def ResBlock_forward(self, x, emb):
    return checkpoint(self._forward, x, emb)


stored = []


def hijack_checkpoint_add():
    if len(stored) != 0:
        return

    stored.extend([
        ldm.modules.attention.BasicTransformerBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.ResBlock.forward,
        ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward
    ])

    ldm.modules.attention.BasicTransformerBlock.forward = BasicTransformerBlock_forward
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = ResBlock_forward
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = AttentionBlock_forward


def hijack_checkpoint_remove():
    if len(stored) == 0:
        return

    ldm.modules.attention.BasicTransformerBlock.forward = stored[0]
    ldm.modules.diffusionmodules.openaimodel.ResBlock.forward = stored[1]
    ldm.modules.diffusionmodules.openaimodel.AttentionBlock.forward = stored[2]

    stored.clear()


#learn_rate_scheduler

def autocast(disable=False):

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")

no_autocast = autocast()

class LearnScheduleIterator:
    def __init__(self, learn_rate, max_steps, cur_step=0):
        """
        specify learn_rate as "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000
        """
        print(learn_rate)
        pairs = learn_rate.split(',')
        self.rates = []
        self.it = 0
        self.maxit = 0
        try:
            for i, pair in enumerate(pairs):
                if not pair.strip():
                    continue
                tmp = pair.split(':')
                if len(tmp) == 2:
                    step = int(tmp[1])
                    if step > cur_step:
                        self.rates.append((float(tmp[0]), min(step, max_steps)))
                        self.maxit += 1
                        if step > max_steps:
                            return
                    elif step == -1:
                        self.rates.append((float(tmp[0]), max_steps))
                        self.maxit += 1
                        return
                else:
                    self.rates.append((float(tmp[0]), max_steps))
                    self.maxit += 1
                    return
            assert self.rates
        except (ValueError, AssertionError):
            raise Exception('Invalid learning rate schedule. It should be a number or, for example, like "0.001:100, 0.00001:1000, 1e-5:10000" to have lr of 0.001 until step 100, 0.00001 until 1000, and 1e-5 until 10000.')


    def __iter__(self):
        return self

    def __next__(self):
        if self.it < self.maxit:
            self.it += 1
            return self.rates[self.it - 1]
        else:
            raise StopIteration


class LearnRateScheduler:
    def __init__(self, learn_rate, max_steps, cur_step=0, verbose=True):
        self.schedules = LearnScheduleIterator(learn_rate, max_steps, cur_step)
        (self.learn_rate,  self.end_step) = next(self.schedules)
        self.verbose = verbose

        if self.verbose:
            print(f'Training at rate of {self.learn_rate} until step {self.end_step}')

        self.finished = False

    def step(self, step_number):
        if step_number < self.end_step:
            return False

        try:
            (self.learn_rate, self.end_step) = next(self.schedules)
        except StopIteration:
            self.finished = True
            return False
        return True

    def apply(self, optimizer, step_number):
        if not self.step(step_number):
            return

        if self.verbose:
            tqdm.tqdm.write(f'Training at rate of {self.learn_rate} until step {self.end_step}')

        for pg in optimizer.param_groups:
            pg['lr'] = self.learn_rate

#dataset
import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from collections import defaultdict
from random import shuffle, choices

import random
from tqdm.autonotebook import tqdm
import re

from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

re_numbers_at_start = re.compile(r"^[-\d]+\s*")


class DatasetEntry:
    def __init__(self, filename=None, filename_text=None, latent_dist=None, latent_sample=None, cond=None, cond_text=None, pixel_values=None):
        self.filename = filename
        self.filename_text = filename_text
        self.latent_dist = latent_dist
        self.latent_sample = latent_sample
        self.cond = cond
        self.cond_text = cond_text
        self.pixel_values = pixel_values

dataset_filename_join_string= " "
dataset_filename_word_regex = ""
class PersonalizedBase(Dataset):
    def __init__(self, data_root, width, height, repeats, flip_p=0.5, placeholder_token="*", model=None, cond_model=None, device=None, template_file=None, include_cond=False, batch_size=1, gradient_step=1, shuffle_tags=False, tag_drop_out=0, latent_sampling_method='once', varsize=False):
        re_word = re.compile(dataset_filename_word_regex) if len(dataset_filename_word_regex) > 0 else None

        self.placeholder_token = placeholder_token

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        self.dataset = []

        with open(template_file, "r") as file:
            lines = [x.strip() for x in file.readlines()]

        self.lines = lines

        assert data_root, 'dataset directory not specified'
        assert os.path.isdir(data_root), "Dataset directory doesn't exist"
        assert os.listdir(data_root), "Dataset directory is empty"

        self.image_paths = [os.path.join(data_root, file_path) for file_path in os.listdir(data_root)]

        self.shuffle_tags = shuffle_tags
        self.tag_drop_out = tag_drop_out
        groups = defaultdict(list)

        print("Preparing dataset...")
        for path in tqdm.tqdm(self.image_paths):
            # if shared.state.interrupted:
            #     raise Exception("interrupted")
            try:
                image = Image.open(path).convert('RGB')
                if not varsize:
                    image = image.resize((width, height), PIL.Image.BICUBIC)
            except Exception:
                continue

            text_filename = os.path.splitext(path)[0] + ".txt"
            filename = os.path.basename(path)

            if os.path.exists(text_filename):
                with open(text_filename, "r", encoding="utf8") as file:
                    filename_text = file.read()
            else:
                filename_text = os.path.splitext(filename)[0]
                filename_text = re.sub(re_numbers_at_start, '', filename_text)
                if re_word:
                    tokens = re_word.findall(filename_text)
                    filename_text = (dataset_filename_join_string or "").join(tokens)

            npimage = np.array(image).astype(np.uint8)
            npimage = (npimage / 127.5 - 1.0).astype(np.float32)

            torchdata = torch.from_numpy(npimage).permute(2, 0, 1).to(device_cuda, dtype=torch.float32)
            latent_sample = None
            #jpsaiart
            with t_autocast:
                latent_dist = model.encode_first_stage(torchdata.unsqueeze(dim=0))

            if latent_sampling_method == "deterministic":
                if isinstance(latent_dist, DiagonalGaussianDistribution):
                    # Works only for DiagonalGaussianDistribution
                    latent_dist.std = 0
                else:
                    latent_sampling_method = "once"
            latent_sample = model.get_first_stage_encoding(latent_dist).squeeze().to(device_cuda)
            if latent_sampling_method == "random":
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_dist=latent_dist)
            else:
                entry = DatasetEntry(filename=path, filename_text=filename_text, latent_sample=latent_sample)

            if not (self.tag_drop_out != 0 or self.shuffle_tags):
                entry.cond_text = self.create_text(filename_text)

            if include_cond and not (self.tag_drop_out != 0 or self.shuffle_tags):
                with t_autocast:
                    entry.cond = cond_model([entry.cond_text]).to(device_cuda).squeeze(0)
            groups[image.size].append(len(self.dataset))
            self.dataset.append(entry)
            del torchdata
            del latent_dist
            del latent_sample

        self.length = len(self.dataset)
        self.groups = list(groups.values())
        assert self.length > 0, "No images have been found in the dataset."
        self.batch_size = min(batch_size, self.length)
        self.gradient_step = min(gradient_step, self.length // self.batch_size)
        self.latent_sampling_method = latent_sampling_method

        if len(groups) > 1:
            print("Buckets:")
            for (w, h), ids in sorted(groups.items(), key=lambda x: x[0]):
                print(f"  {w}x{h}: {len(ids)}")
            print()

    def create_text(self, filename_text):
        text = random.choice(self.lines)
        tags = filename_text.split(',')
        if self.tag_drop_out != 0:
            tags = [t for t in tags if random.random() > self.tag_drop_out]
        if self.shuffle_tags:
            random.shuffle(tags)
        text = text.replace("[filewords]", ','.join(tags))
        text = text.replace("[name]", self.placeholder_token)
        return text

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        entry = self.dataset[i]
        if self.tag_drop_out != 0 or self.shuffle_tags:
            entry.cond_text = self.create_text(entry.filename_text)
        if self.latent_sampling_method == "random":
            entry.latent_sample = model.get_first_stage_encoding(entry.latent_dist).to(device_cuda)
        return entry


class GroupedBatchSampler(Sampler):
    def __init__(self, data_source: PersonalizedBase, batch_size: int):
        super().__init__(data_source)

        n = len(data_source)
        self.groups = data_source.groups
        self.len = n_batch = n // batch_size
        expected = [len(g) / n * n_batch * batch_size for g in data_source.groups]
        self.base = [int(e) // batch_size for e in expected]
        self.n_rand_batches = nrb = n_batch - sum(self.base)
        self.probs = [e%batch_size/nrb/batch_size if nrb>0 else 0 for e in expected]
        self.batch_size = batch_size

    def __len__(self):
        return self.len

    def __iter__(self):
        b = self.batch_size

        for g in self.groups:
            shuffle(g)

        batches = []
        for g in self.groups:
            batches.extend(g[i*b:(i+1)*b] for i in range(len(g) // b))
        for _ in range(self.n_rand_batches):
            rand_group = choices(self.groups, self.probs)[0]
            batches.append(choices(rand_group, k=b))

        shuffle(batches)

        yield from batches


class PersonalizedDataLoader(DataLoader):
    def __init__(self, dataset, latent_sampling_method="once", batch_size=1, pin_memory=False):
        super(PersonalizedDataLoader, self).__init__(dataset, batch_sampler=GroupedBatchSampler(dataset, batch_size), pin_memory=pin_memory)
        if latent_sampling_method == "random":
            self.collate_fn = collate_wrapper_random
        else:
            self.collate_fn = collate_wrapper


class BatchLoader:
    def __init__(self, data):
        self.cond_text = [entry.cond_text for entry in data]
        self.cond = [entry.cond for entry in data]
        self.latent_sample = torch.stack([entry.latent_sample for entry in data]).squeeze(1)
        #self.emb_index = [entry.emb_index for entry in data]
        #print(self.latent_sample.device)

    def pin_memory(self):
        self.latent_sample = self.latent_sample.pin_memory()
        return self

def collate_wrapper(batch):
    return BatchLoader(batch)

class BatchLoaderRandom(BatchLoader):
    def __init__(self, data):
        super().__init__(data)

    def pin_memory(self):
        return self

def collate_wrapper_random(batch):
    return BatchLoaderRandom(batch)

TextualInversionTemplate = namedtuple("TextualInversionTemplate", ["name", "path"])
textual_inversion_templates = {}

class Embedding:
    def __init__(self, vec, name, step=None):
        self.vec = vec
        self.name = name
        self.step = step
        self.shape = None
        self.vectors = 0
        self.cached_checksum = None
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.optimizer_state_dict = None
        self.filename = None

    def save(self, filename):
        embedding_data = {
            "string_to_token": {"*": 265},
            "string_to_param": {"*": self.vec},
            "name": self.name,
            "step": self.step,
            "sd_checkpoint": self.sd_checkpoint,
            "sd_checkpoint_name": self.sd_checkpoint_name,
        }

        torch.save(embedding_data, filename)

        save_optimizer_state=True
        if save_optimizer_state and self.optimizer_state_dict is not None:
            optimizer_saved_dict = {
                'hash': self.checksum(),
                'optimizer_state_dict': self.optimizer_state_dict,
            }
            torch.save(optimizer_saved_dict, filename + '.optim')

    def checksum(self):
        if self.cached_checksum is not None:
            return self.cached_checksum

        def const_hash(a):
            r = 0
            for v in a:
                r = (r * 281 ^ int(v) * 997) & 0xFFFFFFFF
            return r

        self.cached_checksum = f'{const_hash(self.vec.reshape(-1) * 100) & 0xffff:04x}'
        return self.cached_checksum


class DirWithTextualInversionEmbeddings:
    def __init__(self, path):
        self.path = path
        self.mtime = None

    def has_changed(self):
        if not os.path.isdir(self.path):
            return False

        mt = os.path.getmtime(self.path)
        if self.mtime is None or mt > self.mtime:
            return True

    def update(self):
        if not os.path.isdir(self.path):
            return

        self.mtime = os.path.getmtime(self.path)


class EmbeddingDatabase:
    def __init__(self):
        self.ids_lookup = {}
        self.word_embeddings = {}
        self.skipped_embeddings = {}
        self.expected_shape = -1
        self.embedding_dirs = {}

    def add_embedding_dir(self, path):
        self.embedding_dirs[path] = DirWithTextualInversionEmbeddings(path)

    def clear_embedding_dirs(self):
        self.embedding_dirs.clear()

    def register_embedding(self, embedding, model):
        self.word_embeddings[embedding.name] = embedding

        ids = model.cond_stage_model.tokenize([embedding.name])[0]

        first_id = ids[0]
        if first_id not in self.ids_lookup:
            self.ids_lookup[first_id] = []

        self.ids_lookup[first_id] = sorted(self.ids_lookup[first_id] + [(ids, embedding)], key=lambda x: len(x[0]), reverse=True)

        return embedding

    def get_expected_shape(self, model):
        vec = model.cond_stage_model.encode_embedding_init_text(",", 1)
        return vec.shape[1]

    def load_from_file(self, path, filename, model):
        name, ext = os.path.splitext(filename)
        ext = ext.upper()

        if ext in ['.PNG', '.WEBP', '.JXL', '.AVIF']:
            _, second_ext = os.path.splitext(name)
            if second_ext.upper() == '.PREVIEW':
                return

            embed_image = Image.open(path)
            if hasattr(embed_image, 'text') and 'sd-ti-embedding' in embed_image.text:
                data = embedding_from_b64(embed_image.text['sd-ti-embedding'])
                name = data.get('name', name)
            else:
                data = extract_image_data_embed(embed_image)
                name = data.get('name', name)
        elif ext in ['.BIN', '.PT']:
            data = torch.load(path, map_location=device_cuda)
        elif ext in ['.SAFETENSORS']:
            data = safetensors.torch.load_file(path, device=device_cuda)
        else:
            return

        # textual inversion embeddings
        if 'string_to_param' in data:
            param_dict = data['string_to_param']
            if hasattr(param_dict, '_parameters'):
                param_dict = getattr(param_dict, '_parameters')  # fix for torch 1.12.1 loading saved file from torch 1.11
            assert len(param_dict) == 1, 'embedding file has multiple terms in it'
            emb = next(iter(param_dict.items()))[1]
        # diffuser concepts
        elif type(data) == dict and type(next(iter(data.values()))) == torch.Tensor:
            assert len(data.keys()) == 1, 'embedding file has multiple terms in it'

            emb = next(iter(data.values()))
            if len(emb.shape) == 1:
                emb = emb.unsqueeze(0)
        else:
            raise Exception(f"Couldn't identify {filename} as neither textual inversion embedding nor diffuser concept.")

        vec = emb.detach().to(device_cuda, dtype=torch.float32)
        embedding = Embedding(vec, name)
        embedding.step = data.get('step', None)
        embedding.sd_checkpoint = data.get('sd_checkpoint', None)
        embedding.sd_checkpoint_name = data.get('sd_checkpoint_name', None)
        embedding.vectors = vec.shape[0]
        embedding.shape = vec.shape[-1]
        embedding.filename = path

        if self.expected_shape == -1 or self.expected_shape == embedding.shape:
            self.register_embedding(embedding, model)
        else:
            self.skipped_embeddings[name] = embedding

    def load_from_dir(self, embdir, model):
        if not os.path.isdir(embdir.path):
            return

        for root, dirs, fns in os.walk(embdir.path):
            for fn in fns:
                try:
                    fullfn = os.path.join(root, fn)

                    if os.stat(fullfn).st_size == 0:
                        continue

                    self.load_from_file(fullfn, fn, model)
                except Exception:
                    print(f"Error loading embedding {fn}:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    continue

    def load_textual_inversion_embeddings(self, model, force_reload=False):
        if not force_reload:
            need_reload = False
            for path, embdir in self.embedding_dirs.items():
                if embdir.has_changed():
                    need_reload = True
                    break

            if not need_reload:
                return

        self.ids_lookup.clear()
        self.word_embeddings.clear()
        self.skipped_embeddings.clear()
        self.expected_shape = self.get_expected_shape(model)

        for path, embdir in self.embedding_dirs.items():
            self.load_from_dir(embdir, model)
            embdir.update()

        print(f"Textual inversion embeddings loaded({len(self.word_embeddings)}): {', '.join(self.word_embeddings.keys())}")
        if len(self.skipped_embeddings) > 0:
            print(f"Textual inversion embeddings skipped({len(self.skipped_embeddings)}): {', '.join(self.skipped_embeddings.keys())}")

    def find_embedding_at_position(self, tokens, offset):
        token = tokens[offset]
        possible_matches = self.ids_lookup.get(token, None)

        if possible_matches is None:
            return None, None

        for ids, embedding in possible_matches:
            if tokens[offset:offset + len(ids)] == ids:
                return embedding, len(ids)

        return None, None

def txt2img_image_conditioning(model, x, width, height):
    if model.conditioning_key not in {'hybrid', 'concat'}:
        # Dummy zero conditioning if we're not using inpainting model.
        # Still takes up a bit of memory, but no encoder call.
        # Pretty sure we can just make this a 1x1 image since its not going to be used besides its batch size.
        return x.new_zeros(x.shape[0], 5, 1, 1, dtype=x.dtype, device=x.device)

    # The "masked-image" in this case will just be all zeros since the entire image is masked.
    image_conditioning = torch.zeros(x.shape[0], 3, height, width, device=x.device)
    image_conditioning = model.get_first_stage_encoding(model.encode_first_stage(image_conditioning))

    # Add the fake full 1s mask to the first dimension.
    image_conditioning = torch.nn.functional.pad(image_conditioning, (0, 0, 0, 0, 1, 0), value=1.0)
    image_conditioning = image_conditioning.to(x.dtype)

    return image_conditioning

#train embeddings
##markdown $ {\textsf{These Params are FOR TEXTUAL INVERSION TRAINING ONLY!}}$

def create_embedding(name, num_vectors_per_token, overwrite_old, init_text='*'):
    cond_model = model.cond_stage_model
    cond_model.to(device)
    # with t_autocast:
    #     cond_model([""])  # will send cond model to GPU if lowvram/medvram is active
    cond_model.to(device)
    #cond_model expects at least some text, so we provide '*' as backup.
    embedded = cond_model.encode_embedding_init_text(init_text or '*', num_vectors_per_token)
    vec = torch.zeros((num_vectors_per_token, embedded.shape[1]), device=device)

    #Only copy if we provided an init_text, otherwise keep vectors as zeros
    if init_text:
        for i in range(num_vectors_per_token):
            vec[i] = embedded[i * int(embedded.shape[0]) // num_vectors_per_token]

    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    fn = os.path.join(embeddings_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    embedding = Embedding(vec, name)
    embedding.step = 0
    embedding.save(fn)

    return fn

training_write_csv_every=0
def write_loss(log_directory, filename, step, epoch_len, values):
    if training_write_csv_every == 0:
        return

    if step % training_write_csv_every != 0:
        return
    write_csv_header = False if os.path.exists(os.path.join(log_directory, filename)) else True

    with open(os.path.join(log_directory, filename), "a+", newline='') as fout:
        csv_writer = csv.DictWriter(fout, fieldnames=["step", "epoch", "epoch_step", *(values.keys())])

        if write_csv_header:
            csv_writer.writeheader()

        epoch = (step - 1) // epoch_len
        epoch_step = (step - 1) % epoch_len

        csv_writer.writerow({
            "step": step,
            "epoch": epoch,
            "epoch_step": epoch_step,
            **values,
        })

def tensorboard_setup(log_directory):
    os.makedirs(os.path.join(log_directory, "tensorboard"), exist_ok=True)
    training_tensorboard_flush_every=120
    return SummaryWriter(
            log_dir=os.path.join(log_directory, "tensorboard"),
            flush_secs=training_tensorboard_flush_every)

def tensorboard_add(tensorboard_writer, loss, global_step, step, learn_rate, epoch_num):
    tensorboard_add_scaler(tensorboard_writer, "Loss/train", loss, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Loss/train/epoch-{epoch_num}", loss, step)
    tensorboard_add_scaler(tensorboard_writer, "Learn rate/train", learn_rate, global_step)
    tensorboard_add_scaler(tensorboard_writer, f"Learn rate/train/epoch-{epoch_num}", learn_rate, step)

def tensorboard_add_scaler(tensorboard_writer, tag, value, step):
    tensorboard_writer.add_scalar(tag=tag,
        scalar_value=value, global_step=step)

def tensorboard_add_image(tensorboard_writer, tag, pil_image, step):
    # Convert a pil image to a torch tensor
    img_tensor = torch.as_tensor(np.array(pil_image, copy=True))
    img_tensor = img_tensor.view(pil_image.size[1], pil_image.size[0],
        len(pil_image.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))

    tensorboard_writer.add_image(tag, img_tensor, global_step=step)
#textual_inversion.logging
import datetime
import json
import os

saved_params_shared = {"model_name", "model_hash", "initial_step", "num_of_dataset_images", "learn_rate", "batch_size", "clip_grad_mode", "clip_grad_value", "gradient_step", "data_root", "log_directory", "training_width", "training_height", "steps", "create_image_every", "template_file", "gradient_step", "latent_sampling_method"}
saved_params_ti = {"embedding_name", "num_vectors_per_token", "save_embedding_every", "save_image_with_stored_embedding"}
saved_params_hypernet = {"hypernetwork_name", "layer_structure", "activation_func", "weight_init", "add_layer_norm", "use_dropout", "save_hypernetwork_every"}
saved_params_all = saved_params_shared | saved_params_ti | saved_params_hypernet
saved_params_previews = {"preview_prompt", "preview_negative_prompt", "preview_steps", "preview_sampler_index", "preview_cfg_scale", "preview_seed", "preview_width", "preview_height"}

def save_settings_to_file(log_directory, all_params):
        now = datetime.datetime.now()
        params = {"datetime": now.strftime("%Y-%m-%d %H:%M:%S")}

        keys = saved_params_all
        if all_params.get('preview_from_txt2img'):
          keys = keys | saved_params_previews

        params.update({k: v for k, v in all_params.items() if k in keys})

        filename = f'settings-{now.strftime("%Y-%m-%d-%H-%M-%S")}.json'
        with open(os.path.join(log_directory, filename), "w") as file:
            json.dump(params, file, indent=4)

def validate_train_inputs(model_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_model_every, create_image_every, log_directory, name="embedding"):
    assert model_name, f"{name} not selected"
    assert learn_rate, "Learning rate is empty or 0"
    assert isinstance(batch_size, int), "Batch size must be integer"
    assert batch_size > 0, "Batch size must be positive"
    assert isinstance(gradient_step, int), "Gradient accumulation step must be integer"
    assert gradient_step > 0, "Gradient accumulation step must be positive"
    assert data_root, "Dataset directory is empty"
    assert os.path.isdir(data_root), "Dataset directory doesn't exist"
    assert os.listdir(data_root), "Dataset directory is empty"
    assert template_filename, "Prompt template file not selected"
    assert template_file, f"Prompt template file {template_filename} not found"
    assert os.path.isfile(template_file), f"Prompt template file {template_filename} doesn't exist"
    assert steps, "Max steps is empty or 0"
    assert isinstance(steps, int), "Max steps must be integer"
    assert steps > 0, "Max steps must be positive"
    assert isinstance(save_model_every, int), "Save {name} must be integer"
    assert save_model_every >= 0, "Save {name} must be positive or 0"
    assert isinstance(create_image_every, int), "Create image must be integer"
    assert create_image_every >= 0, "Create image must be positive or 0"
    if save_model_every or create_image_every:
        assert log_directory, "Log directory is empty"
import tqdm
def train_embedding(id_task, embedding_name, learn_rate, batch_size, gradient_step, data_root, log_directory, training_width, training_height, varsize, steps, clip_grad_mode, clip_grad_value, shuffle_tags, tag_drop_out, latent_sampling_method, create_image_every, save_embedding_every, template_filename, save_image_with_stored_embedding, preview_from_txt2img, preview_prompt, preview_negative_prompt, preview_steps, preview_sampler_index, preview_cfg_scale, preview_seed, preview_width, preview_height):
    save_embedding_every = save_embedding_every or 0
    create_image_every = create_image_every or 0
    template_file = template_filename
    validate_train_inputs(embedding_name, learn_rate, batch_size, gradient_step, data_root, template_file, template_filename, steps, save_embedding_every, create_image_every, log_directory, name="embedding")
    template_file = template_file

    job = "train-embedding"
    textinfo = "Initializing textual inversion training..."
    job_count = steps

    filename = os.path.join(embeddings_dir, f'{embedding_name}.pt')

    log_directory = os.path.join(log_directory, datetime.datetime.now().strftime("%Y-%m-%d"), embedding_name)
    unload = False

    if save_embedding_every > 0:
        embedding_dir = os.path.join(log_directory, "embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
    else:
        embedding_dir = None

    if create_image_every > 0:
        images_dir = os.path.join(log_directory, "images")
        os.makedirs(images_dir, exist_ok=True)
    else:
        images_dir = None

    if create_image_every > 0 and save_image_with_stored_embedding:
        images_embeds_dir = os.path.join(log_directory, "image_embeddings")
        os.makedirs(images_embeds_dir, exist_ok=True)
    else:
        images_embeds_dir = None

    hijack = model_hijack

    embedding = hijack.embedding_db.word_embeddings[embedding_name]
    # checkpoint = models.select_checkpoint()

    initial_step = embedding.step or 0
    if initial_step >= steps:
        textinfo = "Model has already been trained beyond specified max steps"
        return embedding, filename

    scheduler = LearnRateScheduler(learn_rate, steps, initial_step)
    clip_grad = torch.nn.utils.clip_grad_value_ if clip_grad_mode == "value" else \
        torch.nn.utils.clip_grad_norm_ if clip_grad_mode == "norm" else \
        None
    if clip_grad:
        clip_grad_sched = LearnRateScheduler(clip_grad_value, steps, initial_step, verbose=False)
    # dataset loading may take a while, so input validations and early returns should be done before this
    textinfo = f"Preparing dataset from {html.escape(data_root)}..."
    old_parallel_processing_allowed = False

    training_enable_tensorboard=True
    if training_enable_tensorboard:
        tensorboard_writer = tensorboard_setup(log_directory)

    pin_memory = opts_pin_memory

    ds = PersonalizedBase(data_root=data_root, width=training_width, height=training_height, repeats=1, placeholder_token=embedding_name, model=model, cond_model=model.cond_stage_model, device=device_cuda, template_file=template_file, batch_size=batch_size, gradient_step=gradient_step, shuffle_tags=shuffle_tags, tag_drop_out=tag_drop_out, latent_sampling_method=latent_sampling_method, varsize=varsize)

    save_training_settings_to_txt=True
    if save_training_settings_to_txt:
        save_settings_to_file(log_directory, {**dict(model_name=os.path.basename(custom_checkpoint_path), model_hash=ckpt_hash, num_of_dataset_images=len(ds), num_vectors_per_token=len(embedding.vec)), **locals()})

    latent_sampling_method = ds.latent_sampling_method

    dl = PersonalizedDataLoader(ds, latent_sampling_method=latent_sampling_method, batch_size=ds.batch_size, pin_memory=pin_memory)

    if unload:
        parallel_processing_allowed = False
        model.first_stage_model.to(devices.cpu)

    embedding.vec.requires_grad = True
    optimizer = torch.optim.AdamW([embedding.vec], lr=scheduler.learn_rate, weight_decay=0.0)
    save_optimizer_state=True
    if save_optimizer_state:
        optimizer_state_dict = None
        if os.path.exists(filename + '.optim'):
            optimizer_saved_dict = torch.load(filename + '.optim', map_location='cpu')
            if embedding.checksum() == optimizer_saved_dict.get('hash', None):
                optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)

        if optimizer_state_dict is not None:
            optimizer.load_state_dict(optimizer_state_dict)
            print("Loaded existing optimizer from checkpoint")
        else:
            print("No saved optimizer exists in checkpoint")

    scaler = torch.cuda.amp.GradScaler()

    batch_size = ds.batch_size
    gradient_step = ds.gradient_step
    # n steps = batch_size * gradient_step * n image processed
    steps_per_epoch = len(ds) // batch_size // gradient_step
    max_steps_per_epoch = len(ds) // batch_size - (len(ds) // batch_size) % gradient_step
    loss_step = 0
    _loss_step = 0 #internal

    last_saved_file = "<none>"
    last_saved_image = "<none>"
    forced_filename = "<none>"
    embedding_yet_to_be_embedded = False

    # is_training_inpainting_model = model.conditioning_key in {'hybrid', 'concat'}
    img_c = None

    pbar = tqdm.tqdm(total=steps - initial_step)
    try:
        hijack_checkpoint_add()

        for i in range((steps-initial_step) * gradient_step):
            if scheduler.finished:
                break
            # if interrupted:
            #     break
            for j, batch in enumerate(dl):
                # works as a drop_last=True for gradient accumulation
                if j == max_steps_per_epoch:
                    break
                scheduler.apply(optimizer, embedding.step)
                if scheduler.finished:
                    break
                # if shared.state.interrupted:
                #     break
                if clip_grad:
                    clip_grad_sched.step(embedding.step)
                with t_autocast:
                    x = batch.latent_sample.to(device, non_blocking=pin_memory)
                    c = model.cond_stage_model(batch.cond_text)
                    is_training_inpainting_model=False
                    if is_training_inpainting_model:
                        if img_c is None:
                            img_c = txt2img_image_conditioning(model, c, training_width, training_height)

                        cond = {"c_concat": [img_c], "c_crossattn": [c]}
                    else:
                        cond = c
                    cond = c

                    xloss = model.forward(x, cond)[0] / gradient_step
                    del x

                    _loss_step += xloss.item()
                    # print(f'LOSS ITEMS: {xloss.item()}')
                scaler.scale(xloss).backward()
                # xloss.backward()

                # go back until we reach gradient accumulation steps
                if (j + 1) % gradient_step != 0:
                    continue

                if clip_grad:
                    clip_grad(embedding.vec, clip_grad_sched.learn_rate)

                scaler.step(optimizer)
                # optimizer.step()
                scaler.update()
                embedding.step += 1
                pbar.update()
                optimizer.zero_grad(set_to_none=True)
                loss_step = _loss_step
                _loss_step = 0

                steps_done = embedding.step + 1

                epoch_num = embedding.step // steps_per_epoch
                epoch_step = embedding.step % steps_per_epoch

                description = f"Training textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}"
                pbar.set_description(description)
                # print(f"\033[31mTraining textual inversion [Epoch {epoch_num}: {epoch_step+1}/{steps_per_epoch}] loss: {loss_step:.7f}\033[0m")
                if embedding_dir is not None and steps_done % save_embedding_every == 0:
                    # Before saving, change name to match current checkpoint.
                    embedding_name_every = f'{embedding_name}-{steps_done}'
                    last_saved_file = os.path.join(embedding_dir, f'{embedding_name_every}.pt')
                    save_embedding(embedding, optimizer, embedding_name_every, last_saved_file, remove_cached_checksum=True)
                    embedding_yet_to_be_embedded = True

                write_loss(log_directory, "textual_inversion_loss.csv", embedding.step, steps_per_epoch, {
                    "loss": f"{loss_step:.7f}",
                    "learn_rate": scheduler.learn_rate
                })

                training_tensorboard_save_images=False

                if images_dir is not None and steps_done % create_image_every == 0:
                    forced_filename = f'{embedding_name}-{steps_done}'
                    last_saved_image = os.path.join(images_dir, forced_filename)

                    model.first_stage_model.to(device_cuda)

                    # p = processing.StableDiffusionProcessingTxt2Img(
                    #     model=shared.model,
                    #     do_not_save_grid=True,
                    #     do_not_save_samples=True,
                    #     do_not_reload_embeddings=True,
                    # )

                    if preview_from_txt2img:
                        p.prompt = preview_prompt
                        p.negative_prompt = preview_negative_prompt
                        p.steps = preview_steps
                        p.sampler_name = sd_samplers.samplers[preview_sampler_index].name
                        p.cfg_scale = preview_cfg_scale
                        p.seed = preview_seed
                        p.width = preview_width
                        p.height = preview_height
                    else:
                        cond_prompts = batch.cond_text[0]
                        uncond_prompt = {}
                        steps = 20
                        width = training_width
                        height = training_height
                        preview_text = cond_prompts
                    cond_prompts = batch.cond_text[0]
                    print(f"COND_PROMPTS IS: {cond_prompts}")
                    steps = 20
                    width = training_width
                    height = training_height
                    preview_text = cond_prompts
                    preview_text = cond_prompts
                    cond_prompts = str(cond_prompts)
                    uncond_prompt = "lowres, nsfw"
                    arg_seed = 27
                    sample, image = generate(embedding_name, embedding_dir, cond_prompts, uncond_prompt, width, height, arg_seed, root, return_latent=False, return_sample=True, return_c=False)
                    display.display(image)
                    # processed = processing.process_images(p)
                    # image = processed.images[0] if len(processed.images) > 0 else None
                    if unload:
                        model.first_stage_model.to(device_cuda)

                    samples_format = 'png'
                    # processed.infotexts = ""
                    #jpsaiart
                    image_idx = 0
                    if image is not None:
                        image.save(os.path.join(images_dir, f'image{image_idx}.png'))
                        # last_saved_image, last_text_info = save_image(image, images_dir, "", arg_seed, cond_prompts, samples_format, info=None, p=None, forced_filename=forced_filename, save_to_dirs=False)
                        # last_saved_image += f", prompt: {preview_text}"

                        if training_enable_tensorboard and training_tensorboard_save_images:
                            tensorboard_add_image(tensorboard_writer, f"Validation at epoch {epoch_num}", image, embedding.step)
                    image_idx+=1

                    if save_image_with_stored_embedding and os.path.exists(last_saved_file) and embedding_yet_to_be_embedded:

                        last_saved_image_chunks = os.path.join(images_embeds_dir, f'{embedding_name}-{steps_done}.png')

                        info = PngImagePlugin.PngInfo()
                        data = torch.load(last_saved_file)
                        info.add_text("sd-ti-embedding", embedding_to_b64(data))

                        title = "<{}>".format(data.get('name', '???'))

                        try:
                            vectorSize = list(data['string_to_param'].values())[0].shape[0]
                        except Exception as e:
                            vectorSize = '?'

                        checkpoint = model_checkpoint or custom_checkpoint_path
                        footer_left = os.path.basename(checkpoint)
                        footer_mid = '[{}]'.format(ckpt_hash)
                        footer_right = '{}v {}s'.format(vectorSize, steps_done)

                        captioned_image = caption_image_overlay(image, title, footer_left, footer_mid, footer_right)
                        captioned_image = insert_image_data_embed(captioned_image, data)

                        captioned_image.save(last_saved_image_chunks, "PNG", pnginfo=info)
                        embedding_yet_to_be_embedded = False

                    # last_saved_image, last_text_info = save_image(image, images_dir, "", arg_seed, cond_prompts, samples_format, info=None, p=None, forced_filename=forced_filename, save_to_dirs=False)
                    # last_saved_image += f", prompt: {preview_text}"

                job_no = embedding.step

                textinfo = f"""
# <p>
# Loss: {loss_step:.7f}<br/>
# Step: {steps_done}<br/>
# Last prompt: {html.escape(batch.cond_text[0])}<br/>
# Last saved embedding: {html.escape(last_saved_file)}<br/>
# Last saved image: {html.escape(last_saved_image)}<br/>
# </p>
# """
        filename = os.path.join(embeddings_dir, f'{embedding_name}.pt')
        save_embedding(embedding, optimizer, embedding_name, filename, remove_cached_checksum=True)
    except Exception:
        print(traceback.format_exc(), file=sys.stderr)
        pass
    finally:
        pbar.leave = False
        pbar.close()
        model.first_stage_model.to(device)
        parallel_processing_allowed = old_parallel_processing_allowed
        hijack_checkpoint_remove()

    return embedding, filename


def save_embedding(embedding, optimizer, embedding_name, filename, remove_cached_checksum=True):
    old_embedding_name = embedding.name
    old_sd_checkpoint = embedding.sd_checkpoint if hasattr(embedding, "sd_checkpoint") else None
    old_sd_checkpoint_name = embedding.sd_checkpoint_name if hasattr(embedding, "sd_checkpoint_name") else None
    old_cached_checksum = embedding.cached_checksum if hasattr(embedding, "cached_checksum") else None
    try:
        embedding.sd_checkpoint = ckpt_hash
        custom_checkpoint_path = input()
        embedding.sd_checkpoint_name = os.path.basename(custom_checkpoint_path)
        if remove_cached_checksum:
            embedding.cached_checksum = None
        embedding.name = embedding_name
        embedding.optimizer_state_dict = optimizer.state_dict()
        embedding.save(filename)
    except:
        embedding.sd_checkpoint = old_sd_checkpoint
        embedding.sd_checkpoint_name = old_sd_checkpoint_name
        embedding.name = old_embedding_name
        embedding.cached_checksum = old_cached_checksum
        raise

#cond_func
import importlib

class CondFunc:
    def __new__(cls, orig_func, sub_func, cond_func):
        self = super(CondFunc, cls).__new__(cls)
        if isinstance(orig_func, str):
            func_path = orig_func.split('.')
            for i in range(len(func_path)-1, -1, -1):
                try:
                    resolved_obj = importlib.import_module('.'.join(func_path[:i]))
                    break
                except ImportError:
                    pass
            for attr_name in func_path[i:-1]:
                resolved_obj = getattr(resolved_obj, attr_name)
            orig_func = getattr(resolved_obj, func_path[-1])
            setattr(resolved_obj, func_path[-1], lambda *args, **kwargs: self(*args, **kwargs))
        self.__init__(orig_func, sub_func, cond_func)
        return lambda *args, **kwargs: self(*args, **kwargs)
    def __init__(self, orig_func, sub_func, cond_func):
        self.__orig_func = orig_func
        self.__sub_func = sub_func
        self.__cond_func = cond_func
    def __call__(self, *args, **kwargs):
        if not self.__cond_func or self.__cond_func(self.__orig_func, *args, **kwargs):
            return self.__sub_func(self.__orig_func, *args, **kwargs)
        else:
            return self.__orig_func(*args, **kwargs)

#errors
import sys
import traceback


def print_error_explanation(message):
    lines = message.strip().split("\n")
    max_len = max([len(x) for x in lines])

    print('=' * max_len, file=sys.stderr)
    for line in lines:
        print(line, file=sys.stderr)
    print('=' * max_len, file=sys.stderr)


# def display(e: Exception, task):
#     print(f"{task or 'error'}: {type(e).__name__}", file=sys.stderr)
#     print(traceback.format_exc(), file=sys.stderr)

#     message = str(e)
#     if "copying a param with shape torch.Size([640, 1024]) from checkpoint, the shape in current model is torch.Size([640, 768])" in message:
#         print_error_explanation("""
# The most likely cause of this is you are trying to load Stable Diffusion 2.0 model without specifying its config file.
# See https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Features#stable-diffusion-20 for how to solve this.
#         """)


# already_displayed = {}


def display_once(e: Exception, task):
    if task in already_displayed:
        return

    display(e, task)

    already_displayed[task] = 1


def run(code, task):
    try:
        code()
    except Exception as e:
        display(task, e)

#sub_quad_attn
from functools import partial
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint
import math
from typing import Optional, NamedTuple, List

def without_autocast(disable=False):
    return torch.autocast("cuda", enabled=False) if torch.is_autocast_enabled() and not disable else contextlib.nullcontext()

def narrow_trunc(
    input: Tensor,
    dim: int,
    start: int,
    length: int
) -> Tensor:
    return torch.narrow(input, dim, start, length if input.shape[dim] >= start + length else input.shape[dim] - start)


class AttnChunk(NamedTuple):
    exp_values: Tensor
    exp_weights_sum: Tensor
    max_score: Tensor


class SummarizeChunk:
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> AttnChunk: ...


class ComputeQueryChunkAttn:
    @staticmethod
    def __call__(
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor: ...


def _summarize_chunk(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> AttnChunk:
    attn_weights = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    max_score, _ = torch.max(attn_weights, -1, keepdim=True)
    max_score = max_score.detach()
    exp_weights = torch.exp(attn_weights - max_score)
    exp_values = torch.bmm(exp_weights, value) if query.device.type == 'mps' else torch.bmm(exp_weights, value.to(exp_weights.dtype)).to(value.dtype)
    max_score = max_score.squeeze(-1)
    return AttnChunk(exp_values, exp_weights.sum(dim=-1), max_score)


def _query_chunk_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    summarize_chunk: SummarizeChunk,
    kv_chunk_size: int,
) -> Tensor:
    batch_x_heads, k_tokens, k_channels_per_head = key.shape
    _, _, v_channels_per_head = value.shape

    def chunk_scanner(chunk_idx: int) -> AttnChunk:
        key_chunk = narrow_trunc(
            key,
            1,
            chunk_idx,
            kv_chunk_size
        )
        value_chunk = narrow_trunc(
            value,
            1,
            chunk_idx,
            kv_chunk_size
        )
        return summarize_chunk(query, key_chunk, value_chunk)

    chunks: List[AttnChunk] = [
        chunk_scanner(chunk) for chunk in torch.arange(0, k_tokens, kv_chunk_size)
    ]
    acc_chunk = AttnChunk(*map(torch.stack, zip(*chunks)))
    chunk_values, chunk_weights, chunk_max = acc_chunk

    global_max, _ = torch.max(chunk_max, 0, keepdim=True)
    max_diffs = torch.exp(chunk_max - global_max)
    chunk_values *= torch.unsqueeze(max_diffs, -1)
    chunk_weights *= max_diffs

    all_values = chunk_values.sum(dim=0)
    all_weights = torch.unsqueeze(chunk_weights, -1).sum(dim=0)
    return all_values / all_weights


# TODO: refactor CrossAttention#get_attention_scores to share code with this
def _get_attention_scores_no_kv_chunking(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: float,
) -> Tensor:
    attn_scores = torch.baddbmm(
        torch.empty(1, 1, 1, device=query.device, dtype=query.dtype),
        query,
        key.transpose(1,2),
        alpha=scale,
        beta=0,
    )
    attn_probs = attn_scores.softmax(dim=-1)
    del attn_scores
    hidden_states_slice = torch.bmm(attn_probs, value) if query.device.type == 'mps' else torch.bmm(attn_probs, value.to(attn_probs.dtype)).to(value.dtype)
    return hidden_states_slice

#sub_quadratic_attention
class ScannedChunk(NamedTuple):
    chunk_idx: int
    attn_chunk: AttnChunk

def efficient_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    query_chunk_size=1024,
    kv_chunk_size: Optional[int] = None,
    kv_chunk_size_min: Optional[int] = None,
    use_checkpoint=True,
):
    """Computes efficient dot-product attention given query, key, and value.
      This is efficient version of attention presented in
      https://arxiv.org/abs/2112.05682v2 which comes with O(sqrt(n)) memory requirements.
      Args:
        query: queries for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        key: keys for calculating attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        value: values to be used in attention with shape of
          `[batch * num_heads, tokens, channels_per_head]`.
        query_chunk_size: int: query chunks size
        kv_chunk_size: Optional[int]: key/value chunks size. if None: defaults to sqrt(key_tokens)
        kv_chunk_size_min: Optional[int]: key/value minimum chunk size. only considered when kv_chunk_size is None. changes `sqrt(key_tokens)` into `max(sqrt(key_tokens), kv_chunk_size_min)`, to ensure our chunk sizes don't get too small (smaller chunks = more chunks = less concurrent work done).
        use_checkpoint: bool: whether to use checkpointing (recommended True for training, False for inference)
      Returns:
        Output of shape `[batch * num_heads, query_tokens, channels_per_head]`.
      """
    batch_x_heads, q_tokens, q_channels_per_head = query.shape
    _, k_tokens, _ = key.shape
    scale = q_channels_per_head ** -0.5

    kv_chunk_size = min(kv_chunk_size or int(math.sqrt(k_tokens)), k_tokens)
    if kv_chunk_size_min is not None:
        kv_chunk_size = max(kv_chunk_size, kv_chunk_size_min)

    def get_query_chunk(chunk_idx: int) -> Tensor:
        return narrow_trunc(
            query,
            1,
            chunk_idx,
            min(query_chunk_size, q_tokens)
        )

    summarize_chunk: SummarizeChunk = partial(_summarize_chunk, scale=scale)
    summarize_chunk: SummarizeChunk = partial(checkpoint, summarize_chunk) if use_checkpoint else summarize_chunk
    compute_query_chunk_attn: ComputeQueryChunkAttn = partial(
        _get_attention_scores_no_kv_chunking,
        scale=scale
    ) if k_tokens <= kv_chunk_size else (
        # fast-path for when there's just 1 key-value chunk per query chunk (this is just sliced attention btw)
        partial(
            _query_chunk_attention,
            kv_chunk_size=kv_chunk_size,
            summarize_chunk=summarize_chunk,
        )
    )

    if q_tokens <= query_chunk_size:
        # fast-path for when there's just 1 query chunk
        return compute_query_chunk_attn(
            query=query,
            key=key,
            value=value,
        )

    # TODO: maybe we should use torch.empty_like(query) to allocate storage in-advance,
    # and pass slices to be mutated, instead of torch.cat()ing the returned slices
    res = torch.cat([
        compute_query_chunk_attn(
            query=get_query_chunk(i * query_chunk_size),
            key=key,
            value=value,
        ) for i in range(math.ceil(q_tokens / query_chunk_size))
    ], dim=1)
    return res

#cache
import hashlib
import json
import os.path

import filelock

cache_filename = os.path.join(data_dir, "cache.json")
cache_data = None


def dump_cache():
    with filelock.FileLock(cache_filename+".lock"):
        with open(cache_filename, "w", encoding="utf8") as file:
            json.dump(cache_data, file, indent=4)


def cache(subsection):
    global cache_data

    if cache_data is None:
        with filelock.FileLock(cache_filename+".lock"):
            if not os.path.isfile(cache_filename):
                cache_data = {}
            else:
                with open(cache_filename, "r", encoding="utf8") as file:
                    cache_data = json.load(file)

    s = cache_data.get(subsection, {})
    cache_data[subsection] = s

    return s


def calculate_sha256(filename):
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(blksize), b""):
            hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def sha256_from_cache(filename, title):
    hashes = cache("hashes")
    ondisk_mtime = os.path.getmtime(filename)

    if title not in hashes:
        return None

    cached_sha256 = hashes[title].get("sha256", None)
    cached_mtime = hashes[title].get("mtime", 0)

    if ondisk_mtime > cached_mtime or cached_sha256 is None:
        return None

    return cached_sha256


def sha256(filename, title):
    hashes = cache("hashes")

    sha256_value = sha256_from_cache(filename, title)
    if sha256_value is not None:
        return sha256_value

    print(f"Calculating sha256 for {filename}: ", end='')
    sha256_value = calculate_sha256(filename)
    print(f"{sha256_value}")

    hashes[title] = {
        "mtime": os.path.getmtime(filename),
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value

#hypernetwork
loaded_hypernetworks=[]
import csv
import datetime
import glob
import html
import os
import sys
import traceback
import inspect

import torch
from einops import rearrange, repeat
from ldm.util import default
from torch import einsum
from torch.nn.init import normal_, xavier_normal_, xavier_uniform_, kaiming_normal_, kaiming_uniform_, zeros_

from collections import defaultdict, deque
from statistics import stdev, mean


optimizer_dict = {optim_name : cls_obj for optim_name, cls_obj in inspect.getmembers(torch.optim, inspect.isclass) if optim_name != "Optimizer"}

class HypernetworkModule(torch.nn.Module):
    activation_dict = {
        "linear": torch.nn.Identity,
        "relu": torch.nn.ReLU,
        "leakyrelu": torch.nn.LeakyReLU,
        "elu": torch.nn.ELU,
        "swish": torch.nn.Hardswish,
        "tanh": torch.nn.Tanh,
        "sigmoid": torch.nn.Sigmoid,
    }
    activation_dict.update({cls_name.lower(): cls_obj for cls_name, cls_obj in inspect.getmembers(torch.nn.modules.activation) if inspect.isclass(cls_obj) and cls_obj.__module__ == 'torch.nn.modules.activation'})

    def __init__(self, dim, state_dict=None, layer_structure=None, activation_func=None, weight_init='Normal',
                 add_layer_norm=False, activate_output=False, dropout_structure=None):
        super().__init__()

        self.multiplier = 1.0

        assert layer_structure is not None, "layer_structure must not be None"
        assert layer_structure[0] == 1, "Multiplier Sequence should start with size 1!"
        assert layer_structure[-1] == 1, "Multiplier Sequence should end with size 1!"

        linears = []
        for i in range(len(layer_structure) - 1):

            # Add a fully-connected layer
            linears.append(torch.nn.Linear(int(dim * layer_structure[i]), int(dim * layer_structure[i+1])))

            # Add an activation func except last layer
            if activation_func == "linear" or activation_func is None or (i >= len(layer_structure) - 2 and not activate_output):
                pass
            elif activation_func in self.activation_dict:
                linears.append(self.activation_dict[activation_func]())
            else:
                raise RuntimeError(f'hypernetwork uses an unsupported activation function: {activation_func}')

            # Add layer normalization
            if add_layer_norm:
                linears.append(torch.nn.LayerNorm(int(dim * layer_structure[i+1])))

            # Everything should be now parsed into dropout structure, and applied here.
            # Since we only have dropouts after layers, dropout structure should start with 0 and end with 0.
            if dropout_structure is not None and dropout_structure[i+1] > 0:
                assert 0 < dropout_structure[i+1] < 1, "Dropout probability should be 0 or float between 0 and 1!"
                linears.append(torch.nn.Dropout(p=dropout_structure[i+1]))
            # Code explanation : [1, 2, 1] -> dropout is missing when last_layer_dropout is false. [1, 2, 2, 1] -> [0, 0.3, 0, 0], when its True, [0, 0.3, 0.3, 0].

        self.linear = torch.nn.Sequential(*linears)

        if state_dict is not None:
            self.fix_old_state_dict(state_dict)
            self.load_state_dict(state_dict)
        else:
            for layer in self.linear:
                if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                    w, b = layer.weight.data, layer.bias.data
                    if weight_init == "Normal" or type(layer) == torch.nn.LayerNorm:
                        normal_(w, mean=0.0, std=0.01)
                        normal_(b, mean=0.0, std=0)
                    elif weight_init == 'XavierUniform':
                        xavier_uniform_(w)
                        zeros_(b)
                    elif weight_init == 'XavierNormal':
                        xavier_normal_(w)
                        zeros_(b)
                    elif weight_init == 'KaimingUniform':
                        kaiming_uniform_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    elif weight_init == 'KaimingNormal':
                        kaiming_normal_(w, nonlinearity='leaky_relu' if 'leakyrelu' == activation_func else 'relu')
                        zeros_(b)
                    else:
                        raise KeyError(f"Key {weight_init} is not defined as initialization!")
        self.to(device)

    def fix_old_state_dict(self, state_dict):
        changes = {
            'linear1.bias': 'linear.0.bias',
            'linear1.weight': 'linear.0.weight',
            'linear2.bias': 'linear.1.bias',
            'linear2.weight': 'linear.1.weight',
        }

        for fr, to in changes.items():
            x = state_dict.get(fr, None)
            if x is None:
                continue

            del state_dict[fr]
            state_dict[to] = x

    def forward(self, x):
        return x + self.linear(x) * (self.multiplier if not self.training else 1)

    def trainables(self):
        layer_structure = []
        for layer in self.linear:
            if type(layer) == torch.nn.Linear or type(layer) == torch.nn.LayerNorm:
                layer_structure += [layer.weight, layer.bias]
        return layer_structure


#param layer_structure : sequence used for length, use_dropout : controlling boolean, last_layer_dropout : for compatibility check.
def parse_dropout_structure(layer_structure, use_dropout, last_layer_dropout):
    if layer_structure is None:
        layer_structure = [1, 2, 1]
    if not use_dropout:
        return [0] * len(layer_structure)
    dropout_values = [0]
    dropout_values.extend([0.3] * (len(layer_structure) - 3))
    if last_layer_dropout:
        dropout_values.append(0.3)
    else:
        dropout_values.append(0)
    dropout_values.append(0)
    return dropout_values

def shorthash(self):
        sha256 = sha256(self.filename, f'hypernet/{self.name}')

        return sha256[0:10]

class Hypernetwork:
    filename = None
    name = None

    def __init__(self, name=None, enable_sizes=None, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, activate_output=False, **kwargs):
        self.filename = None
        self.name = name
        self.layers = {}
        self.step = 0
        self.sd_checkpoint = None
        self.sd_checkpoint_name = None
        self.layer_structure = layer_structure
        self.activation_func = activation_func
        self.weight_init = weight_init
        self.add_layer_norm = add_layer_norm
        self.use_dropout = use_dropout
        self.activate_output = activate_output
        self.last_layer_dropout = kwargs.get('last_layer_dropout', True)
        self.dropout_structure = kwargs.get('dropout_structure', None)
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)
        self.optimizer_name = None
        self.optimizer_state_dict = None
        self.optional_info = None

        for size in enable_sizes or []:
            self.layers[size] = (
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
                HypernetworkModule(size, None, self.layer_structure, self.activation_func, self.weight_init,
                                   self.add_layer_norm, self.activate_output, dropout_structure=self.dropout_structure),
            )
        self.eval()

    def weights(self):
        res = []
        for k, layers in self.layers.items():
            for layer in layers:
                res += layer.parameters()
        return res

    def train(self, mode=True):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.train(mode=mode)
                for param in layer.parameters():
                    param.requires_grad = mode

    def to(self, device):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.to(device)

        return self

    def set_multiplier(self, multiplier):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.multiplier = multiplier

        return self

    def eval(self):
        for k, layers in self.layers.items():
            for layer in layers:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def save(self, filename):
        state_dict = {}
        optimizer_saved_dict = {}

        for k, v in self.layers.items():
            state_dict[k] = (v[0].state_dict(), v[1].state_dict())

        state_dict['step'] = self.step
        state_dict['name'] = self.name
        state_dict['layer_structure'] = self.layer_structure
        state_dict['activation_func'] = self.activation_func
        state_dict['is_layer_norm'] = self.add_layer_norm
        state_dict['weight_initialization'] = self.weight_init
        state_dict['sd_checkpoint'] = self.sd_checkpoint
        state_dict['sd_checkpoint_name'] = self.sd_checkpoint_name
        state_dict['activate_output'] = self.activate_output
        state_dict['use_dropout'] = self.use_dropout
        state_dict['dropout_structure'] = self.dropout_structure
        state_dict['last_layer_dropout'] = (self.dropout_structure[-2] != 0) if self.dropout_structure is not None else self.last_layer_dropout
        state_dict['optional_info'] = self.optional_info if self.optional_info else None

        if self.optimizer_name is not None:
            optimizer_saved_dict['optimizer_name'] = self.optimizer_name

        torch.save(state_dict, filename)
        save_optimizer_state=True
        if save_optimizer_state and self.optimizer_state_dict:
            optimizer_saved_dict['hash'] = self.shorthash()
            optimizer_saved_dict['optimizer_state_dict'] = self.optimizer_state_dict
            torch.save(optimizer_saved_dict, filename + '.optim')

    def sha256(filename, title):
        hashes = cache("hashes")

        sha256_value = sha256_from_cache(filename, title)
        if sha256_value is not None:
            return sha256_value

        print(f"Calculating sha256 for {filename}: ", end='')
        sha256_value = calculate_sha256(filename)
        print(f"{sha256_value}")

        hashes[title] = {
            "mtime": os.path.getmtime(filename),
            "sha256": sha256_value,
        }

        dump_cache()

        return sha256_value
    def load(self, filename):
        self.filename = filename
        if self.name is None:
            self.name = os.path.splitext(os.path.basename(filename))[0]

        state_dict = torch.load(filename, map_location='cuda')

        self.layer_structure = state_dict.get('layer_structure', [1, 2, 1])
        self.optional_info = state_dict.get('optional_info', None)
        self.activation_func = state_dict.get('activation_func', None)
        self.weight_init = state_dict.get('weight_initialization', 'Normal')
        self.add_layer_norm = state_dict.get('is_layer_norm', False)
        self.dropout_structure = state_dict.get('dropout_structure', None)
        self.use_dropout = True if self.dropout_structure is not None and any(self.dropout_structure) else state_dict.get('use_dropout', False)
        self.activate_output = state_dict.get('activate_output', True)
        self.last_layer_dropout = state_dict.get('last_layer_dropout', False)
        # Dropout structure should have same length as layer structure, Every digits should be in [0,1), and last digit must be 0.
        if self.dropout_structure is None:
            self.dropout_structure = parse_dropout_structure(self.layer_structure, self.use_dropout, self.last_layer_dropout)

        print_hypernet_extra=True
        if print_hypernet_extra:
            if self.optional_info is not None:
                print(f"  INFO:\n {self.optional_info}\n")

            print(f"  Layer structure: {self.layer_structure}")
            print(f"  Activation function: {self.activation_func}")
            print(f"  Weight initialization: {self.weight_init}")
            print(f"  Layer norm: {self.add_layer_norm}")
            print(f"  Dropout usage: {self.use_dropout}" )
            print(f"  Activate last layer: {self.activate_output}")
            print(f"  Dropout structure: {self.dropout_structure}")

        optimizer_saved_dict = torch.load(self.filename + '.optim', map_location='cuda') if os.path.exists(self.filename + '.optim') else {}

        if self.shorthash() == optimizer_saved_dict.get('hash', None):
            self.optimizer_state_dict = optimizer_saved_dict.get('optimizer_state_dict', None)
        else:
            self.optimizer_state_dict = None
        if self.optimizer_state_dict:
            self.optimizer_name = optimizer_saved_dict.get('optimizer_name', 'AdamW')
            if print_hypernet_extra:
                print("Loaded existing optimizer from checkpoint")
                print(f"Optimizer name is {self.optimizer_name}")
        else:
            self.optimizer_name = "AdamW"
            if print_hypernet_extra:
                print("No saved optimizer exists in checkpoint")

        for size, sd in state_dict.items():
            if type(size) == int:
                self.layers[size] = (
                    HypernetworkModule(size, sd[0], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                    HypernetworkModule(size, sd[1], self.layer_structure, self.activation_func, self.weight_init,
                                       self.add_layer_norm, self.activate_output, self.dropout_structure),
                )

        self.name = state_dict.get('name', self.name)
        self.step = state_dict.get('step', 0)
        self.sd_checkpoint = state_dict.get('sd_checkpoint', None)
        self.sd_checkpoint_name = state_dict.get('sd_checkpoint_name', None)
        self.eval()

    import hashlib

    def shorthash(self):
        sha256 = hashlib.sha256()
        sha256.update(f'{self.filename}hypernet/{self.name}'.encode())
        print(f"sha256= ", sha256.hexdigest()[0:10])
        return sha256.hexdigest()[0:10]


def list_hypernetworks(path):
    res = {}
    for filename in sorted(glob.iglob(os.path.join(path, '**/*.pt'), recursive=True)):
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name] = filename
    return res


def load_hypernetwork(name):
    path = hypernetworks.get(name, None)

    if path is None:
        return None

    hypernetwork = Hypernetwork()

    try:
        hypernetwork.load(path)
    except Exception:
        print(f"Error loading hypernetwork {path}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return None

    return hypernetwork


def load_hypernetworks(names, multipliers=None):
    already_loaded = {}

    for hypernetwork in loaded_hypernetworks:
        if hypernetwork.name in names:
            already_loaded[hypernetwork.name] = hypernetwork

    loaded_hypernetworks.clear()

    for i, name in enumerate(names):
        hypernetwork = already_loaded.get(name, None)
        if hypernetwork is None:
            hypernetwork = load_hypernetwork(name)

        if hypernetwork is None:
            continue

        hypernetwork.set_multiplier(multipliers[i] if multipliers else 1.0)
        loaded_hypernetworks.append(hypernetwork)
    print(f"Hypernetworks Loaded", names)

def find_closest_hypernetwork_name(search: str):
    if not search:
        return None
    search = search.lower()
    applicable = [name for name in hypernetworks if search in name.lower()]
    if not applicable:
        return None
    applicable = sorted(applicable, key=lambda name: len(name))
    return applicable[0]


def apply_single_hypernetwork(hypernetwork, context_k, context_v, layer=None):
    hypernetwork_layers = (hypernetwork.layers if hypernetwork is not None else {}).get(context_k.shape[2], None)

    if hypernetwork_layers is None:
        return context_k, context_v

    if layer is not None:
        layer.hyper_k = hypernetwork_layers[0]
        layer.hyper_v = hypernetwork_layers[1]

    context_k = hypernetwork_layers[0](context_k)
    context_v = hypernetwork_layers[1](context_v)
    return context_k, context_v


def apply_hypernetworks(hypernetworks, context, layer=None):
    context_k = context
    context_v = context
    for hypernetwork in hypernetworks:
        context_k, context_v = apply_single_hypernetwork(hypernetwork, context_k, context_v, layer)

    return context_k, context_v


def attention_CrossAttention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetworks(loaded_hypernetworks, context, self)
    k = self.to_k(context_k)
    v = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    attn = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', attn, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def stack_conds(conds):
    if len(conds) == 1:
        return torch.stack(conds)

    # same as in reconstruct_multicond_batch
    token_count = max([x.shape[0] for x in conds])
    for i in range(len(conds)):
        if conds[i].shape[0] != token_count:
            last_vector = conds[i][-1:]
            last_vector_repeated = last_vector.repeat([token_count - conds[i].shape[0], 1])
            conds[i] = torch.vstack([conds[i], last_vector_repeated])

    return torch.stack(conds)


def statistics(data):
    if len(data) < 2:
        std = 0
    else:
        std = stdev(data)
    total_information = f"loss:{mean(data):.3f}" + u"\u00B1" + f"({std/ (len(data) ** 0.5):.3f})"
    recent_data = data[-32:]
    if len(recent_data) < 2:
        std = 0
    else:
        std = stdev(recent_data)
    recent_information = f"recent 32 loss:{mean(recent_data):.3f}" + u"\u00B1" + f"({std / (len(recent_data) ** 0.5):.3f})"
    return total_information, recent_information


def report_statistics(loss_info:dict):
    keys = sorted(loss_info.keys(), key=lambda x: sum(loss_info[x]) / len(loss_info[x]))
    for key in keys:
        try:
            print("Loss statistics for file " + key)
            info, recent = statistics(list(loss_info[key]))
            print(info)
            print(recent)
        except Exception as e:
            print(e)

def create_hypernetwork(name, enable_sizes, overwrite_old, layer_structure=None, activation_func=None, weight_init=None, add_layer_norm=False, use_dropout=False, dropout_structure=None):
    # Remove illegal characters from name.
    name = "".join( x for x in name if (x.isalnum() or x in "._- "))
    assert name, "Name cannot be empty!"

    fn = os.path.join(hypernetwork_dir, f"{name}.pt")
    if not overwrite_old:
        assert not os.path.exists(fn), f"file {fn} already exists"

    if type(layer_structure) == str:
        layer_structure = [float(x.strip()) for x in layer_structure.split(",")]

    if use_dropout and dropout_structure and type(dropout_structure) == str:
        dropout_structure = [float(x.strip()) for x in dropout_structure.split(",")]
    else:
        dropout_structure = [0] * len(layer_structure)

    hypernet = Hypernetwork(
        name=name,
        enable_sizes=[int(x) for x in enable_sizes],
        layer_structure=layer_structure,
        activation_func=activation_func,
        weight_init=weight_init,
        add_layer_norm=add_layer_norm,
        use_dropout=use_dropout,
        dropout_structure=dropout_structure
    )
    hypernet.save(fn)

    reload_hypernetworks()

#apply_optimizations
hypernetwork=Hypernetwork()
import math
import sys
import traceback
import psutil
import contextlib
import torch
from torch import einsum

from ldm.util import default
from einops import rearrange

if use_xformers:
    try:
        import xformers.ops
        xformers_available = True
    except Exception:
        print("Cannot import xformers", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


def get_available_vram():
    if device == 'cuda':
        stats = torch.cuda.memory_stats(device)
        mem_active = stats['active_bytes.all.current']
        mem_reserved = stats['reserved_bytes.all.current']
        mem_free_cuda, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
        mem_free_torch = mem_reserved - mem_active
        mem_free_total = mem_free_cuda + mem_free_torch
        return mem_free_total
    else:
        return psutil.virtual_memory().available

# see https://github.com/basujindal/stable-diffusion/pull/117 for discussion
def split_cross_attention_forward_v1(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetworks(loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)
    del context, context_k, context_v, x

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    upcast_attn = opts_upcast_attn
    if upcast_attn:
        q, k, v = q.float(), k.float(), v.float()

    upcast_attn = opts_upcast_attn
    with without_autocast(disable=not upcast_attn):
        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
        for i in range(0, q.shape[0], 2):
            end = i + 2
            s1 = einsum('b i d, b j d -> b i j', q[i:end], k[i:end])
            s1 *= self.scale

            s2 = s1.softmax(dim=-1)
            del s1

            r1[i:end] = einsum('b i j, b j d -> b i d', s2, v[i:end])
            del s2
        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# taken from https://github.com/Doggettx/stable-diffusion and modified
def split_cross_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetworks(loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    dtype = q_in.dtype
    upcast_attn = opts_upcast_attn
    if upcast_attn:
        q_in, k_in, v_in = q_in.float(), k_in.float(), v_in if v_in.device.type == 'mps' else v_in.float()
    upcast_attn = opts_upcast_attn
    with without_autocast(disable=not upcast_attn):
        k_in = k_in * self.scale

        del context, x

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        r1 = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)

        mem_free_total = get_available_vram()

        gb = 1024 ** 3
        tensor_size = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size()
        modifier = 3 if q.element_size() == 2 else 2.5
        mem_required = tensor_size * modifier
        steps = 1

        if mem_required > mem_free_total:
            steps = 2 ** (math.ceil(math.log(mem_required / mem_free_total, 2)))
            # print(f"Expected tensor size:{tensor_size/gb:0.1f}GB, cuda free:{mem_free_cuda/gb:0.1f}GB "
            #       f"torch free:{mem_free_torch/gb:0.1f} total:{mem_free_total/gb:0.1f} steps:{steps}")

        if steps > 64:
            max_res = math.floor(math.sqrt(math.sqrt(mem_free_total / 2.5)) / 8) * 64
            raise RuntimeError(f'Not enough memory, use lower resolution (max approx. {max_res}x{max_res}). '
                               f'Need: {mem_required / 64 / gb:0.1f}GB free, Have:{mem_free_total / gb:0.1f}GB free')

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size
            s1 = einsum('b i d, b j d -> b i j', q[:, i:end], k)

            s2 = s1.softmax(dim=-1, dtype=q.dtype)
            del s1

            r1[:, i:end] = einsum('b i j, b j d -> b i d', s2, v)
            del s2

        del q, k, v

    r1 = r1.to(dtype)

    r2 = rearrange(r1, '(b h) n d -> b n (h d)', h=h)
    del r1

    return self.to_out(r2)


# -- Taken from https://github.com/invoke-ai/InvokeAI and modified --
mem_total_gb = psutil.virtual_memory().total // (1 << 30)

def einsum_op_compvis(q, k, v):
    s = einsum('b i d, b j d -> b i j', q, k)
    s = s.softmax(dim=-1, dtype=s.dtype)
    return einsum('b i j, b j d -> b i d', s, v)

def einsum_op_slice_0(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[0], slice_size):
        end = i + slice_size
        r[i:end] = einsum_op_compvis(q[i:end], k[i:end], v[i:end])
    return r

def einsum_op_slice_1(q, k, v, slice_size):
    r = torch.zeros(q.shape[0], q.shape[1], v.shape[2], device=q.device, dtype=q.dtype)
    for i in range(0, q.shape[1], slice_size):
        end = i + slice_size
        r[:, i:end] = einsum_op_compvis(q[:, i:end], k, v)
    return r

def einsum_op_mps_v1(q, k, v):
    if q.shape[0] * q.shape[1] <= 2**16: # (512x512) max q.shape[1]: 4096
        return einsum_op_compvis(q, k, v)
    else:
        slice_size = math.floor(2**30 / (q.shape[0] * q.shape[1]))
        if slice_size % 4096 == 0:
            slice_size -= 1
        return einsum_op_slice_1(q, k, v, slice_size)

def einsum_op_mps_v2(q, k, v):
    if mem_total_gb > 8 and q.shape[0] * q.shape[1] <= 2**16:
        return einsum_op_compvis(q, k, v)
    else:
        return einsum_op_slice_0(q, k, v, 1)

def einsum_op_tensor_mem(q, k, v, max_tensor_mb):
    size_mb = q.shape[0] * q.shape[1] * k.shape[1] * q.element_size() // (1 << 20)
    if size_mb <= max_tensor_mb:
        return einsum_op_compvis(q, k, v)
    div = 1 << int((size_mb - 1) / max_tensor_mb).bit_length()
    if div <= q.shape[0]:
        return einsum_op_slice_0(q, k, v, q.shape[0] // div)
    return einsum_op_slice_1(q, k, v, max(q.shape[1] // div, 1))

def einsum_op_cuda(q, k, v):
    stats = torch.cuda.memory_stats(q.device)
    mem_active = stats['active_bytes.all.current']
    mem_reserved = stats['reserved_bytes.all.current']
    mem_free_cuda, _ = torch.cuda.mem_get_info(q.device)
    mem_free_torch = mem_reserved - mem_active
    mem_free_total = mem_free_cuda + mem_free_torch
    # Divide factor of safety as there's copying and fragmentation
    return einsum_op_tensor_mem(q, k, v, mem_free_total / 3.3 / (1 << 20))

def einsum_op(q, k, v):
    if q.device.type == 'cuda':
        return einsum_op_cuda(q, k, v)

    if q.device.type == 'mps':
        if mem_total_gb >= 32 and q.shape[0] % 32 != 0 and q.shape[0] * q.shape[1] < 2**18:
            return einsum_op_mps_v1(q, k, v)
        return einsum_op_mps_v2(q, k, v)

    # Smaller slices are faster due to L2/L3/SLC caches.
    # Tested on i7 with 8MB L3 cache.
    return einsum_op_tensor_mem(q, k, v, 32)

def split_cross_attention_forward_invokeAI(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    dtype = q.dtype
    upcast_attn = opts_upcast_attn
    if upcast_attn:
        q, k, v = q.float(), k.float(), v if v.device.type == 'mps' else v.float()

    with without_autocast(disable=not upcast_attn):
        k = k * self.scale

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        r = einsum_op(q, k, v)
    r = r.to(dtype)
    return self.to_out(rearrange(r, '(b h) n d -> b n (h d)', h=h))

# -- End of code from https://github.com/invoke-ai/InvokeAI --


# Based on Birch-san's modified implementation of sub-quadratic attention from https://github.com/Birch-san/diffusers/pull/1
# The sub_quad_attention_forward function is under the MIT License listed under Memory Efficient Attention in the Licenses section of the web UI interface
def sub_quad_attention_forward(self, x, context=None, mask=None):
    assert mask is None, "attention-mask not currently implemented for SubQuadraticCrossAttnProcessor."

    h = self.heads

    q = self.to_q(x)
    context = default(context, x)

    context_k, context_v = hypernetwork.apply_hypernetworks(loaded_hypernetworks, context)
    k = self.to_k(context_k)
    v = self.to_v(context_v)
    del context, context_k, context_v, x

    q = q.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    k = k.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)
    v = v.unflatten(-1, (h, -1)).transpose(1,2).flatten(end_dim=1)

    dtype = q.dtype
    upcast_attn = opts_upcast_attn
    if upcast_attn:
        q, k = q.float(), k.float()
    sub_quad_q_chunk_size=1024
    x = sub_quad_attention(q, k, v, q_chunk_size=sub_quad_q_chunk_size, kv_chunk_size=sub_quad_kv_chunk_size, chunk_threshold=shared.cmd_opts.sub_quad_chunk_threshold, use_checkpoint=self.training)

    x = x.to(dtype)

    x = x.unflatten(0, (-1, h)).transpose(1,2).flatten(start_dim=2)

    out_proj, dropout = self.to_out
    x = out_proj(x)
    x = dropout(x)

    return x

def sub_quad_attention(q, k, v, q_chunk_size=1024, kv_chunk_size=None, kv_chunk_size_min=None, chunk_threshold=None, use_checkpoint=True):
    bytes_per_token = torch.finfo(q.dtype).bits//8
    batch_x_heads, q_tokens, _ = q.shape
    _, k_tokens, _ = k.shape
    qk_matmul_size_bytes = batch_x_heads * bytes_per_token * q_tokens * k_tokens

    if chunk_threshold is None:
        chunk_threshold_bytes = int(get_available_vram() * 0.9) if q.device.type == 'mps' else int(get_available_vram() * 0.7)
    elif chunk_threshold == 0:
        chunk_threshold_bytes = None
    else:
        chunk_threshold_bytes = int(0.01 * chunk_threshold * get_available_vram())

    if kv_chunk_size_min is None and chunk_threshold_bytes is not None:
        kv_chunk_size_min = chunk_threshold_bytes // (batch_x_heads * bytes_per_token * (k.shape[2] + v.shape[2]))
    elif kv_chunk_size_min == 0:
        kv_chunk_size_min = None

    if chunk_threshold_bytes is not None and qk_matmul_size_bytes <= chunk_threshold_bytes:
        # the big matmul fits into our memory limit; do everything in 1 chunk,
        # i.e. send it down the unchunked fast-path
        query_chunk_size = q_tokens
        kv_chunk_size = k_tokens

    with without_autocast(disable=q.dtype == v.dtype):
        return efficient_dot_product_attention(
            q,
            k,
            v,
            query_chunk_size=q_chunk_size,
            kv_chunk_size=kv_chunk_size,
            kv_chunk_size_min = kv_chunk_size_min,
            use_checkpoint=use_checkpoint,
        )

xformers_flash_attention=True
def get_xformers_flash_attention_op(q, k, v):
    if not xformers_flash_attention:
        return None

    try:
        flash_attention_op = xformers.ops.MemoryEfficientAttentionFlashAttentionOp
        fw, bw = flash_attention_op
        if fw.supports(xformers.ops.fmha.Inputs(query=q, key=k, value=v, attn_bias=None)):
            return flash_attention_op
    except Exception as e:
        display_once(e, "enabling flash attention")

    return None


def xformers_attention_forward(self, x, context=None, mask=None):
    h = self.heads
    q_in = self.to_q(x)
    context = default(context, x)

    context_k, context_v = apply_hypernetworks(loaded_hypernetworks, context)
    k_in = self.to_k(context_k)
    v_in = self.to_v(context_v)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=h), (q_in, k_in, v_in))
    del q_in, k_in, v_in

    dtype = q.dtype
    upcast_attn = opts_upcast_attn
    if upcast_attn:
        q, k = q.float(), k.float()

    out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=get_xformers_flash_attention_op(q, k, v))

    out = out.to(dtype)

    out = rearrange(out, 'b n h d -> b n (h d)', h=h)
    return self.to_out(out)

def cross_attention_attnblock_forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q1 = self.q(h_)
        k1 = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q1.shape

        q2 = q1.reshape(b, c, h*w)
        del q1

        q = q2.permute(0, 2, 1)   # b,hw,c
        del q2

        k = k1.reshape(b, c, h*w) # b,c,hw
        del k1

        h_ = torch.zeros_like(k, device=q.device)

        mem_free_total = get_available_vram()

        tensor_size = q.shape[0] * q.shape[1] * k.shape[2] * q.element_size()
        mem_required = tensor_size * 2.5
        steps = 1

        if mem_required > mem_free_total:
            steps = 2**(math.ceil(math.log(mem_required / mem_free_total, 2)))

        slice_size = q.shape[1] // steps if (q.shape[1] % steps) == 0 else q.shape[1]
        for i in range(0, q.shape[1], slice_size):
            end = i + slice_size

            w1 = torch.bmm(q[:, i:end], k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
            w2 = w1 * (int(c)**(-0.5))
            del w1
            w3 = torch.nn.functional.softmax(w2, dim=2, dtype=q.dtype)
            del w2

            # attend to values
            v1 = v.reshape(b, c, h*w)
            w4 = w3.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
            del w3

            h_[:, :, i:end] = torch.bmm(v1, w4)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
            del v1, w4

        h2 = h_.reshape(b, c, h, w)
        del h_

        h3 = self.proj_out(h2)
        del h2

        h3 += x

        return h3

def xformers_attnblock_forward(self, x):
    try:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)
        b, c, h, w = q.shape
        q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
        dtype = q.dtype
        upcast_attn = opts_upcast_attn
        if upcast_attn:
            q, k = q.float(), k.float()
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, op=get_xformers_flash_attention_op(q, k, v))
        out = out.to(dtype)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h)
        out = self.proj_out(out)
        return x + out
    except NotImplementedError:
        return cross_attention_attnblock_forward(self, x)

def sub_quad_attnblock_forward(self, x):
    h_ = x
    h_ = self.norm(h_)
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)
    b, c, h, w = q.shape
    q, k, v = map(lambda t: rearrange(t, 'b c h w -> b (h w) c'), (q, k, v))
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    sub_quad_q_chunk_size=1024
    sub_quad_kv_chunk_size=None
    sub_quad_chunk_threshold=None
    out = sub_quad_attention(q, k, v, q_chunk_size=sub_quad_q_chunk_size, kv_chunk_size=sub_quad_kv_chunk_size, chunk_threshold=sub_quad_chunk_threshold, use_checkpoint=self.training)
    out = rearrange(out, 'b (h w) c -> b c h w', h=h)
    out = self.proj_out(out)
    return x + out

#sd_hijack_unet
import torch
from packaging import version

def autocast(disable=False):

    if disable:
        return contextlib.nullcontext()

    if dtype == torch.float32 or precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")

class TorchHijackForUnet:
    """
    This is torch, but with cat that resizes tensors to appropriate dimensions if they do not match;
    this makes it possible to create pictures with dimensions that are multiples of 8 rather than 64
    """

    def __getattr__(self, item):
        if item == 'cat':
            return self.cat

        if hasattr(torch, item):
            return getattr(torch, item)

        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))

    def cat(self, tensors, *args, **kwargs):
        if len(tensors) == 2:
            a, b = tensors
            if a.shape[-2:] != b.shape[-2:]:
                a = torch.nn.functional.interpolate(a, b.shape[-2:], mode="nearest")

            tensors = (a, b)

        return torch.cat(tensors, *args, **kwargs)


th = TorchHijackForUnet()


# Below are monkey patches to enable upcasting a float16 UNet for float32 sampling
def apply_model(orig_func, self, x_noisy, t, cond, **kwargs):

    if isinstance(cond, dict):
        for y in cond.keys():
            cond[y] = [x.to(dtype_unet) if isinstance(x, torch.Tensor) else x for x in cond[y]]

    with t_autocast:
        return orig_func(self, x_noisy.to(dtype_unet), t.to(dtype_unet), cond, **kwargs)
# unet_needs_upcast=False
class GELUHijack(torch.nn.GELU, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.GELU.__init__(self, *args, **kwargs)
    def forward(self, x):
        if unet_needs_upcast:
            return torch.nn.GELU.forward(self.float(), x.float()).to(dtype_unet)
        else:
            return torch.nn.GELU.forward(self, x)

ddpm_edit_hijack = None
def hijack_ddpm_edit():
    global ddpm_edit_hijack
    if not ddpm_edit_hijack:
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
        CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
        ddpm_edit_hijack = CondFunc('modules.models.diffusion.ddpm_edit.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)

unet_needs_upcast = lambda *args, **kwargs: unet_needs_upcast
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.apply_model', apply_model, unet_needs_upcast)
CondFunc('ldm.modules.diffusionmodules.openaimodel.timestep_embedding', lambda orig_func, timesteps, *args, **kwargs: orig_func(timesteps, *args, **kwargs).to(torch.float32 if timesteps.dtype == torch.int64 else dtype_unet), unet_needs_upcast)
if version.parse(torch.__version__) <= version.parse("1.13.1"):
    CondFunc('ldm.modules.diffusionmodules.util.GroupNorm32.forward', lambda orig_func, self, *args, **kwargs: orig_func(self.float(), *args, **kwargs), unet_needs_upcast)
    CondFunc('ldm.modules.attention.GEGLU.forward', lambda orig_func, self, x: orig_func(self.float(), x.float()).to(dtype_unet), unet_needs_upcast)
    CondFunc('open_clip.transformer.ResidualAttentionBlock.__init__', lambda orig_func, *args, **kwargs: kwargs.update({'act_layer': GELUHijack}) and False or orig_func(*args, **kwargs), lambda _, *args, **kwargs: kwargs.get('act_layer') is None or kwargs['act_layer'] == torch.nn.GELU)

first_stage_cond = lambda _, self, *args, **kwargs: unet_needs_upcast and self.model.diffusion_model.dtype == torch.float16
first_stage_sub = lambda orig_func, self, x, **kwargs: orig_func(self, x.to(dtype_vae), **kwargs)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.decode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.encode_first_stage', first_stage_sub, first_stage_cond)
CondFunc('ldm.models.diffusion.ddpm.LatentDiffusion.get_first_stage_encoding', lambda orig_func, *args, **kwargs: orig_func(*args, **kwargs).float(), first_stage_cond)

#@title sd_hijack
import torch
from torch.nn.functional import silu
from types import MethodType
import ldm.modules.attention
import ldm.modules.diffusionmodules.model
import ldm.modules.diffusionmodules.openaimodel
import ldm.models.diffusion.ddim
import ldm.models.diffusion.plms
import ldm.modules.encoders.modules

attention_CrossAttention_forward = ldm.modules.attention.CrossAttention.forward
diffusionmodules_model_nonlinearity = ldm.modules.diffusionmodules.model.nonlinearity
diffusionmodules_model_AttnBlock_forward = ldm.modules.diffusionmodules.model.AttnBlock.forward

# new memory efficient cross attention blocks do not support hypernets and we already
# have memory efficient cross attention anyway, so this disables SD2.0's memory efficient cross attention
ldm.modules.attention.MemoryEfficientCrossAttention = ldm.modules.attention.CrossAttention
ldm.modules.attention.BasicTransformerBlock.ATTENTION_MODES["softmax-xformers"] = ldm.modules.attention.CrossAttention

# silence new console spam from SD2
ldm.modules.attention.print = lambda *args: None
ldm.modules.diffusionmodules.model.print = lambda *args: None
th = TorchHijackForUnet()
def apply_optimizations():
    undo_optimizations()

    ldm.modules.diffusionmodules.model.nonlinearity = silu
    ldm.modules.diffusionmodules.openaimodel.th = th

    optimization_method = None

    if use_xformers and torch.version.cuda and (6, 0) <= torch.cuda.get_device_capability(device) <= (9, 0):
        print("Applying xformers cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = xformers_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = xformers_attnblock_forward
        optimization_method = 'xformers'
    elif use_sub_quad_attention and not use_xformers:
        print("Applying sub-quadratic cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = sub_quad_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = sub_quad_attnblock_forward
        optimization_method = 'sub-quadratic'
    elif use_split_attention_v1 and not use_sub_quad_attention and not use_xformers:
        print("Applying v1 cross attention optimization.")
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_v1
        optimization_method = 'V1'
    elif use_split_cross_attention_forward_invokeAI and not torch.cuda.is_available():
        print("Applying cross attention optimization (InvokeAI).")
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward_invokeAI
        optimization_method = 'InvokeAI'
    elif not disable_opt_split_attention and (opt_split_attention or torch.cuda.is_available()):
        print("Applying cross attention optimization (Doggettx).")
        ldm.modules.attention.CrossAttention.forward = split_cross_attention_forward
        ldm.modules.diffusionmodules.model.AttnBlock.forward = cross_attention_attnblock_forward
        optimization_method = 'Doggettx'

    return optimization_method


def undo_optimizations():
    ldm.modules.attention.CrossAttention.forward = attention_CrossAttention_forward
    ldm.modules.diffusionmodules.model.nonlinearity = diffusionmodules_model_nonlinearity
    ldm.modules.diffusionmodules.model.AttnBlock.forward = diffusionmodules_model_AttnBlock_forward


def fix_checkpoint():
    """checkpoints are now added and removed in embedding/hypernet code, since torch doesn't want
    checkpoints to be added when not training (there's a warning)"""

    pass

def weighted_loss(model, pred, target, mean=True):
    #Calculate the weight normally, but ignore the mean
    loss = model._old_get_loss(pred, target, mean=False)

    #Check if we have weights available
    weight = getattr(model, '_custom_loss_weight', None)
    if weight is not None:
        loss *= weight

    #Return the loss, as mean if specified
    return loss.mean() if mean else loss

def weighted_forward(model, x, c, w, *args, **kwargs):
    try:
        #Temporarily append weights to a place accessible during loss calc
        model._custom_loss_weight = w

        #Replace 'get_loss' with a weight-aware one. Otherwise we need to reimplement 'forward' completely
        #Keep 'get_loss', but don't overwrite the previous old_get_loss if it's already set
        if not hasattr(model, '_old_get_loss'):
            model._old_get_loss = model.get_loss
        model.get_loss = MethodType(weighted_loss, model)

        #Run the standard forward function, but with the patched 'get_loss'
        return model.forward(x, c, *args, **kwargs)
    finally:
        try:
            #Delete temporary weights if appended
            del model._custom_loss_weight
        except AttributeError as e:
            pass

        #If we have an old loss function, reset the loss function to the original one
        if hasattr(model, '_old_get_loss'):
            model.get_loss = model._old_get_loss
            del model._old_get_loss

def apply_weighted_forward(model):
    #Add new function 'weighted_forward' that can be called to calc weighted loss
    model.weighted_forward = MethodType(weighted_forward, model)

def undo_weighted_forward(model):
    try:
        del model.weighted_forward
    except AttributeError as e:
        pass

from transformers import BertPreTrainedModel,BertModel,BertConfig
import torch.nn as nn
import torch
from transformers.models.xlm_roberta.configuration_xlm_roberta import XLMRobertaConfig
from transformers import XLMRobertaModel,XLMRobertaTokenizer
from typing import Optional

class BertSeriesConfig(BertConfig):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, hidden_act="gelu", hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512, type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12, pad_token_id=0, position_embedding_type="absolute", use_cache=True, classifier_dropout=None,project_dim=512, pooler_fn="average",learn_encoder=False,model_type='bert',**kwargs):

        super().__init__(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, max_position_embeddings, type_vocab_size, initializer_range, layer_norm_eps, pad_token_id, position_embedding_type, use_cache, classifier_dropout, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder

class RobertaSeriesConfig(XLMRobertaConfig):
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2,project_dim=512,pooler_fn='cls',learn_encoder=False, **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.project_dim = project_dim
        self.pooler_fn = pooler_fn
        self.learn_encoder = learn_encoder


class BertSeriesModelWithTransformation(BertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    config_class = BertSeriesConfig

    def __init__(self, config=None, **kargs):
        # modify initialization for autoloading
        if config is None:
            config = XLMRobertaConfig()
            config.attention_probs_dropout_prob= 0.1
            config.bos_token_id=0
            config.eos_token_id=2
            config.hidden_act='gelu'
            config.hidden_dropout_prob=0.1
            config.hidden_size=1024
            config.initializer_range=0.02
            config.intermediate_size=4096
            config.layer_norm_eps=1e-05
            config.max_position_embeddings=514

            config.num_attention_heads=16
            config.num_hidden_layers=24
            config.output_past=True
            config.pad_token_id=1
            config.position_embedding_type= "absolute"

            config.type_vocab_size= 1
            config.use_cache=True
            config.vocab_size= 250002
            config.project_dim = 768
            config.learn_encoder = False
        super().__init__(config)
        self.roberta = XLMRobertaModel(config)
        self.transformation = nn.Linear(config.hidden_size,config.project_dim)
        self.pre_LN=nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')
        self.pooler = lambda x: x[:,0]
        self.post_init()

    def encode(self,c):
        device = next(self.parameters()).device
        text = self.tokenizer(c,
                        truncation=True,
                        max_length=77,
                        return_length=False,
                        return_overflowing_tokens=False,
                        padding="max_length",
                        return_tensors="pt")
        text["input_ids"] = torch.tensor(text["input_ids"]).to(device)
        text["attention_mask"] = torch.tensor(
            text['attention_mask']).to(device)
        features = self(**text)
        return features['projection_state']

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) :
        r"""
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        # last module outputs
        sequence_output = outputs[0]


        # project every module
        sequence_output_ln = self.pre_LN(sequence_output)

        # pooler
        pooler_output = self.pooler(sequence_output_ln)
        pooler_output = self.transformation(pooler_output)
        projection_state = self.transformation(outputs.last_hidden_state)

        return {
            'pooler_output':pooler_output,
            'last_hidden_state':outputs.last_hidden_state,
            'hidden_states':outputs.hidden_states,
            'attentions':outputs.attentions,
            'projection_state':projection_state,
            'sequence_out': sequence_output
        }


class RobertaSeriesModelWithTransformation(BertSeriesModelWithTransformation):
    base_model_prefix = 'roberta'
    config_class= RobertaSeriesConfig

class StableDiffusionModelHijack:
    fixes = None
    comments = []
    layers = None
    circular_enabled = False
    clip = None
    optimization_method = None

    embedding_db = EmbeddingDatabase()

    def __init__(self):
        self.embedding_db.add_embedding_dir(embeddings_dir)

    def hijack(self, m):

        # if type(m.cond_stage_model) == BertSeriesModelWithTransformation:
        #     model_embeddings = m.cond_stage_model.roberta.embeddings
        #     model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.word_embeddings, self)
        #     m.cond_stage_model = FrozenXLMREmbedderWithCustomWords(m.cond_stage_model, self)

        if type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenCLIPEmbedder:
            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            model_embeddings.token_embedding = EmbeddingsWithFixes(model_embeddings.token_embedding, self)
            m.cond_stage_model = FrozenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        elif type(m.cond_stage_model) == ldm.modules.encoders.modules.FrozenOpenCLIPEmbedder:
            m.cond_stage_model.model.token_embedding = EmbeddingsWithFixes(m.cond_stage_model.model.token_embedding, self)
            m.cond_stage_model = FrozenOpenCLIPEmbedderWithCustomWords(m.cond_stage_model, self)

        apply_weighted_forward(m)
        if m.cond_stage_key == "edit":
            hijack_ddpm_edit()

        optimization_method = apply_optimizations()

        self.clip = m.cond_stage_model

        def flatten(el):
            flattened = [flatten(children) for children in el.children()]
            res = [el]
            for c in flattened:
                res += c
            return res

        self.layers = flatten(m)

    def undo_hijack(self, m):

        if type(m.cond_stage_model) == FrozenCLIPEmbedderWithCustomWords:
            m.cond_stage_model = m.cond_stage_model.wrapped

            model_embeddings = m.cond_stage_model.transformer.text_model.embeddings
            if type(model_embeddings.token_embedding) == EmbeddingsWithFixes:
                model_embeddings.token_embedding = model_embeddings.token_embedding.wrapped
        elif type(m.cond_stage_model) == FrozenOpenCLIPEmbedderWithCustomWords:
            m.cond_stage_model.wrapped.model.token_embedding = m.cond_stage_model.wrapped.model.token_embedding.wrapped
            m.cond_stage_model = m.cond_stage_model.wrapped

        self.apply_circular(False)
        self.layers = None
        self.clip = None

    def apply_circular(self, enable):
        if self.circular_enabled == enable:
            return

        self.circular_enabled = enable

        for layer in [layer for layer in self.layers if type(layer) == torch.nn.Conv2d]:
            layer.padding_mode = 'circular' if enable else 'zeros'

    def clear_comments(self):
        self.comments = []

    def get_prompt_lengths(self, text):
        _, token_count = self.clip.process_texts([text])

        return token_count, self.clip.get_target_prompt_token_count(token_count)

def cond_cast_unet(input):
    return input.to(dtype_unet) if unet_needs_upcast else input

class EmbeddingsWithFixes(torch.nn.Module):
    def __init__(self, wrapped, embeddings):
        super().__init__()
        self.wrapped = wrapped
        self.embeddings = embeddings

    def forward(self, input_ids):
        batch_fixes = self.embeddings.fixes
        self.embeddings.fixes = None

        inputs_embeds = self.wrapped(input_ids)

        if batch_fixes is None or len(batch_fixes) == 0 or max([len(x) for x in batch_fixes]) == 0:
            return inputs_embeds

        vecs = []
        for fixes, tensor in zip(batch_fixes, inputs_embeds):
            for offset, embedding in fixes:
                emb = cond_cast_unet(embedding.vec)
                emb_len = min(tensor.shape[0] - offset - 1, emb.shape[0])
                tensor = torch.cat([tensor[0:offset + 1], emb[0:emb_len], tensor[offset + 1 + emb_len:]])

            vecs.append(tensor)

        return torch.stack(vecs)


def add_circular_option_to_conv_2d():
    conv2d_constructor = torch.nn.Conv2d.__init__

    def conv2d_constructor_circular(self, *args, **kwargs):
        return conv2d_constructor(self, *args, padding_mode='circular', **kwargs)

    torch.nn.Conv2d.__init__ = conv2d_constructor_circular


model_hijack = StableDiffusionModelHijack()


def register_buffer(self, name, attr):
    """
    Fix register buffer bug for Mac OS.
    """

    if type(attr) == torch.Tensor:
        if attr.device != device:
            attr = attr.to(device=device, dtype=(torch.float32 if device.type == 'mps' else None))

    setattr(self, name, attr)


ldm.models.diffusion.ddim.DDIMSampler.register_buffer = register_buffer
ldm.models.diffusion.plms.PLMSSampler.register_buffer = register_buffer

#@title hijack_clip

import math
from collections import namedtuple

import torch

class PromptChunk:
    """
    This object contains token ids, weight (multipliers:1.4) and textual inversion embedding info for a chunk of prompt.
    If a prompt is short, it is represented by one PromptChunk, otherwise, multiple are necessary.
    Each PromptChunk contains an exact amount of tokens - 77, which includes one for start and end token,
    so just 75 tokens from prompt.
    """

    def __init__(self):
        self.tokens = []
        self.multipliers = []
        self.fixes = []


PromptChunkFix = namedtuple('PromptChunkFix', ['offset', 'embedding'])
"""An object of this type is a marker showing that textual inversion embedding's vectors have to placed at offset in the prompt
chunk. Thos objects are found in PromptChunk.fixes and, are placed into FrozenCLIPEmbedderWithCustomWordsBase.hijack.fixes, and finally
are applied by sd_hijack.EmbeddingsWithFixes's forward function."""


class FrozenCLIPEmbedderWithCustomWordsBase(torch.nn.Module):
    """A pytorch module that is a wrapper for FrozenCLIPEmbedder module. it enhances FrozenCLIPEmbedder, making it possible to
    have unlimited prompt length and assign weights to tokens in prompt.
    """

    def __init__(self, wrapped, hijack):
        super().__init__()

        self.wrapped = wrapped
        """Original FrozenCLIPEmbedder module; can also be FrozenOpenCLIPEmbedder or xlmr.BertSeriesModelWithTransformation,
        depending on model."""

        self.hijack: StableDiffusionModelHijack = hijack
        self.chunk_length = 75

    def empty_chunk(self):
        """creates an empty PromptChunk and returns it"""

        chunk = PromptChunk()
        chunk.tokens = [self.id_start] + [self.id_end] * (self.chunk_length + 1)
        chunk.multipliers = [1.0] * (self.chunk_length + 2)
        return chunk

    def get_target_prompt_token_count(self, token_count):
        """returns the maximum number of tokens a prompt of a known length can have before it requires one more PromptChunk to be represented"""

        return math.ceil(max(token_count, 1) / self.chunk_length) * self.chunk_length

    def tokenize(self, texts):
        """Converts a batch of texts into a batch of token ids"""

        raise NotImplementedError

    def encode_with_transformers(self, tokens):
        """
        converts a batch of token ids (in python lists) into a single tensor with numeric respresentation of those tokens;
        All python lists with tokens are assumed to have same length, usually 77.
        if input is a list with B elements and each element has T tokens, expected output shape is (B, T, C), where C depends on
        model - can be 768 and 1024.
        Among other things, this call will read self.hijack.fixes, apply it to its inputs, and clear it (setting it to None).
        """

        raise NotImplementedError

    def encode_embedding_init_text(self, init_text, nvpt):
        """Converts text into a tensor with this text's tokens' embeddings. Note that those are embeddings before they are passed through
        transformers. nvpt is used as a maximum length in tokens. If text produces less teokens than nvpt, only this many is returned."""

        raise NotImplementedError

    def tokenize_line(self, line):
        """
        this transforms a single prompt into a list of PromptChunk objects - as many as needed to
        represent the prompt.
        Returns the list and the total number of tokens in the prompt.
        """


        parsed = [[line, 1.0]]

        tokenized = self.tokenize([text for text, _ in parsed])

        chunks = []
        chunk = PromptChunk()
        token_count = 0
        last_comma = -1

        def next_chunk(is_last=False):
            """puts current chunk into the list of results and produces the next one - empty;
            if is_last is true, tokens <end-of-text> tokens at the end won't add to token_count"""
            nonlocal token_count
            nonlocal last_comma
            nonlocal chunk

            if is_last:
                token_count += len(chunk.tokens)
            else:
                token_count += self.chunk_length

            to_add = self.chunk_length - len(chunk.tokens)
            if to_add > 0:
                chunk.tokens += [self.id_end] * to_add
                chunk.multipliers += [1.0] * to_add

            chunk.tokens = [self.id_start] + chunk.tokens + [self.id_end]
            chunk.multipliers = [1.0] + chunk.multipliers + [1.0]

            last_comma = -1
            chunks.append(chunk)
            chunk = PromptChunk()

        for tokens, (text, weight) in zip(tokenized, parsed):
            if text == 'BREAK' and weight == -1:
                next_chunk()
                continue

            position = 0
            while position < len(tokens):
                token = tokens[position]

                if token == self.comma_token:
                    last_comma = len(chunk.tokens)
                # this is when we are at the end of alloted 75 tokens for the current chunk, and the current token is not a comma. opts.comma_padding_backtrack
                # is a setting that specifies that if there is a comma nearby, the text after the comma should be moved out of this chunk and into the next.
                # comma_padding_backtrack=74
                # elif comma_padding_backtrack != 0 and len(chunk.tokens) == self.chunk_length and last_comma != -1 and len(chunk.tokens) - last_comma <= comma_padding_backtrack:
                #     break_location = last_comma + 1

                #     reloc_tokens = chunk.tokens[break_location:]
                #     reloc_mults = chunk.multipliers[break_location:]

                #     chunk.tokens = chunk.tokens[:break_location]
                #     chunk.multipliers = chunk.multipliers[:break_location]

                #     next_chunk()
                #     chunk.tokens = reloc_tokens
                #     chunk.multipliers = reloc_mults

                if len(chunk.tokens) == self.chunk_length:
                    next_chunk()

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, position)
                if embedding is None:
                    chunk.tokens.append(token)
                    chunk.multipliers.append(weight)
                    position += 1
                    continue

                emb_len = int(embedding.vec.shape[0])
                if len(chunk.tokens) + emb_len > self.chunk_length:
                    next_chunk()

                chunk.fixes.append(PromptChunkFix(len(chunk.tokens), embedding))

                chunk.tokens += [0] * emb_len
                chunk.multipliers += [weight] * emb_len
                position += embedding_length_in_tokens

        if len(chunk.tokens) > 0 or len(chunks) == 0:
            next_chunk(is_last=True)

        return chunks, token_count

    def process_texts(self, texts):
        """
        Accepts a list of texts and calls tokenize_line() on each, with cache. Returns the list of results and maximum
        length, in tokens, of all texts.
        """

        token_count = 0

        cache = {}
        batch_chunks = []
        for line in texts:
            if line in cache:
                chunks = cache[line]
            else:
                chunks, current_token_count = self.tokenize_line(line)
                token_count = max(current_token_count, token_count)

                cache[line] = chunks

            batch_chunks.append(chunks)

        return batch_chunks, token_count

    def forward(self, texts):
        """
        Accepts an array of texts; Passes texts through transformers network to create a tensor with numerical representation of those texts.
        Returns a tensor with shape of (B, T, C), where B is length of the array; T is length, in tokens, of texts (including padding) - T will
        be a multiple of 77; and C is dimensionality of each token - for SD1 it's 768, and for SD2 it's 1024.
        An example shape returned by this function can be: (2, 77, 768).
        Webui usually sends just one text at a time through this function - the only time when texts is an array with more than one elemenet
        is when you do prompt editing: "a picture of a [cat:dog:0.4] eating ice cream"
        """

        # if opts.use_old_emphasis_implementation:
        #     import modules.sd_hijack_clip_old
        #     return modules.sd_hijack_clip_old.forward_old(self, texts)

        batch_chunks, token_count = self.process_texts(texts)

        used_embeddings = {}
        chunk_count = max([len(x) for x in batch_chunks])

        zs = []
        for i in range(chunk_count):
            batch_chunk = [chunks[i] if i < len(chunks) else self.empty_chunk() for chunks in batch_chunks]

            tokens = [x.tokens for x in batch_chunk]
            multipliers = [x.multipliers for x in batch_chunk]
            self.hijack.fixes = [x.fixes for x in batch_chunk]

            for fixes in self.hijack.fixes:
                for position, embedding in fixes:
                    used_embeddings[embedding.name] = embedding

            z = self.process_tokens(tokens, multipliers)
            zs.append(z)

        if len(used_embeddings) > 0:
            embeddings_list = ", ".join([f'{name} [{embedding.checksum()}]' for name, embedding in used_embeddings.items()])
            self.hijack.comments.append(f"Used embeddings: {embeddings_list}")

        return torch.hstack(zs)

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        """
        sends one single prompt chunk to be encoded by transformers neural network.
        remade_batch_tokens is a batch of tokens - a list, where every element is a list of tokens; usually
        there are exactly 77 tokens in the list. batch_multipliers is the same but for multipliers instead of tokens.
        Multipliers are used to give more or less weight to the outputs of transformers network. Each multiplier
        corresponds to one token.
        """
        tokens = torch.asarray(remade_batch_tokens).to(device)

        # this is for SD2: SD1 uses the same token for padding and end of text, while SD2 uses different ones.
        if self.id_end != self.id_pad:
            for batch_pos in range(len(remade_batch_tokens)):
                index = remade_batch_tokens[batch_pos].index(self.id_end)
                tokens[batch_pos, index+1:tokens.shape[1]] = self.id_pad

        z = self.encode_with_transformers(tokens)

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers = torch.asarray(batch_multipliers).to(device)
        original_mean = z.mean()
        z = z * batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z = z * (original_mean / new_mean)

        return z


class FrozenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)
        self.tokenizer = wrapped.tokenizer

        vocab = self.tokenizer.get_vocab()

        self.comma_token = vocab.get(',</w>', None)

        self.token_mults = {}
        tokens_with_parens = [(k, v) for k, v in vocab.items() if '(' in k or ')' in k or '[' in k or ']' in k]
        for text, ident in tokens_with_parens:
            mult = 1.0
            for c in text:
                if c == '[':
                    mult /= 1.1
                if c == ']':
                    mult *= 1.1
                if c == '(':
                    mult *= 1.1
                if c == ')':
                    mult /= 1.1

            if mult != 1.0:
                self.token_mults[ident] = mult

        self.id_start = self.wrapped.tokenizer.bos_token_id
        self.id_end = self.wrapped.tokenizer.eos_token_id
        self.id_pad = self.id_end

    def tokenize(self, texts):
        tokenized = self.wrapped.tokenizer(texts, truncation=False, add_special_tokens=False)["input_ids"]

        return tokenized

    def encode_with_transformers(self, tokens):
        CLIP_stop_at_last_layers=1
        outputs = self.wrapped.transformer(input_ids=tokens, output_hidden_states=CLIP_stop_at_last_layers)

        z = outputs.last_hidden_state

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.transformer.text_model.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(embedding_layer.token_embedding.wrapped.weight.device)).squeeze(0)

        return embedded

#@title hijack_clip_old
# from drive.MyDrive import sd_hijack_clip
def process_text_old(self: FrozenCLIPEmbedderWithCustomWordsBase, texts):
    id_start = self.id_start
    id_end = self.id_end
    maxlen = self.wrapped.max_length  # you get to stay at 77
    used_custom_terms = []
    remade_batch_tokens = []
    hijack_comments = []
    hijack_fixes = []
    token_count = 0

    cache = {}
    batch_tokens = self.tokenize(texts)
    batch_multipliers = []
    for tokens in batch_tokens:
        tuple_tokens = tuple(tokens)

        if tuple_tokens in cache:
            remade_tokens, fixes, multipliers = cache[tuple_tokens]
        else:
            fixes = []
            remade_tokens = []
            multipliers = []
            mult = 1.0

            i = 0
            while i < len(tokens):
                token = tokens[i]

                embedding, embedding_length_in_tokens = self.hijack.embedding_db.find_embedding_at_position(tokens, i)

                mult_change = self.token_mults.get(token)
                if mult_change is not None:
                    mult *= mult_change
                    i += 1
                elif embedding is None:
                    remade_tokens.append(token)
                    multipliers.append(mult)
                    i += 1
                else:
                    emb_len = int(embedding.vec.shape[0])
                    fixes.append((len(remade_tokens), embedding))
                    remade_tokens += [0] * emb_len
                    multipliers += [mult] * emb_len
                    used_custom_terms.append((embedding.name, embedding.checksum()))
                    i += embedding_length_in_tokens

            if len(remade_tokens) > maxlen - 2:
                vocab = {v: k for k, v in self.wrapped.tokenizer.get_vocab().items()}
                ovf = remade_tokens[maxlen - 2:]
                overflowing_words = [vocab.get(int(x), "") for x in ovf]
                overflowing_text = self.wrapped.tokenizer.convert_tokens_to_string(''.join(overflowing_words))
                hijack_comments.append(f"Warning: too many input tokens; some ({len(overflowing_words)}) have been truncated:\n{overflowing_text}\n")

            token_count = len(remade_tokens)
            remade_tokens = remade_tokens + [id_end] * (maxlen - 2 - len(remade_tokens))
            remade_tokens = [id_start] + remade_tokens[0:maxlen - 2] + [id_end]
            cache[tuple_tokens] = (remade_tokens, fixes, multipliers)

        multipliers = multipliers + [1.0] * (maxlen - 2 - len(multipliers))
        multipliers = [1.0] + multipliers[0:maxlen - 2] + [1.0]

        remade_batch_tokens.append(remade_tokens)
        hijack_fixes.append(fixes)
        batch_multipliers.append(multipliers)
    return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count


def forward_old(self: FrozenCLIPEmbedderWithCustomWordsBase, texts):
    batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = process_text_old(self, texts)

    self.hijack.comments += hijack_comments

    if len(used_custom_terms) > 0:
        self.hijack.comments.append("Used embeddings: " + ", ".join([f'{word} [{checksum}]' for word, checksum in used_custom_terms]))

    self.hijack.fixes = hijack_fixes
    return self.process_tokens(remade_batch_tokens, batch_multipliers)

#@title hijack_open_clip

import open_clip.tokenizer
import torch

tokenizer = open_clip.tokenizer._tokenizer


class FrozenOpenCLIPEmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWordsBase):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.comma_token = [v for k, v in tokenizer.encoder.items() if k == ',</w>'][0]
        self.id_start = tokenizer.encoder["<start_of_text>"]
        self.id_end = tokenizer.encoder["<end_of_text>"]
        self.id_pad = 0

    def tokenize(self, texts):

        tokenized = [tokenizer.encode(text) for text in texts]

        return tokenized

    def encode_with_transformers(self, tokens):
        # set self.wrapped.layer_idx here according to opts.CLIP_stop_at_last_layers
        z = self.wrapped.encode_with_transformer(tokens)

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        ids = tokenizer.encode(init_text)
        ids = torch.asarray([ids], device=map_location, dtype=torch.int)
        embedded = self.wrapped.model.token_embedding.wrapped(ids).squeeze(0)

        return embedded

import open_clip.tokenizer
import torch

class FrozenXLMREmbedderWithCustomWords(FrozenCLIPEmbedderWithCustomWords):
    def __init__(self, wrapped, hijack):
        super().__init__(wrapped, hijack)

        self.id_start = wrapped.config.bos_token_id
        self.id_end = wrapped.config.eos_token_id
        self.id_pad = wrapped.config.pad_token_id

        self.comma_token = self.tokenizer.get_vocab().get(',', None)  # alt diffusion doesn't have </w> bits for comma

    def encode_with_transformers(self, tokens):
        # there's no CLIP Skip here because all hidden layers have size of 1024 and the last one uses a
        # trained layer to transform those 1024 into 768 for unet; so you can't choose which transformer
        # layer to work with - you have to use the last

        attention_mask = (tokens != self.id_pad).to(device=tokens.device, dtype=torch.int64)
        features = self.wrapped(input_ids=tokens, attention_mask=attention_mask)
        z = features['projection_state']

        return z

    def encode_embedding_init_text(self, init_text, nvpt):
        embedding_layer = self.wrapped.roberta.embeddings
        ids = self.wrapped.tokenizer(init_text, max_length=nvpt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        embedded = embedding_layer.token_embedding.wrapped(ids.to(device)).squeeze(0)

        return embedded

def reload_hypernetworks():

    global hypernetworks

    hypernetworks = list_hypernetworks(hypernetwork_dir)


import collections
import os.path
import sys
import gc
import torch
import re
import safetensors.torch
from omegaconf import OmegaConf
from os import mkdir
from urllib import request
import ldm.modules.midas as midas

from ldm.util import instantiate_from_config

# from modules import paths, shared, modelloader, devices, script_callbacks, sd_vae, sd_disable_initialization, errors, hashes, models_config
# from modules.paths import models_path
# from modules.sd_hijack_inpainting import do_inpainting_hijack
# from modules.timer import Timer

model_dir = models_path_gdrive
model_path = models_path_gdrive

checkpoints_list = {}
checkpoint_alisases = {}
checkpoints_loaded = collections.OrderedDict()


class CheckpointInfo:
    def __init__(self, filename):
        self.filename = filename
        abspath = os.path.abspath(filename)

        if models_path_gdrive is not None and abspath.startswith(models_path_gdrive):
            name = abspath.replace(models_path, '')
        elif abspath.startswith(model_path):
            name = abspath.replace(model_path, '')
        else:
            name = os.path.basename(filename)

        if name.startswith("\\") or name.startswith("/"):
            name = name[1:]

        self.name = name
        self.name_for_extra = os.path.splitext(os.path.basename(filename))[0]
        self.model_name = os.path.splitext(name.replace("/", "_").replace("\\", "_"))[0]
        self.hash = model_hash(filename)

        self.sha256 = sha256_from_cache(self.filename, "checkpoint/" + name)
        self.shorthash = self.sha256[0:10] if self.sha256 else None

        self.title = name if self.shorthash is None else f'{name} [{self.shorthash}]'

        self.ids = [self.hash, self.model_name, self.title, name, f'{name} [{self.hash}]'] + ([self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]'] if self.shorthash else [])

    def register(self):
        checkpoints_list[self.title] = self
        for id in self.ids:
            checkpoint_alisases[id] = self

    def calculate_shorthash(self):
        self.sha256 = sha256(self.filename, "checkpoint/" + self.name)
        if self.sha256 is None:
            return

        self.shorthash = self.sha256[0:10]

        if self.shorthash not in self.ids:
            self.ids += [self.shorthash, self.sha256, f'{self.name} [{self.shorthash}]']

        checkpoints_list.pop(self.title)
        self.title = f'{self.name} [{self.shorthash}]'
        self.register()

        return self.shorthash


try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging, CLIPModel

    logging.set_verbosity_error()
except Exception:
    pass


def setup_model():
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    list_models()
    enable_midas_autodownload()


def checkpoint_tiles():
    def convert(name):
        return int(name) if name.isdigit() else name.lower()

    def alphanumeric_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted([x.title for x in checkpoints_list.values()], key=alphanumeric_key)


def list_models():
    checkpoints_list.clear()
    checkpoint_alisases.clear()

    cmd_ckpt = model_checkpoint or custom_checkpoint_path
    if os.path.exists(cmd_ckpt):
        model_url = None
    else:
        model_url = "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.safetensors"

    model_list = modelloader.load_models(model_path=model_path, model_url=model_url, command_path=shared.cmd_opts.ckpt_dir, ext_filter=[".ckpt", ".safetensors"], download_name="v1-5-pruned-emaonly.safetensors", ext_blacklist=[".vae.ckpt", ".vae.safetensors"])

    if os.path.exists(cmd_ckpt):
        checkpoint_info = CheckpointInfo(cmd_ckpt)
        checkpoint_info.register()

        # shared.opts.data['model_checkpoint'] = checkpoint_info.title
    elif cmd_ckpt is not None and cmd_ckpt != model_checkpoint or custom_checkpoint_path:
        print(f"Checkpoint in --ckpt argument not found (Possible it was moved to {model_path}: {cmd_ckpt}", file=sys.stderr)

    for filename in model_list:
        checkpoint_info = CheckpointInfo(filename)
        checkpoint_info.register()


def get_closet_checkpoint_match(search_string):
    checkpoint_info = checkpoint_alisases.get(search_string, None)
    if checkpoint_info is not None:
        return checkpoint_info

    found = sorted([info for info in checkpoints_list.values() if search_string in info.title], key=lambda x: len(x.title))
    if found:
        return found[0]

    return None


def model_hash(filename):
    """old hash that only looks at a small part of the file and is prone to collisions"""

    try:
        with open(filename, "rb") as file:
            import hashlib
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return 'NOFILE'


def select_checkpoint():
    model_checkpoint = model_checkpoint or custom_checkpoint_path

    checkpoint_info = checkpoint_alisases.get(model_checkpoint, None)
    if checkpoint_info is not None:
        return checkpoint_info

    if len(checkpoints_list) == 0:
        print("No checkpoints found. When searching for checkpoints, looked at:", file=sys.stderr)
        if model_checkpoint or custom_checkpoint_path is not None:
            print(f" - file {os.path.abspath(model_checkpoint)}", file=sys.stderr)
        print(f" - directory {model_path}", file=sys.stderr)
        if models_path_gdrive is not None:
            print(f" - directory {os.path.abspath(shared.cmd_opts.ckpt_dir)}", file=sys.stderr)
        print("Can't run without a checkpoint. Find and place a .ckpt or .safetensors file into any of those locations. The program will exit.", file=sys.stderr)
        exit(1)

    checkpoint_info = next(iter(checkpoints_list.values()))
    if model_checkpoint is not None:
        print(f"Checkpoint {model_checkpoint} not found; loading fallback {checkpoint_info.title}", file=sys.stderr)

    return checkpoint_info


chckpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in chckpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k


def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd


def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or shared.weight_load_location or devices.get_optimal_device_name()
        pl_sd = safetensors.torch.load_file(checkpoint_file, device=device)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or shared.weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd


def get_checkpoint_state_dict(checkpoint_info: CheckpointInfo, timer):
    model_hash = checkpoint_info.calculate_shorthash()
    timer.record("calculate hash")

    if checkpoint_info in checkpoints_loaded:
        # use checkpoint cache
        print(f"Loading weights [{model_hash}] from cache")
        return checkpoints_loaded[checkpoint_info]

    print(f"Loading weights [{model_hash}] from {checkpoint_info.filename}")
    res = read_state_dict(checkpoint_info.filename)
    timer.record("load weights from disk")

    return res


def load_model_weights(model, checkpoint_info: CheckpointInfo, state_dict, timer):
    model_hash = checkpoint_info.calculate_shorthash()
    # timer.record("calculate hash")

    # shared.opts.data["model_checkpoint"] = checkpoint_info.title

    if state_dict is None:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    # timer.record("apply weights to model")

    # if shared.opts.sd_checkpoint_cache > 0:
    #     # cache newly loaded model
    #     checkpoints_loaded[checkpoint_info] = model.state_dict().copy()

    # if shared.cmd_opts.opt_channelslast:
    #     model.to(memory_format=torch.channels_last)
    #     timer.record("apply channels_last")
    no_half = False
    if not no_half:
        vae = model.first_stage_model
        depth_model = getattr(model, 'depth_model', None)

        # with --no-half-vae, remove VAE from model when doing half() to prevent its weights from being converted to float16
        # if shared.cmd_opts.no_half_vae:
        #     model.first_stage_model = None
        # # with --upcast-sampling, don't convert the depth model weights to float16
        # if shared.cmd_opts.upcast_sampling and depth_model:
        #     model.depth_model = None

        model.half()
        model.first_stage_model = vae
        if depth_model:
            model.depth_model = depth_model

        # timer.record("apply half()")

    devices.dtype = torch.float32
    devices.dtype_vae = torch.float32
    devices.dtype_unet = model.model.diffusion_model.dtype
    devices.unet_needs_upcast = False

    model.first_stage_model.to(devices.dtype_vae)
    # timer.record("apply dtype to VAE")

    # clean up cache if limit is reached
    while len(checkpoints_loaded) > shared.opts.sd_checkpoint_cache:
        checkpoints_loaded.popitem(last=False)

    model.model_hash = model_hash
    model.model_checkpoint = checkpoint_info.filename
    model.sd_checkpoint_info = checkpoint_info
    shared.opts.data["sd_checkpoint_hash"] = checkpoint_info.sha256

    model.logvar = model.logvar.to(device)  # fix for training

    # sd_vae.delete_base_vae()
    # sd_vae.clear_loaded_vae()
    # vae_file, vae_source = sd_vae.resolve_vae(checkpoint_info.filename)
    # sd_vae.load_vae(model, vae_file, vae_source)
    # timer.record("load VAE")


def enable_midas_autodownload():
    """
    Gives the ldm.modules.midas.api.load_model function automatic downloading.

    When the 512-depth-ema model, and other future models like it, is loaded,
    it calls midas.api.load_model to load the associated midas depth model.
    This function applies a wrapper to download the model to the correct
    location automatically.
    """

    midas_path = os.path.join(paths.models_path, 'midas')

    # stable-diffusion-stability-ai hard-codes the midas model path to
    # a location that differs from where other scripts using this model look.
    # HACK: Overriding the path here.
    for k, v in midas.api.ISL_PATHS.items():
        file_name = os.path.basename(v)
        midas.api.ISL_PATHS[k] = os.path.join(midas_path, file_name)

    midas_urls = {
        "dpt_large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
        "midas_v21": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21-f6b98070.pt",
        "midas_v21_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt",
    }

    midas.api.load_model_inner = midas.api.load_model

    def load_model_wrapper(model_type):
        path = midas.api.ISL_PATHS[model_type]
        if not os.path.exists(path):
            if not os.path.exists(midas_path):
                mkdir(midas_path)

            print(f"Downloading midas model weights for {model_type} to {path}")
            request.urlretrieve(midas_urls[model_type], path)
            print(f"{model_type} downloaded")

        return midas.api.load_model_inner(model_type)

    midas.api.load_model = load_model_wrapper


def repair_config(sd_config):

    if not hasattr(sd_config.model.params, "use_ema"):
        sd_config.model.params.use_ema = False

    if shared.cmd_opts.no_half:
        sd_config.model.params.unet_config.params.use_fp16 = False
    elif shared.cmd_opts.upcast_sampling:
        sd_config.model.params.unet_config.params.use_fp16 = True


sd1_clip_weight = 'cond_stage_model.transformer.text_model.embeddings.token_embedding.weight'
sd2_clip_weight = 'cond_stage_model.model.transformer.resblocks.0.attn.in_proj_weight'

def load_model(checkpoint_info=None, already_loaded_state_dict=None, time_taken_to_load_state_dict=None):
    from modules import lowvram, sd_hijack
    checkpoint_info = checkpoint_info or select_checkpoint()

    if shared.model:
        sd_hijack.model_hijack.undo_hijack(shared.model)
        shared.model = None
        gc.collect()
        devices.torch_gc()

    do_inpainting_hijack()

    timer = Timer()

    if already_loaded_state_dict is not None:
        state_dict = already_loaded_state_dict
    else:
        state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = models_config.find_checkpoint_config(state_dict, checkpoint_info)
    clip_is_included_into_sd = sd1_clip_weight in state_dict or sd2_clip_weight in state_dict

    timer.record("find config")

    sd_config = OmegaConf.load(checkpoint_config)
    repair_config(sd_config)

    timer.record("load config")

    print(f"Creating model from config: {checkpoint_config}")

    model = None
    try:
        with sd_disable_initialization.DisableInitialization(disable_clip=clip_is_included_into_sd):
            model = instantiate_from_config(sd_config.model)
    except Exception as e:
        pass

    if model is None:
        print('Failed to create model quickly; will retry using slow method.', file=sys.stderr)
        model = instantiate_from_config(sd_config.model)

    model.used_config = checkpoint_config

    timer.record("create model")

    load_model_weights(model, checkpoint_info, state_dict, timer)

    if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
        lowvram.setup_for_low_vram(model, shared.cmd_opts.medvram)
    else:
        model.to(shared.device)

    timer.record("move model to device")

    sd_hijack.model_hijack.hijack(model)

    timer.record("hijack")

    model.eval()
    shared.model = model

    sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings(force_reload=True)  # Reload embeddings after model load as they may or may not fit the model

    timer.record("load textual inversion embeddings")

    script_callbacks.model_loaded_callback(model)

    timer.record("scripts callbacks")

    print(f"Model loaded in {timer.summary()}.")

    return model


def reload_model_weights(model=None, info=None):
    from modules import lowvram, devices, sd_hijack
    checkpoint_info = info or select_checkpoint()

    if not model:
        model = shared.model

    if model is None:  # previous model load failed
        current_checkpoint_info = None
    else:
        current_checkpoint_info = model.sd_checkpoint_info
        if model.model_checkpoint == checkpoint_info.filename:
            return

        if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
            lowvram.send_everything_to_cpu()
        else:
            model.to(devices.cpu)

        sd_hijack.model_hijack.undo_hijack(model)

    timer = Timer()

    state_dict = get_checkpoint_state_dict(checkpoint_info, timer)

    checkpoint_config = models_config.find_checkpoint_config(state_dict, checkpoint_info)

    timer.record("find config")

    if model is None or checkpoint_config != model.used_config:
        del model
        checkpoints_loaded.clear()
        load_model(checkpoint_info, already_loaded_state_dict=state_dict, time_taken_to_load_state_dict=timer.records["load weights from disk"])
        return shared.model

    try:
        load_model_weights(model, checkpoint_info, state_dict, timer)
    except Exception as e:
        print("Failed to load checkpoint, restoring previous")
        load_model_weights(model, current_checkpoint_info, None, timer)
        raise
    finally:
        sd_hijack.model_hijack.hijack(model)
        timer.record("hijack")

        script_callbacks.model_loaded_callback(model)
        timer.record("script callbacks")

        if not shared.cmd_opts.lowvram and not shared.cmd_opts.medvram:
            model.to(devices.device)
            timer.record("move model to device")

    print(f"Weights loaded in {timer.summary()}.")

    return model

import torch
from glob import glob
import os
import re
import torch
from typing import Union

def get_state_dict_from_checkpoint(pl_sd):
    pl_sd = pl_sd.pop("state_dict", pl_sd)
    pl_sd.pop("state_dict", None)

    sd = {}
    for k, v in pl_sd.items():
        new_key = transform_checkpoint_dict_key(k)

        if new_key is not None:
            sd[new_key] = v

    pl_sd.clear()
    pl_sd.update(sd)

    return pl_sd

def read_metadata_from_safetensors(filename):
    import json

    with open(filename, mode="rb") as file:
        metadata_len = file.read(8)
        metadata_len = int.from_bytes(metadata_len, "little")
        json_start = file.read(2)

        assert metadata_len > 2 and json_start in (b'{"', b"{'"), f"{filename} is not a safetensors file"
        json_data = json_start + file.read(metadata_len-2)
        json_obj = json.loads(json_data)

        res = {}
        for k, v in json_obj.get("__metadata__", {}).items():
            res[k] = v
            if isinstance(v, str) and v[0:1] == '{':
                try:
                    res[k] = json.loads(v)
                except Exception as e:
                    pass

        return res

weight_load_location = 'cpu'

lora_dir = '/content/drive/MyDrive/models/loras/' #@param {'type':'string'}

def torch_load_file(filename, device):
    result = {}
    with safe_open(filename, framework="pt", device=device) as f:
        for k in f.keys():
            result[k] = f.get_tensor(k)
    return result

# safetensors.torch.load_file = torch_load_file

def read_state_dict(checkpoint_file, print_global_state=False, map_location=None):
    _, extension = os.path.splitext(checkpoint_file)
    if extension.lower() == ".safetensors":
        device = map_location or weight_load_location
        pl_sd = torch_load_file(checkpoint_file, device=device)
    else:
        pl_sd = torch.load(checkpoint_file, map_location=map_location or weight_load_location)

    if print_global_state and "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    sd = get_state_dict_from_checkpoint(pl_sd)
    return sd

metadata_tags_order = {"ss_model_name": 1, "ss_resolution": 2, "ss_clip_skip": 3, "ss_num_train_images": 10, "ss_tag_frequency": 20}

re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}


def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    return key


class LoraOnDisk:
    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.metadata = {}

        _, ext = os.path.splitext(filename)
        if ext.lower() == ".safetensors":
            try:
                self.metadata = read_metadata_from_safetensors(filename)
            except Exception as e:
                print(e, f"reading lora {filename}")

        if self.metadata:
            m = {}
            for k, v in sorted(self.metadata.items(), key=lambda x: metadata_tags_order.get(x[0], 999)):
                m[k] = v

            self.metadata = m

        self.ssmd_cover_images = self.metadata.pop('ssmd_cover_images', None)  # those are cover images and they are too big to display in UI as text


class LoraModule:
    def __init__(self, name):
        self.name = name
        self.multiplier = 1.0
        self.modules = {}
        self.mtime = None


class LoraUpDownModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None


def assign_lora_names_to_compvis_modules(model):
    lora_layer_mapping = {}

    for name, module in model.cond_stage_model.wrapped.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    for name, module in model.model.named_modules():
        lora_name = name.replace(".", "_")
        lora_layer_mapping[lora_name] = module
        module.lora_layer_name = lora_name

    model.lora_layer_mapping = lora_layer_mapping


def load_lora(name, filename):
    lora = LoraModule(name)
    lora.mtime = os.path.getmtime(filename)

    sd = read_state_dict(filename)

    keys_failed_to_match = {}
    is_sd2 = 'model_transformer_resblocks' in model.lora_layer_mapping

    for key_diffusers, weight in sd.items():
        key_diffusers_without_lora_parts, lora_key = key_diffusers.split(".", 1)
        key = convert_diffusers_name_to_compvis(key_diffusers_without_lora_parts, is_sd2)

        sd_module = model.lora_layer_mapping.get(key, None)

        if sd_module is None:
            m = re_x_proj.match(key)
            if m:
                sd_module = model.lora_layer_mapping.get(m.group(1), None)

        if sd_module is None:
            keys_failed_to_match[key_diffusers] = key
            continue

        lora_module = lora.modules.get(key, None)
        if lora_module is None:
            lora_module = LoraUpDownModule()
            lora.modules[key] = lora_module

        if lora_key == "alpha":
            lora_module.alpha = weight.item()
            continue

        if type(sd_module) == torch.nn.Linear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.modules.linear.NonDynamicallyQuantizableLinear:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.MultiheadAttention:
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        elif type(sd_module) == torch.nn.Conv2d:
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            print(f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}')
            continue
            assert False, f'Lora layer {key_diffusers} matched a layer with unsupported type: {type(sd_module).__name__}'

        with torch.no_grad():
            module.weight.copy_(weight)

        module.to(device='cpu', dtype=model.dtype)

        if lora_key == "lora_up.weight":
            lora_module.up = module
        elif lora_key == "lora_down.weight":
            lora_module.down = module
        else:
            assert False, f'Bad Lora layer name: {key_diffusers} - must end in lora_up.weight, lora_down.weight or alpha'

    if len(keys_failed_to_match) > 0:
        print(f"Failed to match keys when loading Lora {filename}: {keys_failed_to_match}")

    return lora


def load_loras(names, multipliers=None):
    already_loaded = {}

    for lora in loaded_loras:
        if lora.name in names:
            already_loaded[lora.name] = lora

    loaded_loras.clear()

    loras_on_disk = [available_loras.get(name, None) for name in names]
    if any([x is None for x in loras_on_disk]):
        list_available_loras()

        loras_on_disk = [available_loras.get(name, None) for name in names]

    for i, name in enumerate(names):
        lora = already_loaded.get(name, None)

        lora_on_disk = loras_on_disk[i]
        if lora_on_disk is not None:
            if lora is None or os.path.getmtime(lora_on_disk.filename) > lora.mtime:
                lora = load_lora(name, lora_on_disk.filename)

        if lora is None:
            print(f"Couldn't find Lora with name {name}")
            continue

        lora.multiplier = multipliers[i] if multipliers else 1.0
        loaded_loras.append(lora)


def lora_calc_updown(lora, module, target):
    with torch.no_grad():
        up = module.up.weight.to(target.device, dtype=target.dtype)
        down = module.down.weight.to(target.device, dtype=target.dtype)

        if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
            updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            updown = up @ down

        updown = updown * lora.multiplier * (module.alpha / module.up.weight.shape[1] if module.alpha else 1.0)

        return updown


def lora_apply_weights(self: Union[torch.nn.Conv2d, torch.nn.Linear, torch.nn.MultiheadAttention]):
    """
    Applies the currently selected set of Loras to the weights of torch layer self.
    If weights already have this particular set of loras applied, does nothing.
    If not, restores orginal weights from backup and alters weights according to loras.
    """

    lora_layer_name = getattr(self, 'lora_layer_name', None)
    if lora_layer_name is None:
        return

    current_names = getattr(self, "lora_current_names", ())
    wanted_names = tuple((x.name, x.multiplier) for x in loaded_loras)

    weights_backup = getattr(self, "lora_weights_backup", None)
    if weights_backup is None:
        if isinstance(self, torch.nn.MultiheadAttention):
            weights_backup = (self.in_proj_weight.to('cpu', copy=True), self.out_proj.weight.to('cpu', copy=True))
        else:
            weights_backup = self.weight.to('cpu', copy=True)

        self.lora_weights_backup = weights_backup

    if current_names != wanted_names:
        if weights_backup is not None:
            if isinstance(self, torch.nn.MultiheadAttention):
                self.in_proj_weight.copy_(weights_backup[0])
                self.out_proj.weight.copy_(weights_backup[1])
            else:
                self.weight.copy_(weights_backup)

        for lora in loaded_loras:
            module = lora.modules.get(lora_layer_name, None)
            if module is not None and hasattr(self, 'weight'):
                self.weight += lora_calc_updown(lora, module, self.weight)
                continue

            module_q = lora.modules.get(lora_layer_name + "_q_proj", None)
            module_k = lora.modules.get(lora_layer_name + "_k_proj", None)
            module_v = lora.modules.get(lora_layer_name + "_v_proj", None)
            module_out = lora.modules.get(lora_layer_name + "_out_proj", None)

            if isinstance(self, torch.nn.MultiheadAttention) and module_q and module_k and module_v and module_out:
                updown_q = lora_calc_updown(lora, module_q, self.in_proj_weight)
                updown_k = lora_calc_updown(lora, module_k, self.in_proj_weight)
                updown_v = lora_calc_updown(lora, module_v, self.in_proj_weight)
                updown_qkv = torch.vstack([updown_q, updown_k, updown_v])

                self.in_proj_weight += updown_qkv
                self.out_proj.weight += lora_calc_updown(lora, module_out, self.out_proj.weight)
                continue

            if module is None:
                continue

            print(f'failed to calculate lora weights for layer {lora_layer_name}')

        setattr(self, "lora_current_names", wanted_names)


def lora_reset_cached_weight(self: Union[torch.nn.Conv2d, torch.nn.Linear]):
    setattr(self, "lora_current_names", ())
    setattr(self, "lora_weights_backup", None)


def lora_Linear_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Linear_forward_before_lora(self, input)


def lora_Linear_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Linear_load_state_dict_before_lora(self, *args, **kwargs)


def lora_Conv2d_forward(self, input):
    lora_apply_weights(self)

    return torch.nn.Conv2d_forward_before_lora(self, input)


def lora_Conv2d_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.Conv2d_load_state_dict_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_forward(self, *args, **kwargs):
    lora_apply_weights(self)

    return torch.nn.MultiheadAttention_forward_before_lora(self, *args, **kwargs)


def lora_MultiheadAttention_load_state_dict(self, *args, **kwargs):
    lora_reset_cached_weight(self)

    return torch.nn.MultiheadAttention_load_state_dict_before_lora(self, *args, **kwargs)


def list_available_loras():
    available_loras.clear()

    os.makedirs(lora_dir, exist_ok=True)

    candidates = \
        glob(os.path.join(lora_dir, '**/*.pt'), recursive=True) + \
        glob(os.path.join(lora_dir, '**/*.safetensors'), recursive=True) + \
        glob(os.path.join(lora_dir, '**/*.ckpt'), recursive=True)

    for filename in sorted(candidates, key=str.lower):
        if os.path.isdir(filename):
            continue

        name = os.path.splitext(os.path.basename(filename))[0]

        available_loras[name] = LoraOnDisk(name, filename)




def unload():
    torch.nn.Linear.forward = torch.nn.Linear_forward_before_lora
    torch.nn.Linear._load_from_state_dict = torch.nn.Linear_load_state_dict_before_lora
    torch.nn.Conv2d.forward = torch.nn.Conv2d_forward_before_lora
    torch.nn.Conv2d._load_from_state_dict = torch.nn.Conv2d_load_state_dict_before_lora
    torch.nn.MultiheadAttention.forward = torch.nn.MultiheadAttention_forward_before_lora
    torch.nn.MultiheadAttention._load_from_state_dict = torch.nn.MultiheadAttention_load_state_dict_before_lora

if not hasattr(torch.nn, 'Linear_forward_before_lora'):
    torch.nn.Linear_forward_before_lora = torch.nn.Linear.forward

if not hasattr(torch.nn, 'Linear_load_state_dict_before_lora'):
    torch.nn.Linear_load_state_dict_before_lora = torch.nn.Linear._load_from_state_dict

if not hasattr(torch.nn, 'Conv2d_forward_before_lora'):
    torch.nn.Conv2d_forward_before_lora = torch.nn.Conv2d.forward

if not hasattr(torch.nn, 'Conv2d_load_state_dict_before_lora'):
    torch.nn.Conv2d_load_state_dict_before_lora = torch.nn.Conv2d._load_from_state_dict

if not hasattr(torch.nn, 'MultiheadAttention_forward_before_lora'):
    torch.nn.MultiheadAttention_forward_before_lora = torch.nn.MultiheadAttention.forward

if not hasattr(torch.nn, 'MultiheadAttention_load_state_dict_before_lora'):
    torch.nn.MultiheadAttention_load_state_dict_before_lora = torch.nn.MultiheadAttention._load_from_state_dict

checkpoint_dict_replacements = {
    'cond_stage_model.transformer.embeddings.': 'cond_stage_model.transformer.text_model.embeddings.',
    'cond_stage_model.transformer.encoder.': 'cond_stage_model.transformer.text_model.encoder.',
    'cond_stage_model.transformer.final_layer_norm.': 'cond_stage_model.transformer.text_model.final_layer_norm.',
}


def transform_checkpoint_dict_key(k):
    for text, replacement in checkpoint_dict_replacements.items():
        if k.startswith(text):
            k = replacement + k[len(text):]

    return k

def inject_lora(model):
  torch.nn.Linear.forward = lora_Linear_forward
  torch.nn.Linear._load_from_state_dict = lora_Linear_load_state_dict
  torch.nn.Conv2d.forward = lora_Conv2d_forward
  torch.nn.Conv2d._load_from_state_dict = lora_Conv2d_load_state_dict
  torch.nn.MultiheadAttention.forward = lora_MultiheadAttention_forward
  torch.nn.MultiheadAttention._load_from_state_dict = lora_MultiheadAttention_load_state_dict

  assign_lora_names_to_compvis_modules(model)

available_loras = {}
loaded_loras = []
import safetensors

def split_lora_from_prompts(prompts):
  re1 = '\<(.*?)\>'
  new_prompt_loras = {}
  new_prompts = {}

  #iterate through prompts keyframes and fill in lora schedules
  for key in prompts.keys():
    subp = prompts[key][0]

    #get a dict of loras:weights from a prompt
    prompt_loras = re.findall(re1, subp)
    prompt_loras_dict = dict([(o.split(':')[1], o.split(':')[-1]) for o in prompt_loras])

    #fill lora dict based on keyframe, lora:weight
    for lora_key in prompt_loras_dict.keys():
      try: new_prompt_loras[lora_key]
      except: new_prompt_loras[lora_key] = {}
      new_prompt_loras[lora_key][key] = float(prompt_loras_dict[lora_key])

    # remove lora keywords from promtps
    new_prompts[key] = [re.sub(re1, '', subp).strip(' ')]

  return new_prompts, new_prompt_loras

def get_loras_weights_for_frame(frame_num, loras_dict):
  loras = list(loras_dict.keys())
  loras_weights = [get_scheduled_arg(frame_num, loras_dict[o]) for o in loras]
  return loras, loras_weights
