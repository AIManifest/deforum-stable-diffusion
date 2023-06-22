from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger

import torch
import copy
import datetime as dt
import gc
from itertools import chain, cycle
import json
import os
import re
import string
import subprocess
from subprocess import Popen, PIPE
import textwrap
import time
import warnings

from IPython.display import display
import numpy as np
import pandas as pd
import panel as pn
from tqdm.auto import tqdm

import whisper

import io
from pathlib import Path
import select
from shutil import rmtree
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO

import librosa
import numpy as np

def find_files(music_video_args):
    out = []
    for file in Path(music_video_args.in_path).iterdir():
        if file.suffix.lower().lstrip(".") in music_video_args.extensions:
            out.append(file)
    return out

def copy_process_streams(process: sp.Popen):
    def raw(stream: Optional[IO[bytes]]) -> IO[bytes]:
        assert stream is not None
        if isinstance(stream, io.BufferedIOBase):
            stream = stream.raw
        return stream

    p_stdout, p_stderr = raw(process.stdout), raw(process.stderr)
    stream_by_fd: Dict[int, Tuple[IO[bytes], io.StringIO, IO[str]]] = {
        p_stdout.fileno(): (p_stdout, sys.stdout),
        p_stderr.fileno(): (p_stderr, sys.stderr),
    }
    fds = list(stream_by_fd.keys())

    while fds:
        # `select` syscall will wait until one of the file descriptors has content.
        ready, _, _ = select.select(fds, [], [])
        for fd in ready:
            p_stream, std = stream_by_fd[fd]
            raw_buf = p_stream.read(2 ** 16)
            if not raw_buf:
                fds.remove(fd)
                continue
            buf = raw_buf.decode()
            std.write(buf)
            std.flush()

def separate_audio_stems(music_video_args):
    inp = music_video_args.in_path
    outp = music_video_args.out_path
    cmd = ["python", "-m", "demucs.separate", "-o", str(outp), "-n", music_video_args.model]
    if music_video_args.mp3:
        cmd += ["--mp3", f"--mp3-bitrate={music_video_args.mp3_rate}"]
    if music_video_args.float32:
        cmd += ["--float32"]
    if music_video_args.int24:
        cmd += ["--int24"]
    if music_video_args.two_stems is not None:
        cmd += [f"--two-stems={music_video_args.two_stems}"]
    files = [str(f) for f in find_files(music_video_args)]
    if not files:
        print(f"No valid audio files in {music_video_args.in_path}")
        return
    print("Going to separate the files:")
    print('\n'.join(files))
    print("With command: ", " ".join(cmd))
    p = sp.Popen(cmd + files, stdout=sp.PIPE, stderr=sp.PIPE)
    copy_process_streams(p)
    p.wait()
    if p.returncode != 0:
        print("Command failed, something went wrong.")
    torch.cuda.empty_cache()

def get_audio_duration_seconds(audio_fpath):
        outv = subprocess.run([
            'ffprobe'
            ,'-i',audio_fpath
            ,'-show_entries', 'format=duration'
            ,'-v','quiet'
            ,'-of','csv=p=0'
            ],
            stdout=subprocess.PIPE
            ).stdout.decode('utf-8')
        return float(outv.strip())

def create_music_video_animation_args(root, music_video_args):
    audio_root = os.path.join(root.output_path_gdrive, f"{music_video_args.project_name}_audio")
    audio_root = Path(audio_root)
    storyboard = OmegaConf.create()
    if music_video_args.yt_video_url:
        logger.info("--Initializing Audio Analysis--")        
        video_url = music_video_args.yt_video_url
        os.makedirs(audio_root, exist_ok=True)
        # check if user provided an audio filepath (or we already have one from youtube) before attempting to download
        if music_video_args.audio_fpath is None:
            storyboard = OmegaConf.create()

            d_ = dict(
                # all the underscore does is make it so each of the following lines can be preceded with a comma
                # otw the first parameter would be offset from the other in the colab form
                _=""

                , video_url = music_video_args.yt_video_url # @param {type:'string'}
                , audio_fpath = '' # @param {type:'string'}
            )

            d_.pop('_')
            storyboard.params = d_

            ytdl_prefix = "DOWNLOADED__"
            ytdl_fname = f"{str(audio_root / ytdl_prefix)}%(title)s.%(ext)s"
            logger.info("..Downloading YT Video..")
            running = subprocess.Popen(['yt-dlp', '-o', f'{ytdl_fname}', f'{video_url}'],
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE)
            output, error = running.communicate()
            print(error)

            matched_files = audio_root.glob(ytdl_prefix+"*")
            most_recent_file = max(matched_files, key=os.path.getctime)
            print(f"downloaded: {most_recent_file}")
            ytdl_fname = most_recent_file
            # new attribute cause why not
            music_video_args.downloaded_video_fpath = ytdl_fname

            #audio_fpath = str( root / 'audio.aac' )
            audio_fpath = str( audio_root / 'audio.m4a' )
            audio_xpath = str( audio_root / 'audio.mp3' )
            input_audio = ytdl_fname
            #!ffmpeg -y -i "{input_audio}" -c:a aac {audio_fpath}
            #!ffmpeg -y -i "{input_audio}" -vn -acodec copy {audio_fpath}
            logger.info("..Converting to '.m4a'..")
            running = subprocess.Popen(['ffmpeg', '-y', '-i', f'{input_audio}',
                                    '-vn', '-c:a', 'aac', f'{audio_fpath}'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            output, error = running.communicate()
            print(error)
            logger.info("..Creating '.mp3' file for demucs separation..")
            running = subprocess.Popen(['ffmpeg', '-y', '-i', f'{audio_fpath}',
                                    f'{audio_xpath}'],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
            output, error = running.communicate()
            print(error)
            
    # to do: write audio and subtitle paths/meta to storyboard
    audio_fpath = str( audio_root / 'audio.m4a' ) if music_video_args.audio_fpath is None else music_video_args.audio_fpath
    # music_video_args.audio_fpath = audio_fpath
    print(f'AUDIO FPATH: {audio_fpath}')
    storyboard_fname = audio_root / 'storyboard.yaml'
    with open(storyboard_fname,'wb') as fp:
        OmegaConf.save(config=storyboard, f=fp.name)

    if music_video_args.video_duration is None:
        # estimate duration from audio file
        audio_fpath = music_video_args.audio_fpath if music_video_args.audio_fpath is not None else audio_fpath
        music_video_args.video_duration = get_audio_duration_seconds(audio_fpath)
        print(music_video_args.video_duration)

    music_video_max_frames = int(music_video_args.video_duration) * music_video_args.music_video_target_fps

    if music_video_args.video_duration is None:
        raise RuntimeError('unable to determine audio duration. was a video url or path to a file supplied?')

    # force use
    music_video_args.whisper_seg = True

    whisper_seg = music_video_args.whisper_seg

    storyboard = OmegaConf.create()

    if 'hf_helper' in locals():
        del hf_helper.img2img
        del hf_helper.text2img
        del hf_helper
    
    gc.collect()
    torch.cuda.empty_cache()

    if whisper_seg:
        logger.info("..Segmenting with Whisper..")
        storyboard_fname = audio_root / 'storyboard.yaml'
        audio_xpath = str( audio_root / 'audio.mp3' )
        audio_xpath = music_video_args.audio_fpath if music_video_args.audio_fpath is not None else audio_xpath
        print(f'AUDIO: {audio_xpath}')
        running = subprocess.Popen(["whisper", 
                "--model", 
                "large-v2", 
                "--word_timestamps", 
                "True",
                "--condition_on_previous_text", "False", "-o", str(audio_root), audio_xpath], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = running.communicate()
        print(output, error)

        # outputs text files as audio.* locally
        #with Path('audio.json').open() as f:
        music_video_args.audio_fpath = str( audio_root / 'audio.m4a' )
        whisper_seg_fpath = Path(music_video_args.audio_fpath).with_suffix('.json')
        whisper_seg_fpath = audio_root / whisper_seg_fpath
        with whisper_seg_fpath.open() as f:
            timings = json.load(f)


        # TODO: this doesn't need to be a function...
        def whisper_segments_to_vktrs_promptstarts(segments):
            for rec in segments:
                rec['ts'] = rec['start']
                rec['prompt'] = rec['text']
            return segments

        prompt_starts = whisper_segments_to_vktrs_promptstarts(timings['segments'])

        # i don't think this is reliable unfortunately.
        #storyboard.params['video_duration'] = storyboard.prompt_starts[-1]['end']
        video_duration = get_audio_duration_seconds(audio_fpath)

        # prompt_starts = storyboard.prompt_starts

        ### checkpoint the processing work we've done to this point

        prompt_starts_copy = copy.deepcopy(prompt_starts)
        prompt_starts = prompt_starts_copy

        with open(storyboard_fname) as fp:
            OmegaConf.save(config=storyboard, f=fp.name)

        prompt_starts_copy = copy.deepcopy(prompt_starts)
        
        theme_prompt = music_video_args.added_prompt

        cond_prompts = {}
        cond_prompts = {int(entry['ts']): entry['prompt'] + theme_prompt for entry in prompt_starts_copy}
        cond_prompts = {key * music_video_args.music_video_target_fps: value for key, value in cond_prompts.items()}

        # to do: deal with timedeltas in asr.py and yt.py
        for rec in prompt_starts_copy:
            for k,v in list(rec.items()):
                if isinstance(v, dt.timedelta):
                    rec[k] = v.total_seconds()
        
        music_video_args.prompt_starts = prompt_starts_copy

        music_video_args.model = "htdemucs"
        music_video_args.extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
        music_video_args.two_stems = "drums"   # only separate one stems from the rest, for instance
        # two_stems = "vocals"

        # Options for the output audio.
        music_video_args.mp3 = True
        music_video_args.mp3_rate = 320
        music_video_args.float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
        music_video_args.int24 = False    # output as int24 wavs, unused if 'mp3' is True.
        # You cannot set both `float32 = True` and `int24 = True` !!

        music_video_args.in_path = audio_root
        music_video_args.out_path = audio_root

        separate_audio_stems(music_video_args)
        logger.info("..Printing to A Interactive Downloadable Table..")
        pn.extension('tabulator') # I don't know that specifying 'tabulator' here is even necessary...
        pn.widgets.Tabulator.theme = 'site'

        tabulator_formatters = {
            'bool': {'type': 'tickCross'}
        }

        # df = pd.DataFrame(cond_prompts).rename(
        #     columns={
        #         'ts':'Timestamp (sec)',
        #         'prompt':'Lyric'
        #     }
        # )
        df = pd.DataFrame(list(cond_prompts.items()), columns=['Timestamp (sec)', 'Lyric'])
        df_pre = copy.deepcopy(df)
        download_df = pn.widgets.Tabulator(df, formatters=tabulator_formatters)

        filename, button = download_df.download_menu(
            text_kwargs={'name': 'Enter filename', 'value': 'default.csv'},
            button_kwargs={'name': 'Download Table'}
        )
        display(pn.Row(
            pn.Column(filename, button),
            download_df
        ))

    return cond_prompts, music_video_max_frames

def get_keyframes_total_for_animation(music_video_args):
    audio, sr = librosa.load("/content/drive/MyDrive/AI/StableDiffusion/audio/htdemucs/audio/drums.mp3")

    rms = librosa.feature.rms(y=audio)[0]

    threshold = 0.1

    baseline = music_video_args.baseline

    keyframe_frames = np.where(rms > threshold)[0]

    keyframe_times = librosa.frames_to_time(keyframe_frames, sr=sr)

    keyframes = {}

    # Assign the volume rises to their respective keyframes
    # for i, frame in enumerate(keyframe_frames):
    #     if i in keyframe_frames:
    #         keyframes[int(keyframe_times[i])] = rms[frame]+0.4
    #     else:

    num_frames = len(rms)

    # Iterate over all frames in the audio file
    for i in range(num_frames):
        # Check if the current frame is present in the keyframe_frames array
        if i in keyframe_frames:
            # Assign the corresponding volume rise (plus 0.4) to the keyframes dictionary
            keyframes[i] = rms[i] + 0.4
        else:
            # Assign a different value to the keyframes dictionary
            keyframes[i] = baseline


    keyframes_total = ''
    for key, value in sorted(keyframes.items()):
        keyframes_total += f"{key}: ({value}), "

    keyframes_total = f"{str(keyframes_total.rstrip(', '))}"

    # print(keyframes_total)

    return keyframes_total

def get_camera_movement(music_video_prompts, amount):
    camera_movement = music_video_prompts

    for key, value in camera_movement.items():
        camera_movement[key] = amount

    new_camera_movement = {}
    for i, (key, value) in enumerate(camera_movement.items()):
        if i % 2 == 0:
            new_value = value
        else:
            new_value = -value
        new_camera_movement[key] = new_value
    camera_movement = new_camera_movement

    for i in range(max(camera_movement.keys()) + 1):
        if i not in camera_movement:
            camera_movement[i] = '(0)'

    # print(camera_movement)

    camera_movement_total = ''
    for key, value in sorted(camera_movement.items()):
        camera_movement_total += f"{key}: ({value}), "

    camera_movement_total = f"{str(camera_movement_total.rstrip(', '))}"

    # print(camera_movement_total)
    return camera_movement_total
