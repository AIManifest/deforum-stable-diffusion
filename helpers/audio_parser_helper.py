from omegaconf import OmegaConf
from pathlib import Path

import torch
import copy
import datetime as dt
import gc
from itertools import chain, cycle
import json
import os
import re
import string
from subprocess import Popen, PIPE
import textwrap
import time
import warnings

from IPython.display import display
import numpy as np
import pandas as pd
import panel as pn
from tqdm.auto import tqdm

import tokenizations
import webvtt
import whisper

from vktrs.utils import remove_punctuation
from vktrs.utils import get_audio_duration_seconds
from vktrs.youtube import (
    YoutubeHelper,
    parse_timestamp,
    vtt_to_token_timestamps,
    srv2_to_token_timestamps,
)

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
    cmd = ["python3", "-m", "demucs.separate", "-o", str(outp), "-n", music_video_args.model]
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

def create_music_video_animation_args(root, music_video_args):
    if music_video_args.yt_video_url:
        print("Initializing Audio Analysis")
        audio_root = os.path.join(root.output_path_gdrive, "audio")
        os.makedirs(audio_root, exist_ok=True)
        # check if user provided an audio filepath (or we already have one from youtube) before attempting to download
        if music_video_args.audio_fpath is None:
            helper = YoutubeHelper(
                music_video_args.yt_video_url,
                ydl_opts = {
                    'outtmpl':{'default':os.path.join(audio_root, f"ytdlp_content.%(ext)s" )},
                    'writeautomaticsub':True,
                    'subtitlesformat':'srv2/vtt'
                    },
            )

            # estimate video end
            video_duration = dt.timedelta(seconds=helper.info['duration'])
            music_video_args.video_duration = video_duration.total_seconds()

            audio_fpath = audio_root + "/audio.mp3"
            input_audio = helper.info['requested_downloads'][-1]['filepath']
            runffmpeg = Popen(['ffmpeg',
                                    '-y', '-i', 
                                    input_audio, 
                                    '-acodec', 'libmp3lame', 
                                    audio_fpath], 
                                    stdout=PIPE, stderr=PIPE)
            foutput, ferror = runffmpeg.communicate()
            if runffmpeg.returncode != 0:
                print(ferror)
                raise RuntimeError(ferror)            

            # to do: write audio and subtitle paths/meta to storyboard
            music_video_args.audio_fpath = audio_fpath

            if False:
                subtitle_format = helper.info['requested_subtitles']['en']['ext']
                subtitle_fpath = helper.info['requested_subtitles']['en']['filepath']

                if subtitle_format == 'srv2':
                    with open(subtitle_fpath, 'r') as f:
                        srv2_xml = f.read() 
                    token_start_times = srv2_to_token_timestamps(srv2_xml)
                    # to do: handle timedeltas...
                    #storyboard.params.token_start_times = token_start_times

                elif subtitle_format == 'vtt':
                    captions = webvtt.read(subtitle_fpath)
                    token_start_times = vtt_to_token_timestamps(captions)
                    # to do: handle timedeltas...
                    #storyboard.params.token_start_times = token_start_times

                # If unable to download supported subtitles, force use whisper
                else:
                    music_video_args.whisper_seg = True
    
    if music_video_args.video_duration is None:
        # estimate duration from audio file
        audio_fpath = music_video_args.audio_fpath
        music_video_args.video_duration = get_audio_duration_seconds(audio_fpath)
        print(music_video_args.video_duration)

    music_video_max_frames = int(music_video_args.video_duration) * music_video_args.music_video_target_fps

    if music_video_args.video_duration is None:
        raise RuntimeError('unable to determine audio duration. was a video url or path to a file supplied?')

    # force use
    music_video_args.whisper_seg = True

    whisper_seg = music_video_args.whisper_seg

    if 'hf_helper' in locals():
        del hf_helper.img2img
        del hf_helper.text2img
        del hf_helper


    if whisper_seg:
        from vktrs.asr import (
            #whisper_lyrics,
            #whisper_transcribe,
            #whisper_align,
            whisper_transmit_meta_across_alignment,
            whisper_segment_transcription,
        )

        #prompt_starts = whisper_lyrics(audio_fpath=storyboard.params.audio_fpath)

        audio_fpath = music_video_args.audio_fpath
        #whispers = whisper_transcribe(audio_fpath)

        # to do: dropdown selectors
        segmentation_model = 'tiny'
        transcription_model = 'large'

        music_video_args.whisper = dict(
            segmentation_model = segmentation_model
            ,transcription_model = transcription_model
        )

        whispers = {
            #'tiny':None, # 5.83 s
            #'large':None # 3.73 s
        }
        # accelerated runtime required for whisper
        # to do: pypi package for whisper

        # to do: use transcripts we've already built if we have them
        #scripts = storyboard.params.whisper.get('transcriptions')
        
        for k in set([segmentation_model, transcription_model]):
            #if k in scripts:

            options = whisper.DecodingOptions(
                language='en',
            )
            # to do: be more proactive about cleaning up these models when we're done with them
            model = whisper.load_model(k).to('cuda')
            start = time.time()
            print(f"Transcribing audio with whisper-{k}")
            
            # to do: calling transcribe like this unnecessarily re-processes audio each time.
            whispers[k] = model.transcribe(audio_fpath) # re-processes audio each time, ~10s overhead?
            print(f"elapsed: {time.time()-start}")
            del model
            gc.collect()

        transcriptions = {}
        transcription_root = os.path.join(root.output_path_gdrive, 'whispers')
        os.makedirs(transcription_root, exist_ok=True)
        writer = whisper.utils.get_writer(output_format='vtt', output_dir=transcription_root) # output dir doesn't do anything...?
        for k in whispers:
            outpath = os.path.join(transcription_root, f"{k}.vtt")
            transcriptions[k] = outpath
            with open(outpath,'w') as f:
                # to do: upstream PR to control verbosity
                writer.write_result(
                    whispers[k],
                    file=f,
                )

        # music_video_args.whisper.transcriptions = transcriptions

        #tiny2large, large2tiny, whispers_tokens = whisper_align(whispers)
        # sanitize and tokenize
        whispers_tokens = {}
        for k in whispers:
            whispers_tokens[k] = [
            remove_punctuation(tok) for tok in whispers[k]['text'].split()
            ]

        # align sequences
        tiny2large, large2tiny = tokenizations.get_alignments(
            whispers_tokens[segmentation_model], #whispers_tokens['tiny'],
            whispers_tokens[transcription_model] #whispers_tokens['large']
        )
        #return tiny2large, large2tiny, whispers_tokens

        token_large_index_segmentations = whisper_transmit_meta_across_alignment(
            whispers,
            large2tiny,
            whispers_tokens,
        )
        prompt_starts = whisper_segment_transcription(
            token_large_index_segmentations,
        )

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

    rms = librosa.feature.rms(audio)[0]

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
