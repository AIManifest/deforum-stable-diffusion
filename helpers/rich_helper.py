from .animation import DeformAnimKeys

import rich
from rich.console import Console
from rich.table import Table
from rich.style import Style
from rich.color import Color
from rich import box

console = Console()

def print_animation_table(args, anim_args, keys, frame_idx):
    keys = DeformAnimKeys(anim_args)
    console = Console()

    table1 = Table(show_header=True, header_style="bold green", box=box.ASCII_DOUBLE_HEAD)
    table1.add_column("Steps", justify="left")
    table1.add_column("Scale", justify="left")
    table1.add_column("Sampler", justify="left")
    table1.add_column("Angle", justify="left")
    table1.add_column("Zoom", justify="left")
    table1.add_column("Tx", justify="left")
    table1.add_column("Ty", justify="left")
    table1.add_column("Tz", justify="left")
    table1.add_column("Rx", justify="left")
    table1.add_column("Ry", justify="left")
    table1.add_column("Rz", justify="left")
    table1.add_column("Rw", justify="left")
    table1.add_column("Aspect Ratio", justify="left")

    table2 = Table(show_header=True, header_style="bold green", box=box.ASCII_DOUBLE_HEAD)
    table2.add_column("Near Plane", justify="left")
    table2.add_column("Far Plane", justify="left")
    table2.add_column("Field Of View", justify="left")
    table2.add_column("Noise", justify="left")
    table2.add_column("Strength", justify="left")
    table2.add_column("Contrast", justify="left")
    table2.add_column("Kernel", justify="left")
    table2.add_column("Sigma", justify="left")
    table2.add_column("Amount", justify="left")
    table2.add_column("Threshold", justify="left")

    table1.add_row(str(int(keys.steps_schedule_series[frame_idx])),
                  str(args.scale),
                  str(args.sampler),
                  f"{keys.angle_series[frame_idx]}",
                  f"{keys.zoom_series[frame_idx]}",
                  f"{keys.translation_x_series[frame_idx]}",
                  f"{keys.translation_y_series[frame_idx]}",
                  f"{keys.translation_z_series[frame_idx]}",
                  f"{keys.rotation_3d_x_series[frame_idx]}",
                  f"{keys.rotation_3d_y_series[frame_idx]}",
                  f"{keys.rotation_3d_z_series[frame_idx]}",
                  f"{keys.rotation_3d_w_series[frame_idx]}",
                  f"{keys.aspect_ratio_series[frame_idx]}")

    table2.add_row(f"{keys.near_series[frame_idx]}",
                  f"{keys.far_series[frame_idx]}",
                  f"{keys.fov_series[frame_idx]}",
                  f"{keys.noise_schedule_series[frame_idx]}",
                  f"{keys.strength_schedule_series[frame_idx]}",
                  f"{keys.contrast_schedule_series[frame_idx]}",
                  str(int(keys.kernel_schedule_series[frame_idx])),
                  f"{keys.sigma_schedule_series[frame_idx]}",
                  f"{keys.amount_schedule_series[frame_idx]}",
                  f"{keys.threshold_schedule_series[frame_idx]}")
                  
    console.print(table1)
    console.print(table2)
