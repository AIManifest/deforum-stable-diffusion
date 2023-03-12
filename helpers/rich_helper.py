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

    table = Table(show_header=True, header_style="bold green", box=box.ROUNDED)
    table.add_column("Parameters", justify="left", style="green")
    table.add_column("Value", justify="left", style="yellow")
    
    table.add_row("Steps", str(int(keys.steps_schedule_series[frame_idx])))
    table.add_row("Scale", str(args.scale))
    table.add_row("Sampler", str(args.sampler))
    table.add_row("Angle", f"{keys.angle_series[frame_idx]}")
    table.add_row("Zoom", f"{keys.zoom_series[frame_idx]}")
    table.add_row("Tx", f"{keys.translation_x_series[frame_idx]}")
    table.add_row("Ty", f"{keys.translation_y_series[frame_idx]}")
    table.add_row("Tz", f"{keys.translation_z_series[frame_idx]}")
    table.add_row("Rx", f"{keys.rotation_3d_x_series[frame_idx]}")
    table.add_row("Ry", f"{keys.rotation_3d_y_series[frame_idx]}")
    table.add_row("Rz", f"{keys.rotation_3d_z_series[frame_idx]}")
    table.add_row("Rw", f"{keys.rotation_3d_w_series[frame_idx]}")
    table.add_row("Aspect Ratio", f"{keys.aspect_ratio_series[frame_idx]}")
    table.add_row("Near Plane", f"{keys.near_series[frame_idx]}")
    table.add_row("Far Plane", f"{keys.far_series[frame_idx]}")
    table.add_row("Field Of View", f"{keys.fov_series[frame_idx]}")
    table.add_row("Noise", f"{keys.noise_schedule_series[frame_idx]}")
    table.add_row("Strength", f"{keys.strength_schedule_series[frame_idx]}")
    table.add_row("Contrast", f"{keys.contrast_schedule_series[frame_idx]}")
    table.add_row("Kernel", str(int(keys.kernel_schedule_series[frame_idx])))
    table.add_row("Sigma", f"{keys.sigma_schedule_series[frame_idx]}")
    table.add_row("Amount", f"{keys.amount_schedule_series[frame_idx]}")
    table.add_row("Threshold", f"{keys.threshold_schedule_series[frame_idx]}")

    console.print(table)
