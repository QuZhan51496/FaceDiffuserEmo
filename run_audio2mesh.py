import argparse
import os
import sys
from pixelai_audio2exp.registry import VISUALIZERS
import librosa
import torch
import numpy as np
import tempfile
import cv2
import subprocess
from tqdm import tqdm
import torch.nn as nn
import json
import torch
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default=None, required=True)
    parser.add_argument("--verts_path", type=str, default=None, required=True)
    parser.add_argument("--save_path", type=str, default="demo/vis_mesh")
    args = parser.parse_args()

    save_dir = args.save_path
    os.makedirs(save_dir, exist_ok=True)

    audio_name = os.path.splitext(os.path.basename(args.audio_path))[0]
    verts_name = os.path.splitext(os.path.basename(args.verts_path))[0]
    video_path = os.path.join(save_dir, f"{verts_name}_{verts_name}.mp4")

    # model & renderer build
    visualizer_dict = dict(
        type="PYRenderVisualizer",
        name="vis",
        img_size=800,
    )
    renderer = VISUALIZERS.build(visualizer_dict)

    # render
    temp_video_pred = tempfile.NamedTemporaryFile(
        "w", suffix=".mp4", dir=os.path.dirname(video_path)
    )
    writer_pred = cv2.VideoWriter(
        temp_video_pred.name,
        cv2.VideoWriter_fourcc(*'mp4v'),
        30,
        (800, 800),
        True,
    )
    K = np.array([[2377.485, 0, 400], [0, 2377.485, 400], [0, 0, 1]])
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    T = np.array([0, 0, 1])
    renderer.set_camera(K, R, T)

    prediction = np.load(args.verts_path)
    prediction = prediction.reshape(-1, 5023, 3)
    for verts in tqdm(prediction):
        render_img, mask = renderer.render(verts)
        writer_pred.write(render_img[:, :, :3].astype(np.uint8))

    writer_pred.release()

    print(f"Write Video to {video_path}")
    cmd = f"ffmpeg -i {temp_video_pred.name} -i {args.audio_path} -c:v libx264 -c:a aac -qscale 0 {video_path} -y"
    subprocess.call(cmd, shell=True)
