import librosa
import argparse
import torch
import trimesh
import numpy as np
import pandas as pd
import cv2
import os
import ffmpeg
import gc
import pyrender
from models import FaceDiff, FaceDiffBeat, FaceDiffDamm, MyFaceDiff
from transformers import Wav2Vec2Processor
import time

from utils import *
import json
import subprocess


def test_model(args):
    if not os.path.exists(args.result_path):
        os.makedirs(args.result_path)

    train_subjects_list = list(json.load(open(os.path.join(os.path.dirname(args.model_name), "spkid_dict.json"))).keys())
    args.train_subjects = " ".join(train_subjects_list)

    model = MyFaceDiff(
        args,
        vertice_dim=args.vertice_dim,
        latent_dim=args.feature_dim,
        diffusion_steps=args.diff_steps,
    )
    print(model)

    model.load_state_dict(torch.load(args.model_name, map_location='cuda'))
    model = model.to(torch.device(args.device))
    model.eval()

    data_dict = json.load(open(args.json_path))
    for video_name, details in data_dict.items():
        template_path = os.path.join(args.data_root, details["template_save_path"])
    template_path = os.path.join(os.path.dirname(template_path), "5_level.ckpt")
    template = dict(torch.load(template_path))
    template = template["5_level"]["verts"].reshape((-1))
    # template_file = os.path.join(args.data_path, args.dataset, args.template_path)
    # with open(template_file, 'rb') as fin:
    #     templates = pickle.load(fin, encoding='latin1')

    # train_subjects_list = [i for i in args.train_subjects.split(" ")]

    one_hot_labels = np.eye(len(train_subjects_list))

    iter = train_subjects_list.index(args.condition)
    one_hot = one_hot_labels[iter]
    one_hot = np.reshape(one_hot, (-1, one_hot.shape[0]))
    one_hot = torch.FloatTensor(one_hot).to(device=args.device)

    # if args.dataset in ["BIWI", "multiface", "vocaset"]:
    #     temp = templates[args.subject]
    # else:
    #     temp = np.zeros((args.vertice_dim // 3, 3))

    # template = temp.reshape((-1))
    template = np.reshape(template, (-1, template.shape[0]))
    template = torch.FloatTensor(template).to(device=args.device)

    wav_path = args.wav_path
    test_name = os.path.basename(wav_path).split(".")[0]
    start_time = time.time()
    speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
    processor = Wav2Vec2Processor.from_pretrained("huggingface/facebook/hubert-xlarge-ls960-ft")
    audio_feature = processor(speech_array, return_tensors="pt", padding="longest",
                              sampling_rate=sampling_rate).input_values
    # audio_feature = np.reshape(audio_feature, (-1, audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device=args.device)

    if args.guidance_scale != 1:
        speech_array_silent = np.zeros_like(speech_array)
        audio_feature_silent = processor(speech_array_silent, return_tensors="pt", padding="longest",
                                sampling_rate=sampling_rate).input_values
        audio_feature_silent = np.reshape(audio_feature_silent, (-1, audio_feature_silent.shape[0]))
        audio_feature_silent = torch.FloatTensor(audio_feature_silent).to(device=args.device)

    diffusion = create_gaussian_diffusion(args)

    # num_frames = int(audio_feature.shape[0] / sampling_rate * args.output_fps)
    num_frames = int(audio_feature.shape[1] / sampling_rate * args.output_fps)
    num_frames -= 1
    prediction = diffusion.p_sample_loop(
        model,
        (1, num_frames, args.vertice_dim),
        clip_denoised=False,
        model_kwargs={
            "cond_embed": audio_feature,
            "one_hot": one_hot,
            "template": template,
            "emo_id": torch.tensor([args.emotion_id]).to(device=args.device),
        },
        skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
        device="cuda"
    )
    if args.guidance_scale != 1:
        prediction_nocond = diffusion.p_sample_loop(
            model,
            (1, num_frames, args.vertice_dim),
            clip_denoised=False,
            model_kwargs={
                "cond_embed": audio_feature_silent,
                "one_hot": one_hot,
                "template": template,
                "emo_id": torch.tensor([args.emotion_id]).to(device=args.device),
            },
            skip_timesteps=args.skip_steps,  # 0 is the default value - i.e. don't skip any step
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
            device="cuda"
        )
        prediction = prediction_nocond + args.guidance_scale * (prediction - prediction_nocond)
    # prediction = prediction_nocond
    prediction = prediction.squeeze()
    prediction = prediction.detach().cpu().numpy()

    # scale back rig parameters for rendering in Maya
    if args.dataset == 'damm_rig_equal':
        with open('data/damm_rig_equal/scaler_192.pkl', 'rb') as f:
            RIG_SCALER = pickle.load(f)
        prediction = RIG_SCALER.inverse_transform(prediction)

    elapsed = time.time() - start_time
    print("Inference time for ", prediction.shape[0], " frames is: ", elapsed, " seconds.")
    print("Inference time for 1 frame is: ", elapsed / prediction.shape[0], " seconds.")
    print("Inference time for 1 second of audio is: ", ((elapsed * args.fps) / prediction.shape[0]), " seconds.")
    out_file_name = test_name + "_" + args.dataset + "_" + args.subject + "_condition_" + args.condition
    np.save(os.path.join(args.result_path, out_file_name), prediction)

    # save csv to be used directly for rendering in Maya
    if args.dataset == 'damm_rig_equal':
        df = pd.DataFrame(prediction)
        df.to_csv(os.path.join(args.result_path, f"{out_file_name}_Damm.csv"), index=None, header=None)

    # render mesh
    # subprocess.call(f"python run_audio2mesh.py --audio_path {wav_path} --verts_path {os.path.join(args.result_path, out_file_name)}.npy", shell=True)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="pretrained_BIWI")
    parser.add_argument("--data_path", type=str, default="data", help='name of the dataset folder. eg: BIWI')
    parser.add_argument("--dataset", type=str, default="BIWI", help='name of the dataset folder. eg: BIWI')
    parser.add_argument("--fps", type=float, default=25, help='frame rate - 25 for BIWI')
    parser.add_argument("--feature_dim", type=int, default=256, help='GRU Vertex Decoder hidden size')
    parser.add_argument("--vertice_dim", type=int, default=70110, help='number of vertices - 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--test_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--wav_path", type=str, default="demo/wav/test.wav",
                        help='path of the input audio signal in .wav format')
    parser.add_argument("--result_path", type=str, default="demo/result", help='path of the predictions in .npy format')
    parser.add_argument("--condition", type=str, default="M3", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="M1",
                        help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--template_path", type=str, default="templates.pkl",
                        help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates",
                        help='path of the mesh in BIWI topology')
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--emotion", type=int, default="1",
                        help='style control for emotion, 1 for expressive animation, 0 for neutral animation')
    parser.add_argument("--diff_steps", type=int, default=1000)
    parser.add_argument("--device_idx", type=int, default=0)
    parser.add_argument("--gru_dim", type=int, default=512)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--skip_steps", type=int, default=0)
    parser.add_argument("--data_root", type=str, default="voice_dataset_root/voice_drive/train_data")
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--emotion_id", type=int, default=5)
    args = parser.parse_args()

    test_model(args)


if __name__ == "__main__":
    main()