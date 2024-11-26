import argparse
import os
import torch
from collections import defaultdict
from torch.utils import data
import numpy as np
import pickle
import random
from tqdm import tqdm
from transformers import Wav2Vec2Processor
from sklearn.model_selection import train_test_split
import librosa
import json
import random
from weighted_sampler import WeightedSamplerBySpkID


def resample_vertices(vertices, L2):
    """
    Resample the vertex sequence to a new length L2 using linear interpolation.

    Parameters:
    vertices: numpy array of shape (L1, N, 3)
    L2: integer, the new length of the sequence

    Returns:
    new_vertices: numpy array of shape (L2, N, 3)
    """
    L1, N, _ = vertices.shape
    new_vertices = np.zeros((L2, N, 3))

    # Generate interpolation factors
    factors = np.linspace(0, L1 - 1, L2)

    # Perform linear interpolation for each vertex
    for i in range(N):
        for j in range(3):  # For x, y, z coordinates
            new_vertices[:, i, j] = np.interp(factors, np.arange(L1), vertices[:, i, j])

    return new_vertices


# class Dataset(data.Dataset):
#     """Custom data.Dataset compatible with data.DataLoader."""

#     def __init__(self, data, subjects_dict, data_type="train"):
#         self.data = data
#         self.len = len(self.data)
#         self.subjects_dict = subjects_dict
#         self.data_type = data_type
#         self.one_hot_labels = np.eye(len(subjects_dict["train"]))

#     def __getitem__(self, index):
#         """Returns one data pair (source and target)."""
#         file_name = self.data[index]["name"]
#         audio = self.data[index]["audio"]
#         vertice = self.data[index]["vertice"]
#         template = self.data[index]["template"]

#         if self.data_type == "train":
#             subject = "_".join(file_name.split("_")[:1])
#             one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
#         else:
#             one_hot = self.one_hot_labels

#         return torch.FloatTensor(audio), vertice, torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

#     def __len__(self):
#         return self.len


class Dataset(data.Dataset):
    def __init__(self, json_paths, data_root, data_type="train"):
        self.data = []
        self.processor = Wav2Vec2Processor.from_pretrained("/root/.cache/huggingface/hub/models--facebook--hubert-xlarge-ls960-ft/snapshots/86a09e67e0c8d074533992379242405825516eca")
        self.subjects_dict = {"train": [], "val": [], "test": []}
        self.data_type = data_type

        for json_path in json_paths:
            data_dict = json.load(open(json_path))
            for video_name, details in data_dict.items():
                audio_path = os.path.join(data_root, details["audio_path"])
                vertices_path = os.path.join(data_root, details["vertices_path"])
                template_path = os.path.join(data_root, details["template_save_path"])
                spk_id = details["speaker_id"]
                frames_num = details["frames_num"]
                self.data.append((video_name, audio_path, vertices_path, template_path, spk_id, frames_num))
                if spk_id not in self.subjects_dict["train"]:
                    self.subjects_dict["train"].append(spk_id)

        self.one_hot_labels = np.eye(len(self.subjects_dict["train"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_name, audio_path, vertices_path, template_path, spk_id, frames_num = self.data[idx]

        # 加载和处理数据
        vertices = np.load(vertices_path)
        temp = dict(torch.load(template_path))
        temp = temp[spk_id]["verts"].reshape((-1))

        speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
        input_values = np.squeeze(self.processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values)
        key = f"{video_name}.wav"

        if frames_num < 50:
            start_frame = 0
            end_frame = frames_num
        else:
            sampled_length = random.randint(50, min(250, frames_num))
            max_start_frame = frames_num - sampled_length
            start_frame = random.randint(0, max_start_frame)
            end_frame = start_frame + sampled_length

        audio_segment = input_values[int(start_frame / 25 * sampling_rate): int(end_frame / 25 * sampling_rate)]
        crop_vertices = vertices[start_frame: end_frame]
        # fps30_frame_num = int(crop_vertices.shape[0] / 25.0 * 30.0)
        # crop_vertices = resample_vertices(crop_vertices, fps30_frame_num)

        if self.data_type == "train":
            subject = spk_id
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels

        return torch.FloatTensor(audio_segment), crop_vertices, torch.FloatTensor(temp), torch.FloatTensor(one_hot), key


# def read_data(args):
    # print("Loading data...")
    # subjects_dict = {"train": [], "val": [], "test": []}
    # subjects_id = 0
    # data = defaultdict(dict)
    # json_list = args.json_list.split(" ")
    # processor = Wav2Vec2Processor.from_pretrained("/root/.cache/huggingface/hub/models--facebook--hubert-xlarge-ls960-ft/snapshots/86a09e67e0c8d074533992379242405825516eca")
    # for json_path in json_list:
    #     personalized_data_dict = json.load(open(json_path))
    #     for video_name in tqdm(personalized_data_dict.keys(), desc=f"Load from {os.path.basename(json_path)}"):
    #         spk_id = personalized_data_dict[video_name]["speaker_id"]
    #         vertices_path = os.path.join(args.data_root, personalized_data_dict[video_name]["vertices_path"])
    #         audio_path = os.path.join(args.data_root, personalized_data_dict[video_name]["audio_path"])
    #         template_path = os.path.join(args.data_root, personalized_data_dict[video_name]["template_save_path"])
    #         vertices = np.load(vertices_path)
    #         temp = dict(torch.load(template_path))
    #         temp = temp[spk_id]["verts"].reshape((-1))
    #         speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    #         input_values = np.squeeze(processor(speech_array, return_tensors="pt", padding="longest", sampling_rate=sampling_rate).input_values)
    #         audio_len = args.audio_len
    #         audios_num = int(input_values.size(0) / sampling_rate / audio_len) + 1

    #         for index in range(audios_num):
    #             key = f"{video_name}_{index}.wav"
    #             data[key]["name"] = key
    #             data[key]["audio"] = input_values[index * audio_len * sampling_rate: (index+1) * audio_len * sampling_rate]
    #             data[key]["template"] = temp
    #             crop_vertices = vertices[index * audio_len * 25: (index+1) * audio_len * 25]
    #             fps30_frame_num = int(crop_vertices.shape[0] / 25.0 * 30.0)
    #             crop_vertices = resample_vertices(crop_vertices, fps30_frame_num)
    #             data[key]["vertice"] = crop_vertices.reshape(crop_vertices.shape[0], -1)

    #         if spk_id not in subjects_dict["train"]:
    #             subjects_dict["train"].append(spk_id)

    # train_data = []
    # valid_data = []
    # test_data = []
    # for key in data:
    #     train_data.append(data[key])
    # print(len(train_data), len(valid_data), len(test_data))
    # return train_data, valid_data, test_data, subjects_dict


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloaders(args):
    g = torch.Generator()
    g.manual_seed(0)
    dataset = {}
    json_list = args.json_list.split(" ")
    train_data = Dataset(json_list, args.data_root, "train")
    weighted_sampler = WeightedSamplerBySpkID(train_data)
    dataset["train"] = data.DataLoader(
        dataset=train_data, 
        batch_size=1, 
        shuffle=False, 
        sampler=weighted_sampler, 
        worker_init_fn=seed_worker,
        num_workers=11,
        generator=g)
    dataset["valid"] = None
    dataset["test"] = None
    return dataset, " ".join(train_data.subjects_dict["train"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.0001, help='learning rate')
    parser.add_argument("--dataset", type=str, default="BIWI", help='Name of the dataset folder. eg: BIWI')
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--vertice_dim", type=int, default=70110, help='number of vertices - 23370*3 for BIWI dataset')
    parser.add_argument("--feature_dim", type=int, default=512, help='Latent Dimension to encode the inputs to')
    parser.add_argument("--gru_dim", type=int, default=512, help='GRU Vertex decoder hidden size')
    parser.add_argument("--gru_layers", type=int, default=2, help='GRU Vertex decoder hidden size')
    parser.add_argument("--wav_path", type=str, default="wav", help='path of the audio signals')
    parser.add_argument("--vertices_path", type=str, default="vertices_npy", help='path of the ground truth')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
    parser.add_argument("--max_epoch", type=int, default=50, help='number of epochs')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="face_diffuser", help='name of the trained model')
    parser.add_argument("--template_file", type=str, default="templates.pkl",
                        help='path of the train subject templates')
    parser.add_argument("--save_path", type=str, default="save", help='path of the trained models')
    parser.add_argument("--result_path", type=str, default="result", help='path to the predictions')
    parser.add_argument("--train_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--val_subjects", type=str, default="F2 F3 F4 M3 M4 M5")
    parser.add_argument("--test_subjects", type=str, default="F1 F2 F3 F4 F5 F6 F7 F8 M1 M2 M3 M4 M5 M6")
    parser.add_argument("--input_fps", type=int, default=50,
                        help='HuBERT last hidden state produces 50 fps audio representation')
    parser.add_argument("--output_fps", type=int, default=25,
                        help='fps of the visual data, BIWI was captured in 25 fps')
    parser.add_argument("--diff_steps", type=int, default=1000, help='number of diffusion steps')
    parser.add_argument("--skip_steps", type=int, default=0, help='number of diffusion steps to skip during inference')
    parser.add_argument("--num_samples", type=int, default=1, help='number of samples to generate per audio')
    parser.add_argument("--data_root", type=str, default="voice_dataset_root/voice_drive/train_data")
    parser.add_argument("--json_list", type=str)
    args = parser.parse_args()

    get_dataloaders(args)
