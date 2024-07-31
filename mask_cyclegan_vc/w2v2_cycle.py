import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from torch.utils.data.dataset import Dataset

import numpy as np
import pandas as pd
import random

import os
import glob
import pickle
from tqdm import tqdm

from mask_cyclegan_vc.model import Generator, Discriminator, GLU
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config).eval().requires_grad_(False)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)

        return hidden_states

    def get_pad(self, input_values):
        return self.classifier(input_values)


class TrainDataPreparing(Dataset):
    def __init__(self, n_frames=64, device="cuda:0"):
        self.vocoder = torch.hub.load('LewisGet/melgan-neurips', 'load_melgan')
        self.wav2vec2 = EmotionModel.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim').to(device).eval().requires_grad_(False)
        self.n_frames = n_frames
        self.device = device

        self.init_clean_dataset()

    def mel2wav(self, mel, labels="a", device=None):
        mean = self.a_mean
        std = self.a_std

        if labels == "b":
            mean = self.b_mean
            std = self.b_std

        if type(mel) == torch.Tensor:
            std = torch.tensor(std, dtype=torch.float32)
            mean = torch.tensor(mean, dtype=torch.float32)

            if device is None:
                std = std.to(self.device)
                mean = mean.to(self.device)

            if device == "cpu":
                std = std.cpu()
                mean = mean.cpu()

        rev = mel * std + mean

        return self.vocoder.inverse(rev.unsqueeze(0))

    def init_clean_dataset(self):
        self.a_mels = []
        self.b_mels = []
        self.a_cut_mels = []
        self.b_cut_mels = []
        self.a_mean = None
        self.a_std = None
        self.b_mean = None
        self.b_std = None

    def save_dataset(self, path="./cache"):
        if not os.path.exists(path):
            os.makedirs(path)

        pickle.dump(self.a_mels, open(os.path.join(path, "a_mels.pkl"), "wb"))
        pickle.dump(self.b_mels, open(os.path.join(path, "b_mels.pkl"), "wb"))

        np.save(os.path.join(path, "a_mean.npy"), self.a_mean)
        np.save(os.path.join(path, "a_std.npy"), self.a_std)

        np.save(os.path.join(path, "b_mean.npy"), self.b_mean)
        np.save(os.path.join(path, "b_std.npy"), self.b_std)

    def load_dataset(self, path="./cache"):
        self.a_mels = pickle.load(open(os.path.join(path, "a_mels.pkl"), "rb"))
        self.b_mels = pickle.load(open(os.path.join(path, "b_mels.pkl"), "rb"))

        self.a_mean = np.load(os.path.join(path, "a_mean.npy"))
        self.a_std = np.load(os.path.join(path, "a_std.npy"))

        self.b_mean = np.load(os.path.join(path, "b_mean.npy"))
        self.b_std = np.load(os.path.join(path, "b_std.npy"))

        self.cut_mels()

    def prepare_mels(self, paths):
        mels = []

        for path in tqdm(paths, leave=False, desc="wavs to mels"):
            _file = open(path, "rb")
            wav, sample_rate = torchaudio.load(_file)
            _file.close()

            mel = self.vocoder(wav)

            if mel.shape[-1] < self.n_frames:
                continue

            mels.append(mel.detach().cpu().numpy()[0])

        mel_join = np.concatenate(mels, axis=1)
        mel_mean = np.mean(mel_join, axis=1, keepdims=True)
        mel_std = np.std(mel_join, axis=1, keepdims=True) + 1e-9

        mel_normalized = []

        for mel in mels:
            mel_normalized.append((mel - mel_mean) / mel_std)

        return mel_normalized, mel_mean, mel_std

    def flush_dataset(self, a_paths, b_paths, path_fixed=None):
        if path_fixed is not None:
            a_paths = path_fixed(a_paths)
            b_paths = path_fixed(b_paths)

        self.a_mels, self.a_mean, self.a_std = self.prepare_mels(a_paths)
        self.b_mels, self.b_mean, self.b_std = self.prepare_mels(b_paths)

    def cut_mels(self):
        for a_mel in self.a_mels:
            if a_mel.shape[-1] < self.n_frames:
                continue

            if a_mel.shape[-1] == self.n_frames:
                self.a_cut_mels.append(a_mel)
                continue

            if a_mel.shape[-1] > self.n_frames:
                start = np.random.randint(0, a_mel.shape[-1] - self.n_frames + 1)
                end = start + self.n_frames
                self.a_cut_mels.append(a_mel[:, start: end])

        for b_mel in self.b_mels:
            if b_mel.shape[-1] < self.n_frames:
                continue

            if b_mel.shape[-1] == self.n_frames:
                self.b_cut_mels.append(b_mel)

            if b_mel.shape[-1] > self.n_frames:
                start = np.random.randint(0, b_mel.shape[-1] - self.n_frames + 1)
                end = start + self.n_frames
                self.b_cut_mels.append(b_mel[:, start: end])

    def __getitem__(self, index):
        return self.a_cut_mels[index], self.b_cut_mels[index]

    def __len__(self):
        return min(len(self.a_cut_mels), len(self.b_cut_mels))


class MyGenerator(Generator):
    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super().__init__()
        self.vec = nn.Linear(3, input_shape[0] * input_shape[1])

    def forward(self, x, y):
        y = self.vec(y)
        y = y.reshape(-1, 80, 64)
        x = torch.stack((x, y), dim=1)
        conv1 = self.conv1(x) * torch.sigmoid(self.conv1_gates(x))

        ## same code after

        # Downsampling
        downsample1 = self.downSample1(conv1)
        downsample2 = self.downSample2(downsample1)

        # Reshape
        reshape2dto1d = downsample2.view(
            downsample2.size(0), self.flattened_channels, 1, -1)
        reshape2dto1d = reshape2dto1d.squeeze(2)

        # 2D -> 1D
        conv2dto1d_layer = self.conv2dto1dLayer(reshape2dto1d)
        conv2dto1d_layer = self.conv2dto1dLayer_tfan(conv2dto1d_layer)

        # Residual Blocks
        residual_layer_1 = self.residualLayer1(conv2dto1d_layer)
        residual_layer_2 = self.residualLayer2(residual_layer_1)
        residual_layer_3 = self.residualLayer3(residual_layer_2)
        residual_layer_4 = self.residualLayer4(residual_layer_3)
        residual_layer_5 = self.residualLayer5(residual_layer_4)
        residual_layer_6 = self.residualLayer6(residual_layer_5)

        # 1D -> 2D
        conv1dto2d_layer = self.conv1dto2dLayer(residual_layer_6)
        conv1dto2d_layer = self.conv1dto2dLayer_tfan(conv1dto2d_layer)

        # Reshape
        reshape1dto2d = conv1dto2d_layer.unsqueeze(2)
        reshape1dto2d = reshape1dto2d.view(reshape1dto2d.size(0), 256, 20, -1)

        # UpSampling
        upsample_layer_1 = self.upSample1(reshape1dto2d)
        upsample_layer_2 = self.upSample2(upsample_layer_1)

        # Conv2d
        output = self.lastConvLayer(upsample_layer_2)
        output = output.squeeze(1)
        return output


class MyDiscriminator(Discriminator):
    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super().__init__()

        self.convLayer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=residual_in_channels // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1)
            ),
            GLU()
        )
        self.input_shape = input_shape
        self.vec = nn.Linear(3, self.input_shape[0] * self.input_shape[1])

    def forward(self, x, y):
        y = self.vec(y)
        y = y.reshape(-1, 1, self.input_shape[0], self.input_shape[1])
        x = x.unsqueeze(1)
        x = torch.cat((x, y), dim=1)
        conv_layer_1 = self.convLayer1(x)
        downsample1 = self.downSample1(conv_layer_1)
        downsample2 = self.downSample2(downsample1)
        downsample3 = self.downSample3(downsample2)
        output = torch.sigmoid(self.outputConvLayer(downsample3))
        return output


class W2V2PADTraining(object):
    def __init__(self, args, default_dataset_path="./cache"):
        self.args = args

        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.discriminator_lr = args.discriminator_lr
        self.mini_batch_size = args.batch_size
        self.device = args.device
        self.epochs_per_save = args.epochs_per_save
        self.epochs_per_plot = args.epochs_per_plot
        self.sample_rate = args.sample_rate

        self.dataset = TrainDataPreparing()
        self.dataset.load_dataset(default_dataset_path)
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.mini_batch_size, shuffle=True)

        self.g_a2b = MyGenerator().to(self.device)
        self.g_b2a = MyGenerator().to(self.device)
        self.g_a_vec = MyGenerator().to(self.device)
        self.g_b_vec = MyGenerator().to(self.device)
        self.d_a = Discriminator().to(self.device)
        self.d_b = Discriminator().to(self.device)

        g_params = list(self.g_a2b.parameters()) + list(self.g_b2a.parameters()) + list(self.g_a_vec.parameters()) + list(self.g_b_vec.parameters())
        d_params = list(self.d_a.parameters()) + list(self.d_b.parameters())

        self.g_optimizer = torch.optim.Adam(g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        self.saver = ModelSaver(args)
        self.logger = TrainLogger(args, len(self.dataset))

    def model_collapse_modifies_lr(self):
        for i, v in enumerate(self.g_optimizer.param_groups):
            self.g_optimizer.param_groups[i]['lr'] = v['lr'] * 1.25

        for i, v in enumerate(self.d_optimizer.param_groups):
            self.d_optimizer.param_groups[i]['lr'] = v['lr'] * 1.25

    def modify_model_collapse_epoch(self):
        return self.logger.epoch % 70 == 0 and self.logger.epoch != 0

    def mel2wav2vec(self, mel, labels="a"):
        with torch.no_grad():
            self.dataset.wav2vec2.eval()
            wav = self.dataset.mel2wav(mel, labels)
            vec = self.dataset.wav2vec2(wav)
            return vec

    def vec2pad(self, vec):
        with torch.no_grad():
            self.dataset.wav2vec2.eval()
            pad = self.dataset.wav2vec2.get_pad(vec)
            return pad

    def mels2wavs2vec2pad(self, mels, labels="a"):
        vecs = []
        pads = []
        with torch.no_grad():
            for mel in mels:
                vecs.append(self.mel2wav2vec(mel, labels))
                pads.append(self.vec2pad(vecs[-1]))
        return torch.tensor(vecs, dtype=torch.float32, device=self.device), torch.tensor(pads, dtype=torch.float32, device=self.device)

    def mel_vec_loss(self, labels, x_mel, f_mel, modifiy):
        loss = 0

        for x, f in zip(x_mel, f_mel):
            x = x.detach().cpu()
            f = f.detach().cpu()

            with torch.no_grad():
                self.dataset.wav2vec2.eval()

                x_wav = self.dataset.mel2wav(x, labels)
                f_wav = self.dataset.mel2wav(f, labels)

                x_vec = self.dataset.wav2vec2(x_wav).detach()
                f_vec = self.dataset.wav2vec2(f_wav).detach()

                loss += torch.mean((x_vec - f_vec) ** 2)

                del x_wav, f_wav, x_vec, f_vec
                torch.cuda.empty_cache()

        return loss

    def true_loss(self, a, b):
        return torch.mean((1 - a) ** 2 + (1 - b) ** 2)

    def false_loss(self, a, b):
        return torch.mean((0 - a) ** 2 + (0 - b) ** 2)

    def pre_train_g(self):
        self.g_a2b.train()
        self.g_b2a.train()
        self.g_a_vec.train()
        self.g_b_vec.train()
        self.d_a.eval()
        self.d_b.eval()

    def pre_train_d(self):
        self.g_a2b.eval()
        self.g_b2a.eval()
        self.g_a_vec.eval()
        self.g_b_vec.eval()
        self.d_a.train()
        self.d_b.train()

    def train_one_mel(self, a_mel, b_mel, pad_change, none_change):
        with torch.set_grad_enabled(True):
            self.pre_train_g()
            self.a_mel = a_mel.unsqueeze(0).to(self.device, dtype=torch.float32)
            self.b_mel = b_mel.unsqueeze(0).to(self.device, dtype=torch.float32)

            # 音色不修改 pad
            self.fake_b = self.g_a2b(self.a_mel.clone(), none_change)
            self.fake_a = self.g_b2a(self.b_mel.clone(), none_change)
            self.fake_ba = self.g_b2a(self.fake_b, none_change)
            self.fake_ab = self.g_a2b(self.fake_a, none_change)
            self.test_b = self.g_a2b(torch.ones_like(self.a_mel), none_change)
            self.test_a = self.g_b2a(torch.ones_like(self.b_mel), none_change)

            # 修改 pad
            self.change_a = self.g_a_vec(self.a_mel.clone(), pad_change)
            self.change_b = self.g_b_vec(self.b_mel.clone(), pad_change)

            fidelity = self.true_loss(self.d_a(self.fake_a), self.d_b(self.fake_b))
            fidelity_pow = self.true_loss(self.d_a(self.fake_ab), self.d_b(self.fake_ba))
            fidelity_empty = self.true_loss(self.d_a(self.test_a), self.d_b(self.test_b))
            fidelity_change = self.true_loss(self.d_a(self.change_a), self.d_b(self.change_b))

        # pad loss
        with torch.set_grad_enabled(False):
            _a = torch.tensor(self.a_mel[0]).to(self.device, dtype=torch.float32)
            _b = torch.tensor(self.b_mel[0]).to(self.device, dtype=torch.float32)

            a_vec = self.mel2wav2vec(_a, "a")
            b_vec = self.mel2wav2vec(_b, "b")

            a_pad = self.vec2pad(a_vec)
            b_pad = self.vec2pad(b_vec)

            del a_vec, b_vec, _a, _b
            torch.cuda.empty_cache()

            _change_a = torch.tensor(self.change_a[0]).to(self.device, dtype=torch.float32)
            _change_b = torch.tensor(self.change_b[0]).to(self.device, dtype=torch.float32)

            change_a_vec = self.mel2wav2vec(_change_a, "a")
            change_b_vec = self.mel2wav2vec(_change_b, "b")

            change_a_pad = self.vec2pad(change_a_vec)
            change_b_pad = self.vec2pad(change_b_vec)

            del change_a_vec, change_b_vec, _change_a, _change_b
            torch.cuda.empty_cache()

        with torch.set_grad_enabled(True):
            f_a_pad_change = a_pad + pad_change
            f_b_pad_change = b_pad + pad_change

            pad_loss = torch.mean((f_a_pad_change - change_a_pad) ** 2 + (f_b_pad_change - change_b_pad) ** 2)

            del a_pad, b_pad, change_a_pad, change_b_pad
            torch.cuda.empty_cache()

            if self.modify_model_collapse_epoch():
                pad_loss = pad_loss * 1.5
                fidelity = fidelity * 0.8
                fidelity_pow = fidelity_pow * 0.8
                fidelity_empty = fidelity_empty * 0.8
                fidelity_change = fidelity_change * 0.8

            g_loss = fidelity + fidelity_pow + fidelity_empty + fidelity_change + pad_loss * 10.0

            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()

            # d train

            self.pre_train_d()
            d_real_loss = self.true_loss(self.d_a(self.a_mel.clone().detach()), self.d_b(self.b_mel.clone().detach()))
            d_fake_loss = self.false_loss(self.d_a(self.fake_a.detach()), self.d_b(self.fake_b.detach()))
            d_change_loss = self.false_loss(self.d_a(self.change_a.detach()), self.d_b(self.change_b.detach()))

            if self.modify_model_collapse_epoch():
                d_real_loss = d_real_loss * 1.5
                d_fake_loss = d_fake_loss * 0.8
                d_change_loss = d_change_loss * 0.8

            d_loss = d_real_loss + d_fake_loss + d_change_loss

            self.d_optimizer.zero_grad()
            d_loss.backward()
            self.d_optimizer.step()

        self.logger.log_iter(
            loss_dict={
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "fidelity": fidelity.item(),
                "fidelity_pow": fidelity_pow.item(),
                "fidelity_empty": fidelity_empty.item(),
                "fidelity_change": fidelity_change.item(),
                "pad_loss": pad_loss.item(),
                "d_real_loss": d_real_loss.item(),
                "d_fake_loss": d_fake_loss.item(),
                "d_change_loss": d_change_loss.item()
            }
        )

    def log_mel(self, mel, labels, name):
        print(mel.shape)
        mel = mel.detach()
        print(mel.shape)
        self.logger.log_img(mel.unsqueeze(0), f"mel_{name}_{labels}")

        wav = self.dataset.mel2wav(mel.to(self.device), labels).cpu()
        self.logger.log_audio(wav.reshape(1, int(16000 * 1.024)).T, f"wav_{name}_{labels}", 16000)

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.start_epoch()

            for i, (a_mels, b_mels) in enumerate(tqdm(self.dataloader)):
                self.logger.start_iter()

                a_mels = a_mels.to(self.device, dtype=torch.float32)
                b_mels = b_mels.to(self.device, dtype=torch.float32)

                pad_change = torch.tensor([random.random() * 2 - 1 for i in range(3)], dtype=torch.float32, device=self.device)
                none_change = torch.zeros_like(pad_change, dtype=torch.float32, device=self.device)

                for batch_idx, (a_mel, b_mel) in enumerate(zip(a_mels, b_mels)):
                    self.train_one_mel(a_mel, b_mel, pad_change, none_change)

                self.logger.end_iter()

            if self.logger.epoch % self.epochs_per_save == 0:
                self.saver.save(self.logger.epoch, self.g_a2b, self.g_optimizer, None, args.device, "g_a2b")
                self.saver.save(self.logger.epoch, self.g_b2a, self.g_optimizer, None, args.device, "g_b2a")
                self.saver.save(self.logger.epoch, self.g_a_vec, self.g_optimizer, None, args.device, "g_a_vec")
                self.saver.save(self.logger.epoch, self.g_b_vec, self.g_optimizer, None, args.device, "g_b_vec")
                self.saver.save(self.logger.epoch, self.d_a, self.d_optimizer, None, args.device, "d_a")
                self.saver.save(self.logger.epoch, self.d_b, self.d_optimizer, None, args.device, "d_b")

            if self.logger.epoch % self.epochs_per_plot == 0:
                self.g_a2b.eval()
                self.g_b2a.eval()
                self.g_a_vec.eval()
                self.g_b_vec.eval()
                self.d_a.eval()
                self.d_b.eval()

                self.log_mel(self.a_mel[0], "a", "real")
                self.log_mel(self.b_mel[0], "b", "real")
                self.log_mel(self.fake_a[0], "a", "fake")
                self.log_mel(self.fake_b[0], "b", "fake")
                self.log_mel(self.fake_ba[0], "a", "pow")
                self.log_mel(self.fake_ab[0], "b", "pow")
                self.log_mel(self.change_a[0], "a", "change")
                self.log_mel(self.change_b[0], "b", "change")

            if self.modify_model_collapse_epoch():
                self.model_collapse_modifies_lr()

            self.logger.end_epoch()


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()

    # prepar data

    prepar_data = TrainDataPreparing()
    # load by pad csv
    # prepar_data.flush_dataset("/var/classifed/kevin_emotion_sources.csv", "/var/classifed/lewis_emotion_sources.csv", path_fixed=lambda x: os.path.join("/", "var", x))
    prepar_data.flush_dataset(glob.glob("/var/16k_data/*a*/*.wav"), glob.glob("/var/16k_data/*b*/*.wav"))
    prepar_data.save_dataset("kevin_a_lewis_b")

    del prepar_data

    w2v2pad = W2V2PADTraining(args, default_dataset_path="kevin_a_lewis_b")
    w2v2pad.train()
