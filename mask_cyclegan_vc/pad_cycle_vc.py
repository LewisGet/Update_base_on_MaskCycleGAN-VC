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
import pickle
from tqdm import tqdm

from mask_cyclegan_vc.model import Generator, Discriminator, GLU
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)

        return hidden_states


class TrainDataPreparing(Dataset):
    def __init__(self, n_frames=64, device="cuda:0"):
        self.vocoder = torch.hub.load('LewisGet/melgan-neurips', 'load_melgan')
        self.wav2vec2 = EmotionModel.from_pretrained('audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim').to(device).eval().requires_grad_(False)
        self.n_frames = n_frames
        self.device = device

        self.init_clean_dataset()

    def mel2wav(self, mel, std, mean):
        rev = mel * std + mean

        return self.vocoder.inverse(rev.unsqueeze(0))

    def init_clean_dataset(self):
        self.a_mels = []
        self.b_mels = []
        self.a_cut_mels = []
        self.b_cut_mels = []
        self.a_vecs = []
        self.b_vecs = []
        self.a_mean = None
        self.a_std = None
        self.b_mean = None
        self.b_std = None

    def save_dataset(self, path="./cache"):
        pickle.dump(self.a_mels, open(os.path.join(path, "a_mels.pkl"), "wb"))
        pickle.dump(self.b_mels, open(os.path.join(path, "b_mels.pkl"), "wb"))

        np.save(os.path.join(path, "a_vecs.npy"), self.a_vecs)
        np.save(os.path.join(path, "b_vecs.npy"), self.b_vecs)

        np.save(os.path.join(path, "a_mean.npy"), self.a_mean)
        np.save(os.path.join(path, "a_std.npy"), self.a_std)

        np.save(os.path.join(path, "b_mean.npy"), self.b_mean)
        np.save(os.path.join(path, "b_std.npy"), self.b_std)

    def load_dataset(self, path="./cache"):
        self.a_mels = pickle.load(open(os.path.join(path, "a_mels.pkl"), "rb"))
        self.b_mels = pickle.load(open(os.path.join(path, "b_mels.pkl"), "rb"))

        self.a_vecs = np.load(os.path.join(path, "a_vecs.npy"))
        self.b_vecs = np.load(os.path.join(path, "b_vecs.npy"))

        self.a_mean = np.load(os.path.join(path, "a_mean.npy"))
        self.a_std = np.load(os.path.join(path, "a_std.npy"))

        self.b_mean = np.load(os.path.join(path, "b_mean.npy"))
        self.b_std = np.load(os.path.join(path, "b_std.npy"))

        self.cut_mels()

    def prepare_org_dataset(self, paths):
        mels = []
        vecs = []

        for path in tqdm(paths, leave=False, desc="wavs to mels"):
            _file = open(path, "rb")
            wav, sample_rate = torchaudio.load(_file)
            _file.close()

            mel = self.vocoder(wav)

            if mel.shape[-1] < self.n_frames:
                continue

            mels.append(mel.detach().cpu().numpy()[0])
            vecs.append(self.wav2vec2(wav.to(self.device)).detach().cpu().numpy())

        mel_join = np.concatenate(mels, axis=1)
        mel_mean = np.mean(mel_join, axis=1, keepdims=True)
        mel_std = np.std(mel_join, axis=1, keepdims=True) + 1e-9

        mel_normalized = []

        for mel in mels:
            mel_normalized.append((mel - mel_mean) / mel_std)

        return mel_normalized, np.array(_vecs), mel_mean, mel_std

    def flush_dataset(self, a_paths, b_paths, path_fixed=None):
        if path_fixed is not None:
            a_paths = path_fixed(a_paths)
            b_paths = path_fixed(b_paths)

        self.a_mels, self.a_vecs, self.a_mean, self.a_std = self.prepare_mels(a_paths, a_vecs)
        self.b_mels, self.b_vecs, self.b_mean, self.b_std = self.prepare_mels(b_paths, b_vecs)

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
        return self.a_cut_mels[index], self.b_cut_mels[index], self.a_vecs[index], self.b_vecs[index]

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


class PADTraining(object):
    def __init__(self, args):
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
        self.dataset.load_dataset()
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.mini_batch_size, shuffle=True)

        self.g_a2b = MyGenerator().to(self.device)
        self.g_b2a = MyGenerator().to(self.device)
        self.d_a = MyDiscriminator().to(self.device)
        self.d_b = MyDiscriminator().to(self.device)

        g_params = list(self.g_a2b.parameters()) + list(self.g_b2a.parameters())
        d_params = list(self.d_a.parameters()) + list(self.d_b.parameters())

        self.g_optimizer = torch.optim.Adam(g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        self.saver = ModelSaver(args)
        self.logger = TrainLogger(args, len(self.dataset))

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.start_epoch()

            for i, (a_mels, b_mels, a_vecs, b_vecs) in enumerate(tqdm(self.dataloader)):
                self.logger.start_iter()

                a_mels = a_mels.to(self.device, dtype=torch.float32)
                b_mels = b_mels.to(self.device, dtype=torch.float32)
                a_vecs = a_vecs.to(self.device, dtype=torch.float32)
                b_vecs = b_vecs.to(self.device, dtype=torch.float32)

                with torch.set_grad_enabled(True):
                    self.g_a2b.train()
                    self.g_b2a.train()
                    self.d_a.eval()
                    self.d_b.eval()

                    # 使用目標的 vec 訓練生成方向控制
                    fake_b = self.g_a2b(a_mels, b_vecs)
                    fake_a = self.g_b2a(b_mels, a_vecs)
                    fake_ba = self.g_b2a(fake_b, a_vecs)
                    fake_ab = self.g_a2b(fake_a, b_vecs)
                    test_b = self.g_a2b(torch.ones_like(a_mels), b_vecs)
                    test_a = self.g_b2a(torch.ones_like(b_mels), a_vecs)

                    # 訓練非資料內的嘗試
                    test_b_rise = self.g_a2b(torch.ones_like(a_mels), b_vecs * 1.1)
                    test_b_drop = self.g_a2b(torch.ones_like(a_mels), b_vecs * 0.9)
                    test_a_rise = self.g_b2a(torch.ones_like(b_mels), a_vecs * 1.1)
                    test_a_drop = self.g_b2a(torch.ones_like(b_mels), a_vecs * 0.9)

                    d_fake_b = self.d_b(fake_b, b_vecs)
                    d_fake_a = self.d_a(fake_a, a_vecs)
                    d_fake_ba = self.d_b(fake_ba, a_vecs)
                    d_fake_ab = self.d_a(fake_ab, b_vecs)
                    d_test_b = self.d_b(test_b, b_vecs)
                    d_test_a = self.d_a(test_a, a_vecs)
                    d_test_b_rise = self.d_b(test_b_rise, b_vecs * 1.1)
                    d_test_b_drop = self.d_b(test_b_drop, b_vecs * 0.9)
                    d_test_a_rise = self.d_a(test_a_rise, a_vecs * 1.1)
                    d_test_a_drop = self.d_a(test_a_drop, a_vecs * 0.9)

                    g_fake_loss = torch.mean((1 - d_fake_b) ** 2 + (1 - d_fake_a) ** 2)
                    g_real_loss = torch.mean((1 - d_test_b) ** 2 + (1 - d_test_a) ** 2)
                    g_fake_pow_loss = torch.mean((1 - d_fake_ba) ** 2 + (1 - d_fake_ab) ** 2)
                    g_rise_loss = torch.mean((1 - d_test_b_rise) ** 2 + (1 - d_test_a_rise) ** 2)
                    g_drop_loss = torch.mean((1 - d_test_b_drop) ** 2 + (1 - d_test_a_drop) ** 2)

                    g_loss = g_fake_loss + g_real_loss + g_fake_pow_loss + g_rise_loss + g_drop_loss

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                    self.g_a2b.eval()
                    self.g_b2a.eval()
                    self.d_a.train()
                    self.d_b.train()

                    # 訓練音色權重
                    d_real_a = self.d_a(a_mels, a_vecs)
                    d_real_b = self.d_b(b_mels, b_vecs)
                    d_fake_a = self.d_a(fake_a.detach(), a_vecs)
                    d_fake_b = self.d_b(fake_b.detach(), b_vecs)


                    # 訓練 vec 權重
                    d_real_not_a = self.d_a(a_mels, b_vecs)
                    d_real_not_b = self.d_b(b_mels, a_vecs)

                    d_fake_not_vec_a = self.d_a(fake_a.detach(), b_vecs)
                    d_fake_not_vec_b = self.d_b(fake_b.detach(), a_vecs)

                    randn = random.random() * 2
                    d_vec_not_match_a_rand = self.d_a(a_mels, a_vecs * randn)
                    d_vec_not_match_b_rand = self.d_b(b_mels, b_vecs * randn)


                    # 音色 loss
                    d_real_loss = torch.mean((1 - d_real_a) ** 2 + (1 - d_real_b) ** 2) * 10.0
                    d_fake_loss = torch.mean((0 - d_fake_a) ** 2 + (0 - d_fake_b) ** 2)

                    # vec loss
                    d_real_not_match_loss = torch.mean((0 - d_real_not_a) ** 2 + (0 - d_real_not_b) ** 2) * 10.0
                    d_fake_not_match_loss = torch.mean((0 - d_fake_not_vec_a) ** 2 + (0 - d_fake_not_vec_b) ** 2)
                    d_vec_not_match_loss = torch.mean((0 - d_vec_not_match_a_rand) ** 2 + (0 - d_vec_not_match_b_rand) ** 2)

                    d_loss = d_real_loss + d_fake_loss + d_real_not_match_loss + d_fake_not_match_loss + d_vec_not_match_loss

                    self.d_optimizer.zero_grad()
                    d_loss.backward()
                    self.d_optimizer.step()

                self.logger.log_iter(
                    loss_dict={
                        "g_loss": g_loss.item(),
                        "d_loss": d_loss.item(),
                        "d_real_loss": d_real_loss.item(),
                        "d_fake_loss": d_fake_loss.item(),
                        "d_real_not_match_loss": d_real_not_match_loss.item(),
                        "d_fake_not_match_loss": d_fake_not_match_loss.item(),
                        "d_vec_not_match_loss": d_vec_not_match_loss.item(),
                        "g_fake_loss": g_fake_loss.item(),
                        "g_real_loss": g_real_loss.item(),
                        "g_fake_pow_loss": g_fake_pow_loss.item(),
                        "g_rise_loss": g_rise_loss.item(),
                        "g_drop_loss": g_drop_loss.item()
                    }
                )
                self.logger.end_iter()

            if self.logger.epoch % self.epochs_per_save == 0:
                self.saver.save(self.logger.epoch, self.g_a2b, self.g_optimizer, None, args.device, "g_a2b")
                self.saver.save(self.logger.epoch, self.g_b2a, self.g_optimizer, None, args.device, "g_b2a")
                self.saver.save(self.logger.epoch, self.d_a, self.d_optimizer, None, args.device, "d_a")
                self.saver.save(self.logger.epoch, self.d_b, self.d_optimizer, None, args.device, "d_b")

            if self.logger.epoch % self.epochs_per_plot == 0:
                tqdm.write("log %d step audio" % i)
                self.g_a2b.eval()
                self.g_b2a.eval()
                self.d_a.eval()
                self.d_b.eval()

                self.logger.log_img(fake_a.detach().cpu()[0].unsqueeze(0), "fake_a")
                self.logger.log_img(fake_b.detach().cpu()[0].unsqueeze(0), "fake_b")
                self.logger.log_img(fake_ba.detach().cpu()[0].unsqueeze(0), "fake_ba")
                self.logger.log_img(fake_ab.detach().cpu()[0].unsqueeze(0), "fake_ab")
                self.logger.log_img(test_a.detach().cpu()[0].unsqueeze(0), "test_a")
                self.logger.log_img(test_b.detach().cpu()[0].unsqueeze(0), "test_b")
                self.logger.log_img(test_b_rise.detach().cpu()[0].unsqueeze(0), "test_b_rise")
                self.logger.log_img(test_b_drop.detach().cpu()[0].unsqueeze(0), "test_b_drop")
                self.logger.log_img(test_a_rise.detach().cpu()[0].unsqueeze(0), "test_a_rise")
                self.logger.log_img(test_a_drop.detach().cpu()[0].unsqueeze(0), "test_a_drop")

                real_a_wav = self.dataset.mel2wav(a_mels[0].detach().cpu(), self.dataset.a_std, self.dataset.a_mean)
                self.logger.log_audio(real_a_wav.reshape(1, int(16000 * 1.024)).T, "real_a", 16000)
                real_b_wav = self.dataset.mel2wav(b_mels[0].detach().cpu(), self.dataset.b_std, self.dataset.b_mean)
                self.logger.log_audio(real_b_wav.reshape(1, int(16000 * 1.024)).T, "real_b", 16000)
                fake_a_wav = self.dataset.mel2wav(fake_a[0].detach().cpu(), self.dataset.a_std, self.dataset.a_mean)
                self.logger.log_audio(fake_a_wav.reshape(1, int(16000 * 1.024)).T, "fake_a", 16000)
                fake_b_wav = self.dataset.mel2wav(fake_b[0].detach().cpu(), self.dataset.b_std, self.dataset.b_mean)
                self.logger.log_audio(fake_b_wav.reshape(1, int(16000 * 1.024)).T, "fake_b", 16000)
                rise_a_wav = self.dataset.mel2wav(test_a_rise[0].detach().cpu(), self.dataset.a_std, self.dataset.a_mean)
                self.logger.log_audio(rise_a_wav.reshape(1, int(16000 * 1.024)).T, "rise_a", 16000)
                drop_a_wav = self.dataset.mel2wav(test_a_drop[0].detach().cpu(), self.dataset.a_std, self.dataset.a_mean)
                self.logger.log_audio(drop_a_wav.reshape(1, int(16000 * 1.024)).T, "drop_a", 16000)
                rise_b_wav = self.dataset.mel2wav(test_b_rise[0].detach().cpu(), self.dataset.b_std, self.dataset.b_mean)
                self.logger.log_audio(rise_b_wav.reshape(1, int(16000 * 1.024)).T, "rise_b", 16000)
                drop_b_wav = self.dataset.mel2wav(test_b_drop[0].detach().cpu(), self.dataset.b_std, self.dataset.b_mean)
                self.logger.log_audio(drop_b_wav.reshape(1, int(16000 * 1.024)).T, "drop_b", 16000)
                fake_pow_a_wav = self.dataset.mel2wav(fake_ba[0].detach().cpu(), self.dataset.a_std, self.dataset.a_mean)
                self.logger.log_audio(fake_pow_a_wav.reshape(1, int(16000 * 1.024)).T, "fake_pow_a", 16000)
                fake_pow_b_wav = self.dataset.mel2wav(fake_ab[0].detach().cpu(), self.dataset.b_std, self.dataset.b_mean)
                self.logger.log_audio(fake_pow_b_wav.reshape(1, int(16000 * 1.024)).T, "fake_pow_b", 16000)

            self.logger.end_epoch()


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()

    #prepar data

    # prepar_data_save_path = os.path.join(args.preprocessed_data_dir, args.speaker_A_id)
    # prepar_data = TrainDataPreparing()
    # prepar_data.flush_dataset("/var/classifed/kevin_emotion_sources.csv", "/var/classifed/lewis_emotion_sources.csv", path_fixed=lambda x: os.path.join("/", "var", x))
    # prepar_data.save_dataset()

    # del prepar_data

    pad = PADTraining(args)
    pad.train()
