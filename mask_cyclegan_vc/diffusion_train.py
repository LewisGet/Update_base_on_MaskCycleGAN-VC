import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataset import Dataset

import numpy as np

import os
import pickle
from tqdm import tqdm

from mask_cyclegan_vc.model import Generator
from args.cycleGAN_train_arg_parser import CycleGANTrainArgParser
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class TrainDataPreparing(Dataset):
    def __init__(self, path, decay_level=1000, n_frames=64, device="cuda:0"):
        self.decay_level = decay_level
        self.device = torch.device(device)
        self.n_frames = n_frames

        with open(f"{path}_normalized.pickle", 'rb') as f:
            self.org_mels = pickle.load(f)
        dataset_stats = np.load(f"{path}_norm_stat.npz")

        self.mean = torch.tensor(dataset_stats['mean']).to(self.device)
        self.std = torch.tensor(dataset_stats['std']).to(self.device)

        self.vocoder = torch.hub.load('LewisGet/melgan-neurips', 'load_melgan')

        self.init_clean_dataset()

    def mel2wav(self, mel):
        rev = mel * self.std + self.mean

        return self.vocoder.inverse(rev.unsqueeze(0))

    def init_clean_dataset(self):
        self.mels = []
        self.masks = []
        self.mel_with_masks = []

    def load_dataset(self, load_path):
        self.mels = np.load(os.path.join(load_path, 'mels.npy'), allow_pickle=True).tolist()

    def save_dataset(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, 'mels.npy'), np.array(self.mels))

    def flush_dataset(self, path=None, start_at=0, save_plot=0):
        self.init_clean_dataset()

        if start_at > 0 and path is not None:
            self.load_dataset(path)

        for i, mel in enumerate(tqdm(self.org_mels[start_at:], leave=False), start=start_at):
            mel = torch.tensor(mel)
            self.mel_format(mel)

            if save_plot > 0 and i % save_plot == 0 and path is not None:
                self.save_dataset(path)

        if path is not None:
            self.save_dataset(path)

    def mel_format(self, mel):
        mel_len = mel.shape[-1]

        if mel_len < self.n_frames:
            return None

        end = mel_len - self.n_frames

        if end > 0:
            start = np.random.randint(0, end)
            mel = mel[:, start:start + self.n_frames]

        self.mels.append(mel.detach().cpu().numpy())

    def create_decay_dataset(self, mel):
        _masks = []
        _with_masks = []

        x = torch.tensor(mel)

        for i in range(self.decay_level):
            decay_index = np.random.randint(x.shape[-1])
            mask = torch.ones_like(x)

            if i != 0:
                mask = torch.tensor(_masks[-1].tolist())

            org_reserve_ply = 1 - i / self.decay_level
            decay_ply = i / self.decay_level

            org_data = torch.clone(mask[:, decay_index]) * org_reserve_ply
            decay_data = torch.rand_like(mask[:, decay_index]) * decay_ply

            mask[:, decay_index] = org_data + decay_data
            _masks.append(mask)
            _with_masks.append(x * mask)

        return np.array(_masks), np.array(_with_masks)

    def __getitem__(self, index):
        _mask, _with_masks = self.create_decay_dataset(self.mels[index])

        return np.array(self.mels[index]), _mask, _with_masks

    def __len__(self):
        return len(self.mels)


class MyGenerator(Generator):
    def __init__(self, input_shape=(80, 64), residual_in_channels=256):
        super().__init__()
        self.decay = nn.Linear(1, 80 * 64)

    def forward(self, x, y):
        y = self.decay(y)
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


class DiffusionTraining(object):
    def __init__(self, args):
        self.args = args

        self.num_epochs = args.num_epochs
        self.start_epoch = args.start_epoch
        self.generator_lr = args.generator_lr
        self.mini_batch_size = args.batch_size
        self.device = args.device
        self.epochs_per_save = args.epochs_per_save
        self.epochs_per_plot = args.epochs_per_plot
        self.sample_rate = args.sample_rate

        self.dataset = TrainDataPreparing(os.path.join(args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}"))
        self.dataset.load_dataset(os.path.join(args.preprocessed_data_dir, args.speaker_A_id))
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=self.mini_batch_size, shuffle=True)

        self.g = MyGenerator().to(self.device)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), lr=self.generator_lr, betas=(0.5, 0.999))

        self.saver = ModelSaver(args)
        self.logger = TrainLogger(args, len(self.dataset))

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.logger.start_epoch()
            self.g.train()

            for i, (mels, masks, mel_with_masks) in enumerate(tqdm(self.dataloader)):
                self.logger.start_iter()
                mels = torch.tensor(mels).to(self.device, dtype=torch.float32)

                for decay_level in tqdm(range(self.dataset.decay_level - 1), leave=False):
                    real_batch_size = len(mels)
                    _decay_level = self.dataset.decay_level - decay_level

                    f0 = mel_with_masks[:real_batch_size, decay_level]
                    x0 = mel_with_masks[:real_batch_size, decay_level + 1]

                    f0 = torch.tensor(f0).to(self.device, dtype=torch.float32)
                    x0 = torch.tensor(x0).to(self.device, dtype=torch.float32)

                    x1 = torch.tensor([[_decay_level]]).repeat(real_batch_size, 1).to(self.device, dtype=torch.float32)

                    g_data = self.g(x0, x1)
                    g_loss = F.mse_loss(g_data, f0)

                    self.g_optimizer.zero_grad()
                    g_loss.backward()
                    self.g_optimizer.step()

                self.logger.log_iter(loss_dict={"g_loss": g_loss.item()})
                self.logger.end_iter()

            if self.logger.epoch % self.epochs_per_save == 0:
                self.saver.save(self.logger.epoch, self.g, self.g_optimizer, None, args.device, "g")

            if self.logger.epoch % self.epochs_per_plot == 0:
                tqdm.write("log %d step audio" % i)
                self.g.eval()

                with torch.no_grad():
                    t = torch.tensor(x0[0]).unsqueeze(0).to(self.device)

                    for decay_level in range(self.dataset.decay_level - 1):
                        _decay_level = self.dataset.decay_level - decay_level
                        x1 = torch.tensor([[_decay_level]]).to(self.device, dtype=torch.float32)

                        t = self.g(t, x1)

                        if decay_level % 10 == 0:
                            self.logger.log_img(t, str(decay_level))

                real_wav = self.dataset.mel2wav(mels[0])
                self.logger.log_audio(real_wav.cpu().reshape(1, int(16000 * 1.024)).T, "real", 16000)
                faked_wav = self.dataset.mel2wav(t[0])
                self.logger.log_audio(faked_wav.cpu().reshape(1, int(16000 * 1.024)).T, "fake", 16000)

            self.logger.end_epoch()


if __name__ == "__main__":
    parser = CycleGANTrainArgParser()
    args = parser.parse_args()

    #prepar data

    prepar_data_save_path = os.path.join(args.preprocessed_data_dir, args.speaker_A_id)
    prepar_data = TrainDataPreparing(os.path.join(args.preprocessed_data_dir, args.speaker_A_id, f"{args.speaker_A_id}"))
    prepar_data.flush_dataset()
    prepar_data.save_dataset(prepar_data_save_path)

    del prepar_data

    diffusion = DiffusionTraining(args)
    diffusion.train()
