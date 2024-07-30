#!/bin/sh

python3 -W ignore::UserWarning -m mask_cyclegan_vc.w2v2_cycle \
--name w2v2pad \
--seed 0 \
--save_dir results/ \
--preprocessed_data_dir ./cache/ \
--speaker_A_id a \
--speaker_B_id b \
--epochs_per_save 150 \
--epochs_per_plot 10 \
--num_epochs 4000 \
--batch_size 1 \
--generator_lr 5e-4 \
--discriminator_lr 5e-5 \
--decay_after 1e4 \
--sample_rate 16000 \
--num_frames 64 \
--max_mask_len 10 \
--gpu_ids 0
