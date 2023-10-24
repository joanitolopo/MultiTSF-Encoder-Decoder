import random
import torch
from torch import nn


class MULTITSFMODEL(nn.Module):

    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.device = device

    def forward(self, inputs, targets, tf_ratio=0.5, training_types="recursive", dynamic_tf=False, step=1):
        # encoder
        out, enc_hn, enc_cn = self.encoder(inputs.to(self.device))

        if step > 1:
            targets = targets[:, -step:]  # [batch_size, n_step, n_out]

        batch_size, n_seq, out_size = targets.shape
        outputs = torch.zeros(batch_size, n_seq, out_size).to(self.device)

        if step > 1:
            decoder_in = torch.zeros(batch_size, step, out_size).to(self.device)
        else:
            decoder_in = torch.zeros(batch_size, 1, out_size).to(self.device)

        for t in range(n_seq):
            # decoder
            out, dec_hn, dec_cn = self.decoder(decoder_in, enc_hn, enc_cn)
            outputs[:, t, :] = out.squeeze(1)

            if training_types == "recursive":
                decoder_in = out
            elif training_types == "teacher_forcing":
                decoder_in = targets[:, t, :].unsqueeze(1)
            elif training_types == "mixed_teacher_forcing":
                if random.random() < tf_ratio:
                    decoder_in = targets[:, t, :].unsqueeze(1)
                else:
                    decoder_in = out

            enc_hn, enc_cn = dec_hn, dec_cn

        if dynamic_tf and tf_ratio > 0:
            tf_ratio = tf_ratio - 0.02

        return outputs
