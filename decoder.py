import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self,
                 output_dim,
                 dec_hid_dim,
                 n_layers,
                 dropout,
                 device):
        super().__init__()

        self.output_dim = output_dim
        self.dec_hid_dim = dec_hid_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        self.multi_lstm = nn.ModuleList([nn.LSTM(
            input_size=output_dim,
            hidden_size=dec_hid_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout)
            for i in range(output_dim)])

        self.multi_linear = nn.ModuleList([nn.Linear(dec_hid_dim, 1) for _ in range(self.output_dim)])

    def forward(self, decoder_input, hidden, cell):
        # input = [bs, n_seq, n_feature]
        # hidden = [n_layers, batch_size, hidden_size]
        # enc_out = [batch_size, n_seq, 2 * hidden_size]
        batch_size = decoder_input.shape[0]
        n_seq = decoder_input.shape[1]

        output = torch.zeros(batch_size, n_seq, self.output_dim).to(self.device)

        for i in range(self.output_dim):
            output_i, (hidden_i, cell_i) = self.multi_lstm[i](decoder_input, (hidden, cell))
            output_i = self.multi_linear[i](output_i)
            output[:, :, i] = output_i.squeeze(2)
            hidden, cell = hidden_i, cell_i

        return output, hidden, cell
