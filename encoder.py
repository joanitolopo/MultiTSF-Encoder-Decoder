import torch
from torch import nn
from torch.nn import functional as f


class Encoder(nn.Module):
    def __init__(self,
                 input_dim,
                 enc_hid_dim,
                 cnn_layers,
                 n_layers,
                 kernel_size,
                 dropout,
                 device):
        super().__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd!"

        self.device = device
        self.enc_hid_dim = enc_hid_dim
        self.n_layers = n_layers

        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)

        self.convolutions = nn.ModuleList([nn.Conv1d(
            in_channels=input_dim,
            out_channels=2 * input_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2)
            for i in range(cnn_layers)])

        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(2 * input_dim) for i in range(cnn_layers)])

        self.dropout = nn.Dropout(dropout)

        self.bi_lstm = nn.LSTM(input_dim, enc_hid_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(enc_hid_dim * 2, enc_hid_dim)

    def forward(self, encode_input):
        # inputs = [batch_size, n_seq, n_feature]
        conv_input = encode_input.transpose(1, 2).to(self.device)

        for i, (conv, bn) in enumerate(zip(self.convolutions, self.batch_norm)):
            convolution = conv(self.dropout(conv_input))
            convolution = bn(convolution)  # Apply BatchNorm
            convolution = f.glu(convolution, dim=1)
            convolution = (convolution + conv_input) * self.scale
            conv_input = convolution

        rnn_input = conv_input.transpose(1, 2)
        out, (hidden, cell) = self.bilstm(rnn_input)

        # Move hidden and cell to GPU
        hidden = hidden.to(self.device)
        cell = cell.to(self.device)

        # out = [batch_size, n_seq, n_directions * hidden_size]
        # hidden == cell = [n_layers * n_directions, batch_size, hidden_size]

        # concatenation
        hidden = torch.tanh(
            self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))  # [batch_size, hidden_size]
        cell = torch.tanh(self.fc(torch.cat((cell[-2, :, :], cell[-1, :, :]), dim=1)))  # [batch_size, hidden_size]
        hidden = hidden.unsqueeze(0).repeat(self.n_layers, 1, 1)  # [n_layers, batch_size, hidden_size]
        cell = cell.unsqueeze(0).repeat(self.n_layers, 1, 1)  # [n_layers, batch_size, hidden_size]

        return out, hidden, cell
