import torch
from torch.utils.data import DataLoader, Dataset
from jcopdl.utils.exception import TruncationError


class TimeSeriesDataset(Dataset):

    def __init__(self, time_series, target_col_names, seq_len, target_lag=1, trunc_len=None, summary=True):
        n_row, self.n_feature = time_series.shape
        self.n_target = len(target_col_names)
        self.n_seq = seq_len
        if trunc_len is None:
            self.truncated = False
            self.n_truncate = seq_len
        elif seq_len % trunc_len == 0:
            self.truncated = True
            self.n_truncate = trunc_len
        else:
            raise TruncationError("seq_len should be divisible by trunc_len for tBPTT to work")
        self.n_step = self.n_seq // self.n_truncate

        input_series = time_series.iloc[:-target_lag, :]
        target_series = time_series[target_col_names].iloc[target_lag:, :]

        if hasattr(time_series, 'index'):
            self.input_ticks = input_series.index
            self.target_ticks = target_series.index

        self.X = self._to_sequence(input_series)
        self.y = self._to_sequence(target_series)
        self.n_data = self.X.shape[0]

        if not self.truncated:
            self.X.squeeze_(1)
            self.y.squeeze_(1)

        if summary:
            n_removed = n_row - target_lag - self.n_data * self.n_seq
            if self.truncated:
                print(f"(N, T, S, F): ({self.n_data}, {self.n_step}, {self.n_truncate}, {self.n_feature})")
            else:
                print(f"(N, S, F): ({self.n_data}, {self.n_seq}, {self.n_feature})")
            print(f"Note: last \x1b[31m{n_removed} data excluded\x1b[0m\n")

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.X)

    def _to_sequence(self, series):
        dim = series.shape[1]
        series = torch.FloatTensor(series.values)
        series = series.unfold(0, self.n_seq, self.n_seq)
        series = series.view(-1, dim, self.n_step, self.n_truncate)
        series = series.permute(0, 2, 3, 1)
        return series
