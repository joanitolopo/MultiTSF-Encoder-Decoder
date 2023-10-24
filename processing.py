import pandas as pd
from dataset import TimeSeriesDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class Processing:
    def __init__(self,
                 dataframe,
                 target_columns=None,
                 scaler="MinMax",
                 test_size=0.2,
                 n_forecast = 35064,
                 seq_len=24,
                 batch_size = 32):

        if target_columns is None:
            target_columns = ["PM25", "PM10", "SO2", "NO2"]
        self.df = dataframe
        self.target_columns = target_columns
        self.n_forecast = n_forecast
        self.scaler = scaler
        self.test_size = test_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.normalized_data = None

    def scaling(self):
        # imputation
        self.df.interpolate(method='linear', inplace=True)
        self.df.fillna(self.df.median(), inplace=True)

        if self.scaler == "MinMax":
            minmax = MinMaxScaler()
            self.normalized_data = pd.DataFrame(minmax.fit_transform(self.df.values),
                                                columns=self.df.columns,
                                                index=self.df.index)

    def dataloader(self):
        ts_train, ts_test = train_test_split(self.normalized_data[:self.n_forecast], self.test_size=0.2, shuffle=False)
        seq_len = 24  # per 1 hour
        bs = 32  # (8,16,32,64)

        train_set = TimeSeriesDataset(ts_train, self.target_columns, self.seq_len)
        train_loader = DataLoader(train_set, batch_size=self.batch_size)

        test_set = TimeSeriesDataset(ts_test, self.target_columns, self.seq_len)
        test_loader = DataLoader(test_set, batch_size=self.batch_size)

        forecast_set = TimeSeriesDataset(self.normalized_data[self.n_forecast:], self.target_columns, seq_len)
        forecast_loader = DataLoader(forecast_set, batch_size=bs)

        return train_loader, test_loader, forecast_loader
