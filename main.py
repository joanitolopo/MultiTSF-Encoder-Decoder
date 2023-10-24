import torch
import jcopdl
import pandas as pd
import argparse
from utils import load_data
from torch import nn, optim
from jcopdl.callback import Callback, set_config
from processing import Processing
from encoder import Encoder
from decoder import Decoder
from multitsf import MULTITSFMODEL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Run Model with Command-Line Arguments")
    parser.add_argument("--file", required=True, help="CSV File")
    parser.add_argument("--stations", default="Gucheng")
    parser.add_argument('--targets', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument("--n_layers", type=int, default=4, help="n_layers")
    parser.add_argument("--cnn_layers", type=int, default=2, help="cnn_layers")
    parser.add_argument("--enc_hid_dim", type=int, default=64, help="enc_hid_dim")
    parser.add_argument("--dec_hid_dim", type=int, default=64, help="dec_hid_dim")
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel_size")
    args = parser.parse_args()

    data = load_data(args.stations)

    process = Processing(dataframe=data, target_columns=args.targets)
    train_loader, test_loader, forecast_loader = process.dataloader()

    config = set_config({"input_dim": len(data.drop(args.targets)),
                         "output_dim": len(args.targets),
                         "n_layers": args.n_layers,
                         "cnn_layers": args.cnn_layers,
                         "enc_hid_dim": args.enc_hid_dim,
                         "dec_hid_dim": args.dec_hid_dim,
                         "dropout": args.dropout,
                         "kernel_size": args.kernel_size})

    encoder = Encoder(config.input_dim, config.enc_hid_dim, config.cnn_layers,
                      config.n_layers, config.kernel_size, config.dropout, device)
    decoder = Decoder(config.output_dim, config.dec_hid_dim, config.n_layers, config.dropout, device)
    model = MULTITSFMODEL(encoder, decoder, device)

    criterion = nn.MSELoss(reduction="none")
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    callback = Callback(model, config, outdir=f"{PATH}artifact/single-output", early_stop_patience=20)


if __name__ == "__main__":
    main()
