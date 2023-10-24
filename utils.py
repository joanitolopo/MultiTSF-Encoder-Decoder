import torch
import os
import pandas as pd
from torch.optim.lr_scheduler import StepLR

def calculate_losses(criterion, output, target):
    loss = criterion(output, target)

    # MSE loss per output
    mse_per_output = torch.mean(loss, dim=0)

    # Calculate RMSE
    rmse_per_output = torch.sqrt(mse_per_output)

    # Calculate MAE
    mae_per_output = torch.mean(torch.abs(output - target), dim=0)

    # Calculate MAPE
    mape_per_output = torch.mean(torch.abs((output - target) / target), dim=0) * 100

    return loss, mse_per_output, rmse_per_output, mae_per_output, mape_per_output


def train_model(model, train_loader, criterion, optimizer, device, step=1):
    model.train()
    running_loss = 0.0

    for feature, target in train_loader:
        feature, target = feature.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(feature, target, tf_ratio=0.5, training_types="teacher_forcing", dynamic_tf=True, step=step)

        # Reshape the output and target tensors to have the same shape
        output = output.view(-1, target.shape[-1])  # Flatten the output along the time dimension
        target = target.view(-1, target.shape[-1])  # Flatten the target along the time dimension
        loss = criterion(output, target)
        loss = torch.mean(loss)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        running_loss += loss.item()

    return running_loss / len(train_loader)


def evaluate_model(model, test_loader, criterion, device, step=1):
    model.eval()
    total_loss = 0.0
    all_mse_per_output = []
    all_rmse_per_output = []
    all_mae_per_output = []
    all_mape_per_output = []

    with torch.no_grad():
        for feature, target in test_loader:
            feature, target = feature.to(device), target.to(device)
            output = model(feature, target, tf_ratio=0.5, training_types="mixed_teacher_forcing", dynamic_tf=False,
                           step=step)

            # # Reshape the output and target tensors to have the same shape
            output = output.view(-1, target.shape[-1])  # Flatten the output along the time dimension
            target = target.view(-1, target.shape[-1])  # Flatten the target along the time dimension
            loss = criterion(output, target)
            loss = torch.mean(loss)

            # Calculate losses
            _, mse_per_output, rmse_per_output, mae_per_output, mape_per_output = calculate_losses(criterion, output,
                                                                                                   target)

            total_loss += loss.item() * feature.size(0)

            # Accumulate individual losses per output
            all_mse_per_output.append(mse_per_output)
            all_rmse_per_output.append(rmse_per_output)
            all_mae_per_output.append(mae_per_output)
            all_mape_per_output.append(mape_per_output)

    avg_loss = total_loss / len(test_loader.dataset)

    # Concatenate and stack individual losses per output
    all_mse_per_output = torch.stack(all_mse_per_output, dim=0)
    all_rmse_per_output = torch.stack(all_rmse_per_output, dim=0)
    all_mae_per_output = torch.stack(all_mae_per_output, dim=0)
    all_mape_per_output = torch.stack(all_mape_per_output, dim=0)

    return avg_loss, all_mse_per_output, all_rmse_per_output, all_mae_per_output, all_mape_per_output


def loop(optimizer, callback, model, criterion, train_loader, test_loader):
    # Define the learning rate schedule
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    all_mse_per_output_list = []
    all_rmse_per_output_list = []
    all_mae_per_output_list = []
    all_mape_per_output_list = []

    while True:
        # Training & Evaluation
        train_loss = train_model(model, train_loader, criterion, optimizer, device, step=1)
        test_loss, all_mse_per_output, all_rmse_per_output, all_mae_per_output, all_mape_per_output = evaluate_model(
            model, test_loader, criterion, device, step=1)

        # Update the learning rate schedule
        scheduler.step()

        # Logging
        callback.log(train_loss, test_loss)

        # Checkpoint
        callback.save_checkpoint()

        # Runtime Plotting
        callback.cost_runtime_plotting()

        # Early Stopping
        if callback.early_stopping(model, monitor='test_cost'):
            callback.plot_cost()

            # Store the all losses per output for each epoch
            all_mse_per_output_list.append(all_mse_per_output)
            all_rmse_per_output_list.append(all_rmse_per_output)
            all_mae_per_output_list.append(all_mae_per_output)
            all_mape_per_output_list.append(all_mape_per_output)
            break


def load_data(stations="Gucheng"):
    data = []
    # Iterate over each file in the folder
    for file_name in os.listdir("/dataset/"):
        if file_name.endswith('.csv'):  # Only consider CSV files
            file_path = os.path.join("/dataset/", file_name)
            data_frame = pd.read_csv(file_path, index_col=["No"])
            data_frame["sites"] = file_name
            data.append(data_frame)

    # Concatenate all DataFrames into a single DataFrame
    df = pd.concat(data, ignore_index=True)
    df['date'] = pd.to_datetime(df[['year', 'month', 'day', 'hour']])
    df.drop(columns=['year', 'month', 'day', 'hour', 'wd', 'station'], inplace=True)
    df.set_index('date', inplace=True)

    return df[[df.sites == f"{stations}.csv"]]
