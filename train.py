import os
from functools import reduce
from typing import overload

import neptune.new as neptune
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from dotenv import load_dotenv
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataprep.dataset_binance import BinanceCoinDataset
from ml.models.BSM import ResNetBSM4

load_dotenv()


TRUE_RUN = True
NEPTUNE_API_TOKEN = os.getenv("NEPTUNE_API_TOKEN")
MODE = "async" if TRUE_RUN else "debug"
run = neptune.init(
    project="crazyleg11/binance-predict",
    mode=MODE,
    tags=["BSM5", "MSE"],
    source_files=["**/*.py"],
    api_token=NEPTUNE_API_TOKEN,
)  # your credentials

params = {"learning_rate": 0.0001, "optimizer": "Adam"}
run["parameters"] = params


device = torch.device("cuda")


criterion = nn.MSELoss()

batch_size = 256

C = [
    "AVAXUSDT",
    "SOLUSDT",
    "ETCETH",
    "ETCBTC",
    "MKRUSDT",
    "IOTAUSDT",
    "ADAUSDT",
    "XLMUSDT",
    "TRXUSDT",
    "XMRUSDT",
    "EOSUSDT",
    "ETHUSDT",
    "BTCUSDT",
]

train_set = BinanceCoinDataset(test=(False, 0.999), window_size=512, currencies=C)
val_set = BinanceCoinDataset(test=(True, 0.999), window_size=512, currencies=C)

n_features = train_set.data.shape[1]
net = ResNetBSM4(n_features=n_features, n_outputs=len(train_set.currencies))
# net.load_state_dict(torch.load("BSM_big.pth"))
net.cuda(device)
optimizer = optim.Adam(net.parameters(), lr=params["learning_rate"])

trainloader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=16
)

testloader = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False, num_workers=16
)
best_loss = 99999

scheduler = ReduceLROnPlateau(optimizer, "min")

# Generate test sample
# get timestamp and results -> dump them to the file

for epoch in range(4):  # loop over the dataset multiple times
    net.train()
    running_loss = 0.0

    y = []
    y_hat = []
    losses = []

    for i, data in enumerate(trainloader, 0):
        # if i > 2: break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = (
            inputs.to(device),
            labels.to(device),
        )
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        outputs = (labels != -1).detach() * outputs
        labels = (labels != -1).detach() * labels

        loss = criterion(outputs, labels)
        loss.backward()
        # print(loss.item())

        optimizer.step()

        y.append(labels.cpu().detach())
        y_hat.append(outputs.cpu().detach())
        losses.append(loss.cpu().detach())

        # print statistics

        if (i + 1) % 10 == 0:  # print every 2000 mini-batches
            run["train/loss"].log(loss.item())
            print("[%d, %5d] loss: %.8f " % (epoch + 1, i + 1, loss.item()))

    y = np.vstack(y)
    y_hat = np.vstack(y_hat)
    losses = np.vstack(losses)

    print("starting evaluations")

    losses = []
    pearsons = []
    pred = []
    labels_h = []
    y_test = []
    y_hat_test = []
    losses_test = []

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            inputs, labels = (
                inputs.to(device),
                labels.to(device),
            )

            outputs = net(inputs)
            outputs = (labels != -1).detach() * outputs
            labels = (labels != -1).detach() * labels

            loss = criterion(outputs, labels)

            losses.append(loss.item())
            pred.append((outputs).cpu().detach())
            labels_h.append((labels).cpu().detach())

        y_test = np.vstack(labels_h)
        y_hat_test = np.vstack(pred)
        losses_test = np.vstack(losses)

        scheduler.step(np.array(losses).mean())

        run["test/loss_total_mean"].log(np.array(losses).mean())

        if best_loss > np.array(losses).mean():
            best_loss = np.array(losses).mean()
            torch.save(net.state_dict(), "best_model.pth")
            run["model/saved_model"].upload("best_model.pth")

        torch.save(net.state_dict(), "last_model.pth")
        run["model/last_model"].upload("last_model.pth")

print("Finished Training")
run.stop()
