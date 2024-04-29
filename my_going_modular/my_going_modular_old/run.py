import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import transforms


import engine_train
import model_creator
import plot
import data_setup

print(f"\nThe CWD is {os.getcwd()}\n")
parser = argparse.ArgumentParser()

parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=10)
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--hidden_units", type=int, default=10)
parser.add_argument("--input_shape", type=int, default=3)
parser.add_argument("--learning_rate", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--plot", type=bool, default=True)


args = parser.parse_args()

NUM_EPOHES = args.num_epochs
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
INPUT_SHAPE = args.input_shape
SEED = args.seed
VERBOSE = args.verbose
PLOT = args.plot


device = "cuda" if torch.cuda.is_available() else "cpu"

my_data_path = Path("my_data")
images_path = my_data_path / "my_pizza_steak_sushi"

train_path = images_path / "train"
test_path = images_path / "test"


data_transform = transforms.Compose(
    [
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ]
)

train_dataloader, test_dataloader = data_setup.create_data_loaders(
    train_path=train_path,
    test_path=test_path,
    data_transform=data_transform,
    bach_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# TODO How to Get information from data_loader
OUPUT_SHAPE = 3
torch.manual_seed(SEED)

model = model_creator.MyTinnyVGG(
    input_shape=INPUT_SHAPE, hidden_units=HIDDEN_UNITS, output_shape=OUPUT_SHAPE
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)


results = engine_train.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOHES,
    verbose=VERBOSE,
    device=device,
)


plot_path = Path("my_going_modular/results.png")

if PLOT:
    plot.plot_results(results=results)
    plt.savefig(plot_path, format="png")
