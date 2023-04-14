import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import TransformerModel, MODEL_DICT
from dataloader import CornellDataset, RedditQADataset, StandfordQADataset
from argparse import ArgumentParser

parser = ArgumentParser(description="NLP Project")

parser.add_argument("-name", help="name for saving data", required=True, type=str)
parser.add_argument("-epochs", help="number of epochs to run for", default=10, type=int)
parser.add_argument("-bs", help="batch size", default=4, type=int)
parser.add_argument("-model", help="name of model to train", required=True, type=str)
parser.add_argument("-dataset", help="name of dataset to use for training", required=True, type=str)


args = parser.parse_args()

name = args.name
epochs = args.epochs
batch_size = args.bs


# data module
if args.dataset == "cornell":
    dataset = CornellDataset(
        "data/cornell movie-dialogs corpus",
        batch_size=batch_size,
        model=MODEL_DICT[args.model]["dialogue"],
        max_seq_len=512,
    )
elif args.dataset == "reddit":
    dataset = RedditQADataset(
        "data/reddit-qa",
        batch_size=batch_size,
        model=MODEL_DICT[args.model]["dialogue"],
        max_seq_len=256,
    )
elif args.dataset == "stanford":
    dataset = StandfordQADataset(
        batch_size=batch_size,
        model=MODEL_DICT[args.model]["qa"],
        max_seq_len=512,
    )

# need to do setup to calculate vocab size first
dataset.setup("fit")

# tensorboard
tboard = TensorBoardLogger("tensorboard", name)

# model
if args.dataset == "stanford":
    model = TransformerModel(dataset.vocab_size, ten_logger=tboard, model=args.model, task="qa")
else:
    model = TransformerModel(dataset.vocab_size, ten_logger=tboard, model=args.model, task="dialogue")


# print(model.summary())


# callbacks
checkpoint = ModelCheckpoint(
    "checkpoints", name + "_{epoch}-{val_loss:.2f}", monitor="val_loss", verbose=True, save_last=True, mode="min"
)

callbacks = [checkpoint]


# train
trainer = pl.Trainer(
    accelerator="gpu",
    gpus=[0],
    callbacks=callbacks,
    max_epochs=epochs,
)

trainer.fit(model, datamodule=dataset)
