import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from models import TransformerModel
from dataloader import CornellDataset

name = "bert_cornell"
epochs = 10
batch_size = 4


# data module
dataset = CornellDataset("data/cornell movie-dialogs corpus", batch_size=batch_size, model="bert-base-uncased")
# need to do setup to calculate vocab size first
dataset.setup("fit")


# model
tboard = TensorBoardLogger("tensorboard", name)
model = TransformerModel(dataset.vocab_size, ten_logger=tboard, model="bert", task="dialogue")
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
