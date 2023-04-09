import torch
import pytorch_lightning as pl
import evaluate
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from transformers import (
    BertLMHeadModel,
    BertTokenizer,
    BertForQuestionAnswering,
    BertConfig,
    OPTConfig,
    OPTForCausalLM,
    OPTForQuestionAnswering,
    AutoTokenizer,
    LongformerConfig,
    LongformerForQuestionAnswering,
    LongformerModel,
    LongformerTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    AdamW,
)

from metrics import perplexity, accuracy


class DummyTokenizer:
    def __init__(self):
        pass

    def tokenize(self, x):
        return x


class TransformerModel(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        ten_logger: TensorBoardLogger = None,
        num_layers: int = 12,
        num_heads: int = 12,
        learning_rate: float = 2e-5,
        model: str = "bert",
        task: str = "dialogue",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.model_name = model

        if model == "bert":
            self.config = BertConfig(
                vocab_size=vocab_size, num_hidden_layers=num_layers, num_attention_heads=num_heads, is_decoder=True
            )
            if task == "dialogue":
                self.model = BertLMHeadModel.from_pretrained(
                    "bert-base-uncased",
                    config=self.config,
                    ignore_mismatched_sizes=True,
                )
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
            elif task == "qa":
                self.model = BertForQuestionAnswering.from_pretrained(
                    "deepset/bert-base-cased-squad2", config=self.config, ignore_mismatched_sizes=True
                )
                self.tokenizer = BertTokenizer.from_pretrained("deepset/bert-base-cased-squad2")

        elif model == "opt":
            self.config = OPTConfig(vocab_size=vocab_size, num_hidden_layers=num_layers, num_attention_heads=num_heads)
            if task == "dialogue":
                self.model = OPTForCausalLM.from_pretrained(
                    "facebook/opt-350m", config=self.config, ignore_mismatched_sizes=True
                )
            elif task == "qa":
                self.model = OPTForQuestionAnswering.from_pretrained(
                    "facebook/opt-350m", config=self.config, ignore_mismatched_sizes=True
                )

            self.tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        elif model == "gpt2":
            self.config = GPT2Config(vocab_size=vocab_size, num_hidden_layers=num_layers, num_attention_heads=num_heads)
            if task == "dialogue" or task == "qa":
                self.model = GPT2LMHeadModel.from_pretrained("gpt2", config=self.config, ignore_mismatched_sizes=True)

            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        elif model == "longformer":
            self.config = LongformerConfig(
                vocab_size=vocab_size, num_hidden_layers=num_layers, num_attention_heads=num_heads
            )
            if task == "dialogue":
                self.model = LongformerModel.from_pretrained(
                    "allenai/longformer-base-4096", config=self.config, ignore_mismatched_sizes=True
                )
                self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            elif task == "qa":
                self.model = LongformerForQuestionAnswering.from_pretrained(
                    "allenai/longformer-large-4096-finetuned-triviaqa", config=self.config, ignore_mismatched_sizes=True
                )
                self.tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")

        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.ten_logger = ten_logger

        self.metrics = {
            "rogue": evaluate.load("rouge"),
            "perplexity": perplexity,
            "accuracy": accuracy,
        }

    def forward(self, **inputs):
        # inputs is a dict that has: labels, input_ids, attention_masks, and token_type_ids
        # output is a dict containing loss, logits, labels, etc: https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertLMHeadModel
        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertModel

        # if using longformer for dialogue task, need to generate global attentions
        # for qa task, global attentions are automatically generated
        if self.model == "longformer" and self.task == "dialogue":
            global_attention_mask = torch.zeros(
                inputs["input_ids"].shape, dtype=torch.long, device=inputs["input_ids"].device
            )
            global_attention_mask[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] = 1
            return self.model(global_attention_mask=global_attention_mask, **inputs)

        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)

        return {"loss": outputs["loss"]}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)

        return {
            "val_loss": outputs["loss"],
            "labels": batch["labels"],
            "logits": outputs["logits"],
        }

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs]).detach().cpu()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu()
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)

        # convert logits to tokens using softmax activation
        predictions = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(predictions, dim=-1)

        self.log_dict(self.metrics["accuracy"](predictions, labels), prog_bar=True)
        self.log_dict(
            self.metrics["perplexity"](predictions.type(torch.FloatTensor), labels.type(torch.FloatTensor)),
            prog_bar=True,
        )
        # self.log_dict(
        #    self.metrics["rogue"].compute(
        #        predictions=self.decode(predictions),
        #        references=self.decode(labels),
        #        tokenizer=self.tokenizer,
        #    ),
        #    prog_bar=True,
        # )

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer

    def decode(self, encoded_seq):
        return self.tokenizer.decode(encoded_seq)


if __name__ == "__main__":
    model = TransformerModel(32786, model="bert", task="dialogue")
    model = TransformerModel(32786, model="bert", task="qa")

    model = TransformerModel(32786, model="opt", task="dialogue")
    model = TransformerModel(32786, model="opt", task="qa")

    model = TransformerModel(32786, model="longformer", task="dialogue")
    model = TransformerModel(32786, model="longformer", task="qa")

    model = TransformerModel(32786, model="gpt2", task="dialogue")
    model = TransformerModel(32786, model="gpt2", task="qa")
