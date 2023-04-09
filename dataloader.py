import os
import unicodedata
import json
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

transformers.logging.set_verbosity_error()


def collate_fn(batch):
    batch = default_collate(batch)
    batch["input_ids"] = batch["input_ids"].squeeze()
    if "labels" in batch:
        batch["labels"] = batch["labels"].squeeze()
    batch["attention_mask"] = batch["attention_mask"].squeeze()
    batch["token_type_ids"] = batch["token_type_ids"].squeeze()

    return batch


class CornellDataset(pl.LightningDataModule):
    # taken in part from https://github.com/SudharshanShanmugasundaram/Chatbot/blob/master/chatbot.ipynb
    def __init__(
        self,
        path: str,
        max_seq_len: int = 256,
        min_word_count: int = 3,
        model: str = "bert-base-uncased",
        batch_size: int = 16,
    ):
        super().__init__()
        self.path = path
        self.lines_path = os.path.join(path, "movie_lines.txt")
        self.conv_path = os.path.join(path, "movie_conversations.txt")
        self.min_word_count = min_word_count
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = 0

        if "uncased" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def setup(self, stage: str):
        if self.vocab_size > 0:
            # already setup
            return

        # splits each line of the file into a dictionary of fields(lineID,characterID,movieID,character,text)
        line_fields = ["lineID", "characterID", "movieID", "character", "text"]
        lines = {}

        with open(self.lines_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                lineObj = {}
                for i, field in enumerate(line_fields):
                    lineObj[field] = values[i]
                lines[lineObj["lineID"]] = lineObj

        # Grouping fields of lines from the above loaded lines into conversation based on "movie_conversations.txt"
        conv_fields = ["character1ID", "character2ID", "movieID", "utteranceIDs"]
        conversations = []

        with open(self.conv_path, "r", encoding="iso-8859-1") as f:
            for line in f:
                values = line.split(" +++$+++ ")
                # Extract fields
                convObj = {}
                for i, field in enumerate(conv_fields):
                    convObj[field] = values[i]
                lineIds = eval(convObj["utteranceIDs"])
                # Reassemble lines
                convObj["lines"] = []
                for lineId in lineIds:
                    convObj["lines"].append(lines[lineId])
                conversations.append(convObj)

        # Extracts pairs of sentences from conversations
        qa_pairs = []
        for conversation in tqdm(conversations, desc="Processing conversations"):
            # Iterate over all the lines of the conversation
            for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
                inputLine = conversation["lines"][i]["text"].strip()
                targetLine = conversation["lines"][i + 1]["text"].strip()
                # Filter wrong samples (if one of the lists is empty)
                if inputLine and targetLine:
                    # convert unicode to plain ascii
                    inputLine = "".join(
                        c for c in unicodedata.normalize("NFD", inputLine) if unicodedata.category(c) != "Mn"
                    )
                    targetLine = "".join(
                        c for c in unicodedata.normalize("NFD", targetLine) if unicodedata.category(c) != "Mn"
                    )

                    # lowercase, trim,
                    inputLine = inputLine.lower().strip()
                    targetLine = targetLine.lower().strip()

                    qa_pairs.append([inputLine, targetLine])

        # remove low frequency words
        words = {}
        for input_line, target_line in tqdm(qa_pairs, desc="Calculating word freq"):
            raw_words = input_line.split(" ") + target_line.split(" ")

            for r in raw_words:
                try:
                    words[r] += 1
                except:
                    words[r] = 1

        remove_words = []
        for w, c in words.items():
            if c < self.min_word_count:
                remove_words.append(w)

        self.vocab_size = len(words) - len(remove_words)

        # remove pairs with those words
        remove_words = set(remove_words)
        remove_idxs = []
        i = 0
        for input_line, target_line in tqdm(qa_pairs, desc="Removing low freq words"):
            raw_words = set(input_line.split(" ") + target_line.split(" "))

            if len(raw_words & remove_words) > 0:
                remove_idxs.append(i)

            i += 1

        for i in remove_idxs:
            try:
                del qa_pairs[i]
            except IndexError:
                pass

        # tokenize and split dataset
        # tokenizer also returns token type ids to indicate start and end of sequences, attention masks, and position ids
        # return type of each tokenizer output is a dict containing these tensors
        # https://huggingface.co/docs/transformers/glossary#token-type-ids
        tokenzize_pairs = []
        for qa in tqdm(qa_pairs, desc="Tokenizing sequences"):
            x = self.tokenizer.encode_plus(
                qa[0],
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            y = self.tokenizer.encode_plus(
                qa[1],
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            x["labels"] = y["input_ids"]

            tokenzize_pairs.append(x)

        self.train, self.test, _, _ = train_test_split(tokenzize_pairs, tokenzize_pairs, test_size=0.4)
        self.val, self.test, _, _ = train_test_split(self.test, self.test, test_size=0.5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=collate_fn)

    # have to define but not needed
    def prepare_data(self):
        return

    # have to define but not needed
    def prepare_data_per_node(self):
        return

    def decode(self, encoded_seq):
        return self.tokenizer.decode(encoded_seq)


class RedditQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        max_seq_len: int = 256,
        min_word_count: int = 3,
        model: str = "bert",
        batch_size: int = 16,
    ):
        super().__init__()
        self.path = path
        self.questions = os.path.join(path, "reddit_questions.csv")
        self.answers = os.path.join(path, "reddit_answers.csv")
        self.min_word_count = min_word_count
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = 0

        if "uncased" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

    def setup(self, stage: str):
        if self.vocab_size > 0:
            # already setup
            return

        questions = pd.read_csv(self.questions, index_col=0, delimiter=";")
        answers = pd.read_csv(self.answers, index_col=1, delimiter=";")
        qa_pairs = []

        for q_id in tqdm(questions.index, desc="Reading data from csv files"):
            q = questions.loc[q_id, "text"].strip('"').strip("'")
            try:
                data = answers.loc[q_id, "text"]
                if isinstance(data, str):
                    a = [data.strip('"').strip("'")]
                else:
                    a = [an.strip('"').strip("'") for an in data]
            except KeyError:
                continue

            qa_pairs.append([q, *a])

        # remove low frequency words
        words = {}
        for lines in tqdm(qa_pairs, desc="Calculating word freq"):
            raw_words = []
            for line in lines:
                raw_words += line.split(" ")

            for r in raw_words:
                try:
                    words[r] += 1
                except:
                    words[r] = 1

        remove_words = []
        for w, c in words.items():
            if c < self.min_word_count:
                remove_words.append(w)

        self.vocab_size = len(words) - len(remove_words)

        # remove pairs with those words
        remove_words = set(remove_words)
        remove_idxs = []
        i = 0
        for lines in tqdm(qa_pairs, desc="Removing low freq words"):
            raw_words = []
            for line in lines:
                raw_words += line.split(" ")

            raw_words = set(raw_words)

            if len(raw_words & remove_words) > 0:
                remove_idxs.append(i)

            i += 1

        for i in remove_idxs:
            try:
                del qa_pairs[i]
            except IndexError:
                pass

        # tokenize and split dataset
        # tokenizer also returns token type ids to indicate start and end of sequences, attention masks, and position ids
        # return type of each tokenizer output is a dict containing these tensors
        # https://huggingface.co/docs/transformers/glossary#token-type-ids
        tokenzize_pairs = []
        for qa in tqdm(qa_pairs, desc="Tokenizing sequences"):
            x = self.tokenizer.encode_plus(
                qa,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            tokenzize_pairs.append(x)

        self.train, self.test, _, _ = train_test_split(tokenzize_pairs, tokenzize_pairs, test_size=0.4)
        self.val, self.test, _, _ = train_test_split(self.test, self.test, test_size=0.5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=8, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=8, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=8, collate_fn=collate_fn)

    # have to define but not needed
    def prepare_data(self):
        return

    # have to define but not needed
    def prepare_data_per_node(self):
        return

    def decode(self, encoded_seq):
        return self.tokenizer.decode(encoded_seq)


class StandfordQADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        path: str,
        max_seq_len: int = 256,
        min_word_count: int = 3,
        model: str = "bert",
        batch_size: int = 16,
    ):
        super().__init__()
        self.path = path

        self.min_word_count = min_word_count
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = 0

        if "uncased" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)


if __name__ == "__main__":
    """dataset = CornellDataset("data/cornell movie-dialogs corpus")
    dataset.setup("fit")
    example = next(iter(dataset.train_dataloader()))
    print(example)
    print(example["input_ids"].shape)
    print(example["labels"].shape)
    print(example["attention_mask"].shape)
    print(example["token_type_ids"].shape)"""

    dataset = RedditQADataset("data/reddit-qa")
    dataset.setup("fit")
    example = next(iter(dataset.train_dataloader()))
    print(example)
    print(example["input_ids"].shape)
    print(example["attention_mask"].shape)
    print(example["token_type_ids"].shape)

    print(dataset.decode(example["input_ids"]))
