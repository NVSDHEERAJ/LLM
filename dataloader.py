import os
import unicodedata
import numpy as np
import pandas as pd
import multiprocessing as mp

import torch
import pytorch_lightning as pl
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, default_collate
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def collate_fn(batch):
    batch = default_collate(batch)
    batch["input_ids"] = batch["input_ids"].squeeze()
    if "labels" in batch:
        batch["labels"] = batch["labels"].squeeze()
    batch["attention_mask"] = batch["attention_mask"].squeeze()
    if "token_type_ids" in batch:
        batch["token_type_ids"] = batch["token_type_ids"].squeeze()

    return batch


def process_df(questions_chunk, answers):
    qa_pairs = []
    try:
        position = mp.current_process()._identity[0]
    except IndexError:
        position = 0
    for q_id in tqdm(questions_chunk.index, desc="Reading data from csv files", leave=False, position=position):
        q = str(questions_chunk.loc[q_id, "text"].strip('"').strip("'"))
        try:
            data = str(answers.loc[q_id, "text"])
            if isinstance(data, str):
                a = [str(data.strip('"').strip("'"))]
            else:
                a = [str(an.strip('"').strip("'")) for an in data]
        except KeyError:
            continue

        for answer in a:
            qa_pairs.append([q, answer])

    return qa_pairs


class CornellDataset(pl.LightningDataModule):
    # taken in part from https://github.com/SudharshanShanmugasundaram/Chatbot/blob/master/chatbot.ipynb
    def __init__(
        self,
        path: str,
        max_seq_len: int = 256,
        min_word_count: int = 2,
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

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

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
        convos = []
        for conversation in tqdm(conversations, desc="Processing conversations"):
            # Iterate over all the lines of the conversation
            finished_seqs = []
            seq = []
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

                    current_len = np.array([len(s) for s in seq]).sum()

                    # account for extra tokens for new lines, start and end tokens, etc
                    if len(inputLine) + current_len > self.max_seq_len - len(seq) - 50:
                        finished_seqs.append("\n".join(seq))
                        seq = [inputLine, targetLine]
                    elif len(inputLine) + len(targetLine) > self.max_seq_len - len(seq) - 50:
                        seq += [inputLine]
                        finished_seqs.append("\n".join(seq))
                        seq = [targetLine]
                    else:
                        seq += [inputLine, targetLine]

                if len(seq) > 2:
                    finished_seqs.append("\n".join(seq))

                convos.extend(finished_seqs)

        # remove low frequency words
        words = {}
        for convo in tqdm(convos, desc="Calculating word freq"):
            raw_words = convo.split(" ")

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
        for convo in tqdm(convos, desc="Removing low freq words"):
            raw_words = set(convo.split(" "))

            if len(raw_words & remove_words) > 0:
                remove_idxs.append(i)

            i += 1

        for i in remove_idxs:
            try:
                del convos[i]
            except IndexError:
                pass

        # tokenize and split dataset
        # tokenizer also returns token type ids to indicate start and end of sequences, attention masks, and position ids
        # return type of each tokenizer output is a dict containing these tensors
        # https://huggingface.co/docs/transformers/glossary#token-type-ids
        tokenzize_pairs = []
        for convo in tqdm(convos, desc="Tokenizing sequences"):
            x = self.tokenizer.encode_plus(
                convo,
                max_length=self.max_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            y = torch.clone(x["input_ids"])
            y[y == 0] = -100

            x["labels"] = y

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


class RedditQADataset(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        max_seq_len: int = 256,
        min_word_count: int = 2,
        model: str = "bert-base-uncased",
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

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def setup(self, stage: str):
        if self.vocab_size > 0:
            # already setup
            return

        mp.freeze_support()  # for Windows support
        tqdm.set_lock(mp.RLock())  # for managing output contention

        questions = pd.read_csv(self.questions, index_col=0, delimiter=";")
        answers = pd.read_csv(self.answers, index_col=1, delimiter=";")
        questions = questions.loc[questions.index.isin(answers.index)]

        threads = mp.cpu_count()

        per_thread = int(len(questions) / threads)
        args = [
            (
                questions.loc[questions.index[i : i + per_thread]],
                answers.loc[questions.index[i : i + per_thread]].copy(),
            )
            for i in range(0, len(questions), per_thread)
        ]

        with mp.Pool(threads, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
            qa_pairs = p.starmap(process_df, args)

        temp = []
        for qa in qa_pairs:
            temp.extend(qa)

        qa_pairs = temp

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

            y = torch.clone(x["input_ids"])
            y[y == 0] = -100

            x["labels"] = y

            tokenzize_pairs.append(x)

        self.train, self.test, _, _ = train_test_split(tokenzize_pairs, tokenzize_pairs, test_size=0.4)
        self.val, self.test, _, _ = train_test_split(self.test, self.test, test_size=0.5)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

    # have to define but not needed
    def prepare_data(self):
        return

    # have to define but not needed
    def prepare_data_per_node(self):
        return

    def decode(self, encoded_seq):
        return self.tokenizer.decode(encoded_seq)


class StandfordQADataset(pl.LightningDataModule):
    def __init__(
        self,
        max_seq_len: int = 256,
        model: str = "bert-base-uncased",
        batch_size: int = 16,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.vocab_size = 0

        if "uncased" in model:
            self.tokenizer = AutoTokenizer.from_pretrained(model, do_lower_case=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)

        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # https://github.com/tshrjn/Finetune-QA/blob/master/data.py
    @staticmethod
    def get_correct_alignement(context, answer):
        """Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here."""
        gold_text = answer["text"][0]
        start_idx = answer["answer_start"][0]
        end_idx = start_idx + len(gold_text)
        if context[start_idx:end_idx] == gold_text:
            return start_idx, end_idx  # When the gold label position is good
        elif context[start_idx - 1 : end_idx - 1] == gold_text:
            return start_idx - 1, end_idx - 1  # When the gold label is off by one character
        elif context[start_idx - 2 : end_idx - 2] == gold_text:
            return start_idx - 2, end_idx - 2  # When the gold label is off by two character
        else:
            raise ValueError()

    # Tokenize our training dataset
    def convert_to_features(self, batch):
        # Tokenize contexts and questions (as pairs of inputs)
        input_pairs = list(zip(batch["context"], batch["question"]))

        encodings = self.tokenizer.batch_encode_plus(
            input_pairs,
            return_token_type_ids=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )

        # Compute start and end tokens for labels using Transformers's fast tokenizers alignement methodes.
        start_positions, end_positions = [], []
        for i, (context, answer) in enumerate(zip(batch["context"], batch["answers"])):
            start_idx, end_idx = self.get_correct_alignement(context, answer)
            start_positions.append(encodings.char_to_token(i, start_idx))
            end_positions.append(encodings.char_to_token(i, end_idx - 1))

        if start_positions == [] and end_positions == []:
            start_positions.append(self.tokenizer.cls_token)
            end_positions.append(self.tokenizer.cls_token)

        encodings.update(
            {"start_positions": torch.LongTensor(start_positions), "end_positions": torch.LongTensor(end_positions)}
        )

        return encodings

    def setup(self, stage: str):
        if self.vocab_size > 0:
            # already setup
            return

        datasets = load_dataset("squad")

        vocab = set()

        for d in datasets["train"]:
            vocab |= set(d["question"].split(" ") + d["context"].split(" "))

        for d in datasets["validation"]:
            vocab |= set(d["question"].split(" ") + d["context"].split(" "))

        self.train = datasets["train"].map(self.convert_to_features, batched=True, batch_size=self.batch_size)
        self.val = datasets["validation"].map(self.convert_to_features, batched=True, batch_size=self.batch_size)

        self.train.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"],
        )

        self.val.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "start_positions", "end_positions"],
        )

        self.vocab_size = len(vocab)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, collate_fn=collate_fn)

    # have to define but not needed
    def prepare_data(self):
        return

    # have to define but not needed
    def prepare_data_per_node(self):
        return

    def decode(self, encoded_seq):
        return self.tokenizer.decode(encoded_seq)


if __name__ == "__main__":
    batch_size = 16
    """
    dataset = CornellDataset("data/cornell movie-dialogs corpus", max_seq_len=512, batch_size=batch_size)
    dataset.setup("fit")
    example = next(iter(dataset.train_dataloader()))
    print(example)
    print(example["input_ids"].shape)
    print(example["labels"].shape)
    print(example["attention_mask"].shape)
    print(example["token_type_ids"].shape)

    print(dataset.decode(example["input_ids"][0]))
    """

    dataset = RedditQADataset("data/reddit-qa", max_seq_len=256)
    dataset.setup("fit")
    example = next(iter(dataset.train_dataloader()))
    print(example)
    print(example["input_ids"].shape)
    print(example["labels"].shape)
    print(example["attention_mask"].shape)
    print(example["token_type_ids"].shape)

    # test dataloaders
    for batch in dataset.train_dataloader():
        for name, data in batch.items():
            try:
                # assert data.shape[0] == batch_size
                assert len(data.shape) > 1
            except:
                print(name)
                print(data.shape)
                raise ValueError("sad")

    for batch in dataset.val_dataloader():
        for name, data in batch.items():
            try:
                # assert data.shape[0] == batch_size
                assert len(data.shape) > 1
            except:
                print(name)
                print(data.shape)
                raise ValueError("sad")

    for batch in dataset.test_dataloader():
        for name, data in batch.items():
            try:
                # assert data.shape[0] == batch_size
                assert len(data.shape) > 1
            except:
                print(name)
                print(data.shape)
                raise ValueError("sad")

    """
    
    dataset = StandfordQADataset(max_seq_len=256)
    dataset.setup("fit")
    example = next(iter(dataset.val_dataloader()))
    print(example)
    print(example["input_ids"].shape)
    print(example["start_positions"].shape)
    print(example["end_positions"].shape)
    print(example["attention_mask"].shape)
    print(example["token_type_ids"].shape)

    print(dataset.decode(example["input_ids"][0]))
    """
