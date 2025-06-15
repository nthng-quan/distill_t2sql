import json
import torch
import gc

from datasets import Dataset
from torch.utils.data import Dataset
from schema_item_filter import SchemaItemClassifierInference, filter_schema
from utils.db_utils import get_db_schema_sequence, get_matched_content_sequence


def prepare_text2sql_prefix_sequence(data):
    prefix_seq = (
        data["schema_sequence"]
        + "\n"
        + data["content_sequence"]
        + "\n"
        + data["text"]
        + "\n"
    )

    return prefix_seq


def prepare_inputs_and_labels_kd(
    prefix_seq, teacher_model, target_seq, tokenizer, max_tokens
):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)[
        "input_ids"
    ]
    target_ids = tokenizer(target_seq, truncation=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens:  # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else:  # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens - 1) :]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]

    input_ids = torch.tensor(attention_mask, dtype=torch.int64).to("cuda")
    attention_mask = torch.tensor(attention_mask, dtype=torch.int64).to("cuda")

    with torch.no_grad():
        teacher_logits = teacher_model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
        ).logits
        teacher_logits = teacher_logits.squeeze(0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": torch.tensor(labels, dtype=torch.int64),
        "teacher_logits": teacher_logits,
    }


def prepare_inputs_and_labels(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)[
        "input_ids"
    ]
    target_ids = tokenizer(target_seq, truncation=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens:  # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else:  # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens - 1) :]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def prepare_inputs_and_labels_ppo(prefix_seq, target_seq, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)[
        "input_ids"
    ]
    target_ids = tokenizer(target_seq, truncation=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens:  # pad inputs with pad_token_id
        # pad_length = max_tokens - seq_length

        input_ids = [tokenizer.pad_token_id] * (
            max_tokens - len(prefix_ids)
        ) + prefix_ids

        target_ids = [tokenizer.pad_token_id] * (
            max_tokens - len(target_ids)
        ) + target_ids

        # tell the model to ignore the padding tokens when performing (masked) self-attention
        attention_mask = [0] * (max_tokens - len(prefix_ids)) + [1] * (len(prefix_ids))
        # # only target_ids produces gradients
        # labels = [-100] * pad_length + [-100] * len(prefix_ids) + target_ids
    else:  # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens - 1) :]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
        "response": torch.tensor(target_ids, dtype=torch.int64),
    }


def prepare_inputs(prefix_seq, tokenizer, max_prefix_length):
    input_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)[
        "input_ids"
    ]

    if len(input_ids) > max_prefix_length:
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_prefix_length - 1) :]

    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
    }


def prepare_seqkd_inputs(prefix_seq, teacher_model, tokenizer, max_tokens):
    prefix_ids = [tokenizer.bos_token_id] + tokenizer(prefix_seq, truncation=False)[
        "input_ids"
    ]
    # generate target sequence from teacher model
    prefix_ids_ts = tokenizer(prefix_seq, truncation=False, return_tensors="pt").to(
        teacher_model.device
    )
    teacher_gen = teacher_model.generate(
        **prefix_ids_ts,
        max_new_tokens=256,
        num_beams=4,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )

    target_seq = tokenizer.decode(teacher_gen[:, len(prefix_ids) - 1 :][0])
    target_ids = tokenizer(target_seq, truncation=False)["input_ids"] + [
        tokenizer.eos_token_id
    ]

    seq_length = len(prefix_ids) + len(target_ids)
    if seq_length <= max_tokens:  # pad inputs with pad_token_id
        pad_length = max_tokens - seq_length
        input_ids = prefix_ids + target_ids + [tokenizer.pad_token_id] * pad_length
        # tell the model to ignore the padding tokens when performing (masked) self-attention
        attention_mask = [1] * seq_length + [0] * pad_length
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids + [-100] * pad_length
    else:  # no padding
        print("the current input sequence exceeds the max_tokens, we will truncate it.")
        input_ids = prefix_ids + target_ids
        # pre-truncate input ids
        input_ids = [tokenizer.bos_token_id] + input_ids[-(max_tokens - 1) :]
        attention_mask = [1] * max_tokens
        # only target_ids produces gradients
        labels = [-100] * len(prefix_ids) + target_ids
        # pre-truncate labels
        labels = labels[-max_tokens:]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.int64),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class SFTSQLGenerationDataset(Dataset):
    def __init__(
        self,
        text2sql_data_dir,
        tokenizer,
        max_tokens,
        mode,
        table_num,
        column_num,
        sic_path,
        teacher_model=None,
        # device="cuda",
    ):
        super().__init__()
        dataset = json.load(open(text2sql_data_dir))

        print("apply filtering strategies...")
        if mode in ["train", "seqkd", "ppo", "kd_sft"]:
            dataset = filter_schema(dataset, "train", None, table_num, column_num)
        elif mode == "eval":
            sic = SchemaItemClassifierInference(sic_path)
            dataset = filter_schema(dataset, "eval", sic, table_num, column_num)
            del sic
            torch.cuda.empty_cache()
        else:
            raise NotImplementedError

        # prepare schema sequence and content sequence
        for data in dataset:
            data["schema_sequence"] = get_db_schema_sequence(data["schema"])
            data["content_sequence"] = get_matched_content_sequence(
                data["matched_contents"]
            )

        self.mode = mode
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.teacher_model = teacher_model
        # self.device = device

    def __getitem__(self, index):
        data = self.dataset[index]
        prefix_seq = prepare_text2sql_prefix_sequence(data)
        if index < 2:
            print(prefix_seq)

        if self.mode == "train":
            target_seq = data["sql"]
            return prepare_inputs_and_labels(
                prefix_seq, target_seq, self.tokenizer, self.max_tokens
            )
            # return prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens)
        if self.mode == "kd_sft":
            target_seq = data["sql"]
            return prepare_inputs_and_labels_kd(
                prefix_seq,
                self.teacher_model,
                target_seq,
                self.tokenizer,
                self.max_tokens,
            )
        elif self.mode == "eval":
            return prepare_inputs(prefix_seq, self.tokenizer, self.max_tokens)
        elif self.mode == "seqkd":
            return prepare_seqkd_inputs(
                prefix_seq,
                self.teacher_model,
                self.tokenizer,
                self.max_tokens,
                # self.device,
            )
        elif self.mode == "ppo":
            target_seq = data["sql"]
            return prepare_inputs_and_labels_ppo(
                prefix_seq, target_seq, self.tokenizer, self.max_tokens
            )

    def __len__(self):
        return len(self.dataset)
