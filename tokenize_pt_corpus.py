from tqdm import tqdm
from datasets import Dataset
import numpy as np
from transformers import AutoTokenizer

# import argparse


def tokenize(sequences):
    input_ids = tokenizer(sequences, truncation=False)["input_ids"]
    # add EOS token
    input_ids = [ids + [tokenizer.eos_token_id] for ids in input_ids]
    length = [len(ids) for ids in input_ids]

    return {"input_ids": input_ids, "length": length}


def process_sql_corpus(examples):
    sequences = [sql for sql in examples["sql"]]
    sequences = [seq.strip() for seq in sequences]
    return tokenize(sequences)


def process_text2code(examples):
    sequences = [
        text + "\n" + code for text, code in zip(examples["text"], examples["code"])
    ]
    sequences = [sequence.strip() for sequence in sequences]
    return tokenize(sequences)


def process_text2text(examples):
    sequences = [
        input_text + "\n" + output_text
        for input_text, output_text in zip(
            examples["input_text"], examples["output_text"]
        )
    ]
    sequences = [sequence.strip() for sequence in sequences]
    return tokenize(sequences)


def get_slice(dataset, name):
    slices = []
    len_dataset = dataset.num_rows
    for size in [2, 4, 6, 8]:
        slice_size = len_dataset // size
        if name == "pure_sql":
            slices.append(dataset.select(range(0, slice_size * 2)))
        else:
            slices.append(dataset.select(range(0, slice_size)))

    return slices


def prepare(pt_corpus_dir, mode='sql'):
    # set `model_max_length` to a very large integer to avoid warning
    tokenizer.model_max_length = int(1e30)

    num_proc = 32

    data_dir = "./data_zip/codes_pretrain_corpus"
    pure_sql_dataset = Dataset.from_json(f"{data_dir}/pure_sql.jsonl")
    text2code_dataset = Dataset.from_json(f"{data_dir}/text2code.jsonl")
    text2text_dataset = Dataset.from_json(f"{data_dir}/text2text.jsonl")

    print(
        pure_sql_dataset.num_rows,
        text2code_dataset.num_rows,
        text2text_dataset.num_rows,
    )

    pure_sql_dataset = pure_sql_dataset.map(
        process_sql_corpus,
        num_proc=num_proc,
        desc="tokenizing the sql only dataset.",
        remove_columns=["sql"],
        batched=True,
    )

    print("Slicing dataset...")
    print(pure_sql_dataset)
    pure_sql_slices = get_slice(pure_sql_dataset, "pure_sql")
    
    if mode != 'sql':
        text2code_dataset = text2code_dataset.map(
            process_text2code,
            num_proc=num_proc,
            desc="tokenizing the text2code dataset.",
            remove_columns=["text", "code"],
            batched=True,
        )

        text2text_dataset = text2text_dataset.map(
            process_text2text,
            num_proc=num_proc,
            desc="tokenizing the text2text dataset.",
            remove_columns=["input_text", "output_text"],
            batched=True,
        )
        print(text2code_dataset)
        print(text2text_dataset)

        text2code_slices = get_slice(text2code_dataset, "text2code")
        text2text_slices = get_slice(text2text_dataset, "text2text")



    if mode == 'sql':
        for index, pure_sql in enumerate(pure_sql_slices):
            arr_len = np.sum(pure_sql["length"])
            print("There are {} tokens in the corpus".format(arr_len))

            arr = np.memmap(
                f"./pt_bin/SQL_{index}_{pt_corpus_dir}",
                dtype=np.uint16,
                mode="w+",
                shape=(arr_len,),
            )

            idx = 0
            total_batches = 2048

            for batch_idx in tqdm(range(total_batches)):
                # Batch together samples for faster write
                batch = pure_sql.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                ).with_format("numpy")
                arr_batch = np.concatenate(batch["input_ids"])

                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()
    else:
        for index, (pure_sql, text2code, text2text) in enumerate(
            zip(pure_sql_slices, text2code_slices, text2text_slices)
        ):
            final_corpus = [pure_sql, text2code, text2text]
            arr_len = sum(
                np.sum(tokenized_dataset["length"]) for tokenized_dataset in final_corpus
            )
            print("There are {} tokens in the corpus".format(arr_len))

            arr = np.memmap(
                f"./pt_bin/{index}_{pt_corpus_dir}",
                dtype=np.uint16,
                mode="w+",
                shape=(arr_len,),
            )  # (can do since starcoder's vocab size == 49152 is < 2**16 == 65536)
            idx = 0
            total_batches = 2048

            # concatenate all the ids in each dataset into one large file
            for tokenized_dataset in final_corpus:
                for batch_idx in tqdm(range(total_batches)):
                    # Batch together samples for faster write
                    batch = tokenized_dataset.shard(
                        num_shards=total_batches, index=batch_idx, contiguous=True
                    ).with_format("numpy")
                    arr_batch = np.concatenate(batch["input_ids"])

                    # Write into mmap
                    arr[idx : idx + len(arr_batch)] = arr_batch
                    idx += len(arr_batch)
                arr.flush()

    return


if __name__ == "__main__":
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "bigcode/starcoder", token="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"
    )
    prepare(pt_corpus_dir="tokenized_corpus.bin")
