"""
Data preparation script for FineWeb-Edu dataset.

Streams the dataset from Hugging Face, tokenizes with GPT-2 tokenizer (tiktoken),
and writes sharded .npy files with a per-token document-level character feature.

Feature stored per token:
  char_end = chars_before_token + token_len_in_chars
          = character index in the document *after* this token (i.e., chars consumed)

We also append an explicit end-of-document (EOT) token after every document with char_end = 0
so downstream code can detect boundaries and reset any doc-level state.

Output shards are numpy arrays of shape (N, 2):
  col0: token_id (uint16)
  col1: char_end (uint32)

Usage:
    python data_prepare.py --subset sample-10BT --output data/fineweb_edu
    python data_prepare.py --sample --sample-docs 10000
"""

import os
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import tiktoken


def _byte_offset_to_char_index(text: str) -> np.ndarray:
    """
    Build a lookup table mapping UTF-8 byte offsets -> character indices.

    table[b] = number of unicode characters fully covered by the first b bytes
               of text.encode("utf-8").

    This allows converting cumulative bytes-consumed (from token bytes) into
    a correct character count in Python's unicode string.
    """
    b = text.encode("utf-8")
    table = np.empty(len(b) + 1, dtype=np.uint32)

    byte_pos = 0
    char_pos = 0
    table[0] = 0

    for ch in text:
        byte_pos += len(ch.encode("utf-8"))
        char_pos += 1
        table[byte_pos] = char_pos

    # Safety: fill monotone for any offsets not directly set (shouldn't occur)
    last = 0
    for i in range(len(table)):
        if table[i] < last:
            table[i] = last
        else:
            last = table[i]

    return table


def get_char_end_after_each_token(text: str, tokens: list[int], enc: tiktoken.Encoding) -> list[int]:
    """
    For each token, compute:
        char_end = chars_before_token + token_len_in_chars

    Implemented robustly by:
      - summing token lengths in UTF-8 BYTES via decode_single_token_bytes
      - converting cumulative byte offsets into character indices using a lookup table

    Returns:
        List[int] of length len(tokens)
    """
    if not tokens:
        return []

    text_bytes = text.encode("utf-8")
    byte_to_char = _byte_offset_to_char_index(text)

    bytes_consumed = 0
    char_end = []

    for t in tokens:
        tok_bytes = enc.decode_single_token_bytes(t)
        bytes_consumed += len(tok_bytes)

        # Clamp to avoid any index issues if something unexpected happens.
        if bytes_consumed > len(text_bytes):
            bytes_consumed = len(text_bytes)

        char_end.append(int(byte_to_char[bytes_consumed]))

    return char_end


class ShardWriter:
    """Writes tokenized (token_id, char_end) rows to sharded numpy files."""

    def __init__(self, output_dir: str, shard_size: int, prefix: str = "shard"):
        self.output_dir = output_dir
        self.shard_size = int(shard_size)
        self.prefix = prefix

        os.makedirs(output_dir, exist_ok=True)

        self.current_shard = 0
        self.token_buffer: list[int] = []
        self.char_end_buffer: list[int] = []
        self.total_rows = 0

    def add(self, tokens: list[int], char_end: list[int]):
        if len(tokens) != len(char_end):
            raise ValueError(f"Length mismatch: tokens={len(tokens)} vs char_end={len(char_end)}")

        self.token_buffer.extend(tokens)
        self.char_end_buffer.extend(char_end)
        self.total_rows += len(tokens)

        while len(self.token_buffer) >= self.shard_size:
            self._flush(self.shard_size)

    def _flush(self, n: int):
        shard_tokens = np.array(self.token_buffer[:n], dtype=np.uint16)
        shard_char_end = np.array(self.char_end_buffer[:n], dtype=np.uint32)

        shard_data = np.stack([shard_tokens, shard_char_end], axis=1)

        shard_path = os.path.join(self.output_dir, f"{self.prefix}_{self.current_shard:06d}.npy")
        np.save(shard_path, shard_data)

        print(f"Saved shard {self.current_shard}: {shard_path} ({n:,} rows)")

        self.token_buffer = self.token_buffer[n:]
        self.char_end_buffer = self.char_end_buffer[n:]
        self.current_shard += 1

    def finalize(self):
        if self.token_buffer:
            self._flush(len(self.token_buffer))

        metadata = {
            "total_rows": int(self.total_rows),
            "num_shards": int(self.current_shard),
            "shard_size": int(self.shard_size),
            "token_dtype": "uint16",
            "char_end_dtype": "uint32",
            "format": "(token_id, char_end_after_token_in_doc)",
            "note": "EOT token appended per document with char_end=0.",
        }
        np.save(os.path.join(self.output_dir, "metadata.npy"), metadata)

        print(f"\nTotal rows written: {self.total_rows:,}")
        print(f"Total shards: {self.current_shard}")


def prepare_dataset(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    subset: str = "sample-10BT",
    output_dir: str = "data/fineweb_edu",
    shard_size: int = 100_000_000,
    split: str = "train",
    val_ratio: float = 0.001,   # doc-level
):
    print(f"Loading dataset: {dataset_name}/{subset} (split={split}, streaming=True)")
    print(f"Output directory: {output_dir}")
    print(f"Shard size: {shard_size:,} rows")
    print(f"Validation ratio (doc-level): {val_ratio}")

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    ds = load_dataset(dataset_name, subset, split=split, streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256 for GPT-2

    train_writer = ShardWriter(train_dir, shard_size, prefix="train")
    val_writer = ShardWriter(val_dir, max(1, shard_size // 10), prefix="val")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    # Deterministic doc-level split: every k-th doc goes to val
    k = max(1, int(round(1.0 / val_ratio)))

    doc_idx = 0
    skipped = 0

    for doc in tqdm(ds, desc="Documents"):
        text = doc.get("text", "")
        if not text or not text.strip():
            skipped += 1
            continue

        tokens = enc.encode_ordinary(text)
        if not tokens:
            skipped += 1
            continue

        char_end = get_char_end_after_each_token(text, tokens, enc)

        # Append explicit doc boundary marker
        tokens.append(eot)
        char_end.append(0)

        if (doc_idx % k) == 0:
            val_writer.add(tokens, char_end)
        else:
            train_writer.add(tokens, char_end)

        doc_idx += 1

    print(f"\nSkipped empty/un-tokenizable docs: {skipped:,}")

    print("\nFinalizing training...")
    train_writer.finalize()

    print("\nFinalizing validation...")
    val_writer.finalize()

    print("\nDataset preparation complete!")


def prepare_small_sample(
    output_dir: str = "data/fineweb_edu_sample",
    num_docs: int = 10_000,
    shard_size: int = 1_000_000,
    val_ratio: float = 0.05,
):
    print(f"Preparing small sample: {num_docs:,} documents")
    print(f"Output directory: {output_dir}")
    print(f"Shard size: {shard_size:,} rows")
    print(f"Validation ratio (doc-level): {val_ratio}")

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    train_writer = ShardWriter(train_dir, shard_size, prefix="train")
    val_writer = ShardWriter(val_dir, max(1, shard_size // 10), prefix="val")

    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1")

    k = max(1, int(round(1.0 / val_ratio)))

    kept = 0
    skipped = 0

    for doc in tqdm(ds, total=num_docs, desc="Documents"):
        if kept >= num_docs:
            break

        text = doc.get("text", "")
        if not text or not text.strip():
            skipped += 1
            continue

        tokens = enc.encode_ordinary(text)
        if not tokens:
            skipped += 1
            continue

        char_end = get_char_end_after_each_token(text, tokens, enc)

        tokens.append(eot)
        char_end.append(0)

        if (kept % k) == 0:
            val_writer.add(tokens, char_end)
        else:
            train_writer.add(tokens, char_end)

        kept += 1

    print(f"\nKept docs: {kept:,}")
    print(f"Skipped docs: {skipped:,}")

    print("\nFinalizing training...")
    train_writer.finalize()

    print("\nFinalizing validation...")
    val_writer.finalize()

    print("\nSample preparation complete!")


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu dataset")
    parser.add_argument("--subset", type=str, default="sample-10BT", help="Dataset subset/config name")
    parser.add_argument("--output", type=str, default="data/fineweb_edu", help="Output directory")
    parser.add_argument("--shard-size", type=int, default=100_000_000, help="Rows (tokens) per shard")
    parser.add_argument("--val-ratio", type=float, default=0.001, help="Validation ratio (doc-level)")
    parser.add_argument("--sample", action="store_true", help="Prepare a small sample for testing")
    parser.add_argument("--sample-docs", type=int, default=10_000, help="Number of docs for sample")
    args = parser.parse_args()

    if args.sample:
        prepare_small_sample(
            output_dir=args.output + "_sample",
            num_docs=args.sample_docs,
            shard_size=max(1000, args.shard_size // 100),
            val_ratio=max(0.01, args.val_ratio * 50),
        )
    else:
        prepare_dataset(
            subset=args.subset,
            output_dir=args.output,
            shard_size=args.shard_size,
            val_ratio=args.val_ratio,
        )


if __name__ == "__main__":
    main()
