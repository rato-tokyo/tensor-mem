"""Data loading and preprocessing utilities for WikiText-2."""

from __future__ import annotations

import torch


def download_wikitext2() -> tuple[str, str, str]:
    """Download WikiText-2 dataset and return train/val/test text."""
    import ssl
    import urllib.request

    # Try datasets library first (most reliable for Hugging Face)
    try:
        print("Downloading WikiText-2 using datasets library...")
        from datasets import load_dataset

        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        train_text = "\n".join(dataset["train"]["text"])
        val_text = "\n".join(dataset["validation"]["text"])
        test_text = "\n".join(dataset["test"]["text"])

        return train_text, val_text, test_text

    except ImportError:
        print("datasets library not available, trying alternative source...")
    except Exception as e:
        print(f"datasets library failed: {e}, trying alternative source...")

    # Fallback: Try raw GitHub (PyTorch examples)
    try:
        print("Downloading WikiText-2 from PyTorch examples...")
        base_url = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"

        # Create SSL context that doesn't verify (for Colab compatibility)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        def fetch(url: str) -> str:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx) as response:
                return response.read().decode("utf-8")

        train_text = fetch(f"{base_url}/train.txt")
        val_text = fetch(f"{base_url}/valid.txt")
        test_text = fetch(f"{base_url}/test.txt")

        return train_text, val_text, test_text

    except Exception as e:
        print(f"Failed to download: {e}")
        raise RuntimeError(
            "Could not download WikiText-2. Please install datasets:\n  pip install datasets"
        ) from e


def build_vocab(text: str, max_vocab: int) -> tuple[dict[str, int], dict[int, str]]:
    """Build vocabulary from text."""
    from collections import Counter

    words = text.split()
    word_counts = Counter(words)

    # Reserve 0 for <unk>, 1 for <eos>
    vocab = {"<unk>": 0, "<eos>": 1}

    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


def tokenize(text: str, vocab: dict[str, int]) -> list[int]:
    """Tokenize text using vocabulary."""
    unk_id = vocab["<unk>"]
    eos_id = vocab["<eos>"]

    tokens = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            for word in line.split():
                tokens.append(vocab.get(word, unk_id))
            tokens.append(eos_id)

    return tokens


def batchify(data: list[int], batch_size: int, device: torch.device) -> torch.Tensor:
    """Reshape data into [seq_len, batch_size] for language modeling."""
    nbatch = len(data) // batch_size
    data = data[: nbatch * batch_size]
    data = torch.tensor(data, dtype=torch.long, device=device)
    return data.view(batch_size, -1).t().contiguous()


def get_batch(source: torch.Tensor, i: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Get a batch of data for language modeling."""
    seq_len = min(seq_len, len(source) - 1 - i)
    data = source[i : i + seq_len].t()  # [batch, seq_len]
    target = source[i + 1 : i + 1 + seq_len].t()  # [batch, seq_len]
    return data, target
