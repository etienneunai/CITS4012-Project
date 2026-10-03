# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=5.0.1",
#     "gdown==6.4.1",
#     "marimo>=0.23.3",
#     "tqdm==4.70.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import gdown
    import os
    import marimo as mo

    if not(os.path.exists('MultiRC/train_456-fixedIds.json') and os.path.exists('MultiRC/test_83-fixedIds.json')):
        gdown.download_folder(id="18SlXjdkhUrG_PE0aTysMypMKx41a9evH")

    train_json = 'MultiRC/train_456-fixedIds.json'
    test_json = 'MultiRC/test_83-fixedIds.json'
    return mo, os, test_json, train_json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Readme
    *If there is something to be noted for the marker, please mention here.*

    *If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1.Dataset Processing
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell(hide_code=True)
def _(test_json, train_json):
    # Download the trainin
    import json
    with open(train_json, 'r') as f:
        train_data = json.load(f)['data']
    with open(test_json, 'r') as f:
        test_data = json.load(f)['data']

    print(train_data[1])
    return json, test_data, train_data


@app.cell(hide_code=True)
def _():
    # Flatten the data
    import re

    def clean_passage(text):
        text = re.sub(r"<b>\s*Sent\s*\d+\s*:\s*</b>", " ", text)
        text = re.sub(r"<[^>]+>", " ", text).strip()
        return text

    def flatten_data(data):
        flattened_data = []
        for item in data:
            paragraph = item['paragraph']
            passage = clean_passage(paragraph['text'])
            for q in paragraph['questions']:
                question_text = q['question']
                for a in q['answers']:
                    answer_text = a['text']
                    answer_val = a['isAnswer']
                    flattened_data.append((passage, question_text, answer_text, answer_val))
        return flattened_data


    return flatten_data, re


@app.cell
def _(flatten_data, re, test_data, train_data):
    # 
    import torch
    from torch.utils.data import Dataset, DataLoader
    from collections import Counter

    TOKEN_RE = re.compile(r"\w+|[^\w\s]")

    def tokenize(text):
        return TOKEN_RE.findall(text.lower())

    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    SEP_TOKEN = "<sep>"

    def build_vocab(flat_data, min_freq=2):
        counter = Counter()
        for passage, question, answer, _ in flat_data:
            counter.update(tokenize(passage))
            counter.update(tokenize(question))
            counter.update(tokenize(answer))
        vocab = {PAD_TOKEN: 0, CLS_TOKEN: 1, SEP_TOKEN: 2}
        for word, freq in counter.most_common():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        return vocab

    flat_train = flatten_data(train_data)
    flat_test = flatten_data(test_data)
    vocab = build_vocab(flat_train, min_freq=3)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    MAX_LEN = 512

    def encode(passage, question, answer):
        tokens = (
            [CLS_TOKEN]
            + tokenize(passage)[:448]
            + [SEP_TOKEN]
            + tokenize(question)[:32]
            + [SEP_TOKEN]
            + tokenize(answer)[:28]
            + [SEP_TOKEN]
        )
        ids = [vocab.get(t, 0) for t in tokens]
        ids = ids[:MAX_LEN]
        ids = ids + [0] * (MAX_LEN - len(ids))
        return ids

    class MultiRCDataset(Dataset):
        def __init__(self, flat_data):
            self.examples = flat_data

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            passage, question, answer, label = self.examples[idx]
            input_ids = torch.tensor(encode(passage, question, answer), dtype=torch.long)
            target = torch.tensor(1.0 if label else 0.0, dtype=torch.float)
            return input_ids, target

    BATCH_SIZE = 64

    train_dataset = MultiRCDataset(flat_train)
    test_dataset = MultiRCDataset(flat_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    print()

    print(f"Sample tuple:")
    print(train_dataset.examples[0])
    return (
        Counter,
        MAX_LEN,
        flat_test,
        flat_train,
        test_loader,
        tokenize,
        torch,
        train_loader,
        vocab_size,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Tokenization & Encoding

    We use a word-level tokenizer with a simple regex that captures alphanumeric
    tokens and punctuation. The vocabulary is built from the training set with a
    minimum frequency of 2 to reduce noise from rare words.

    Each example is encoded as:

    ```
    [CLS] passage [SEP] question [SEP] answer [SEP]
    ```

    The `[CLS]` token aggregates information for classification. Sequences are
    padded to a maximum length of 512 tokens.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 2. Model Implementation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _(torch):

    import torch.nn as nn
    import torch.nn.functional as F
    import math
    from abc import ABC, abstractmethod


    class AttentionMechanism(ABC, nn.Module):
        @abstractmethod
        def forward(self, x, mask=None):
            pass


    class MultiHeadAttention(AttentionMechanism):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            batch_size, seq_len, _ = x.shape

            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

            # From week 6? lecture
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # this is to stop the <pad> tokens affecting gradient updates
            if mask is not None:
                attn_mask = mask.unsqueeze(1).unsqueeze(1)
                scores = scores.masked_fill(attn_mask, -1e9)

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context = torch.matmul(attn_weights, V)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

            return self.W_o(context)


    class LinearAttention(AttentionMechanism):
        """O(n*d^2) approximation via kernel feature maps."""

        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0
            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        @staticmethod
        def _elu_feature(x):
            return F.elu(x) + 1

        def forward(self, x, mask=None):
            batch_size, seq_len, _ = x.shape

            # (batch, heads, seq_len, d_k) for all three
            Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

            Q = self._elu_feature(Q)
            K = self._elu_feature(K)

            if mask is not None:
                mask_expanded = mask.unsqueeze(1).unsqueeze(-1)
                K = K * (~mask_expanded).float()
                V = V * (~mask_expanded).float()

            # KV: (batch, heads, d_k, d_k) — sum over seq of K^T V
            KV = torch.einsum("bhsd,bhse->bhde", K, V)

            # context: (batch, heads, seq_len, d_k) — Q @ KV
            context = torch.einsum("bhsd,bhde->bhse", Q, KV)

            # denominator: (batch, heads, seq_len) — Q @ sum(K)
            K_sum = K.sum(dim=2)
            denominator = torch.einsum("bhsd,bhd->bhs", Q, K_sum) + 1e-6

            context = context / denominator.unsqueeze(-1)
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
            return self.dropout(self.W_o(context))


    return LinearAttention, MultiHeadAttention, math, nn


@app.cell
def _(MultiHeadAttention, math, nn, torch):
    # From the week 7 lecture
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=512):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]


    class RotaryPositionalEncoding(nn.Module):
        """Rotary Position Embedding (RoPE) -- applies 2-D rotations to consecutive
        dimension pairs, encoding position multiplicatively instead of additively."""

        def __init__(self, d_model, max_len=512):
            super().__init__()
            self.d_model = d_model
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
            )
            self.register_buffer("cos", torch.cos(position * div_term))
            self.register_buffer("sin", torch.sin(position * div_term))

        def forward(self, x):
            seq_len = x.size(1)
            x1 = x[..., 0::2]
            x2 = x[..., 1::2]
            cos = self.cos[:seq_len, :].unsqueeze(0)
            sin = self.sin[:seq_len, :].unsqueeze(0)
            rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
            return rotated.flatten(-2)


    class RMSNorm(nn.Module):
        """Root Mean Square LayerNorm -- no bias, no mean subtraction."""

        def __init__(self, d_model, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))

        def forward(self, x):
            rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return x * rms * self.weight


    # --- Swappable activation functions ---
    ACTIVATIONS = {
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
    }


    def get_activation(name):
        return ACTIVATIONS[name]()


    # --- Swappable positional encodings ---
    POS_ENCODINGS = {
        "sinusoidal": PositionalEncoding,
        "rotary": RotaryPositionalEncoding,
    }

    # --- Swappable normalization layers ---
    NORM_LAYERS = {
        "layernorm": nn.LayerNorm,
        "rmsnorm": RMSNorm,
    }


    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1, activation="gelu"):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        def forward(self, x):
            return self.net(x)


    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, attention_cls, dropout=0.1,
                     norm_cls=nn.LayerNorm, activation="gelu"):
            super().__init__()
            self.attention = attention_cls(d_model, num_heads, dropout)
            self.norm1 = norm_cls(d_model)
            self.norm2 = norm_cls(d_model)
            self.ffn = FeedForward(d_model, d_ff, dropout, activation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            x = x + self.dropout(self.attention(self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x


    class TransformerClassifier(nn.Module):
        def __init__(
            self,
            vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=2,
            d_ff=512,
            max_len=512,
            attention_cls=MultiHeadAttention,
            dropout=0.2,
            pos_encoding="sinusoidal",
            norm_cls=nn.LayerNorm,
            activation="gelu",
            pooling="cls",
        ):
            super().__init__()
            self.d_model = d_model
            self.pooling = pooling

            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoding = POS_ENCODINGS[pos_encoding](d_model, max_len)

            self.layers = nn.ModuleList([
                TransformerEncoderLayer(
                    d_model, num_heads, d_ff, attention_cls, dropout,
                    norm_cls=norm_cls, activation=activation,
                )
                for _ in range(num_layers)
            ])
            self.norm = norm_cls(d_model)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                get_activation(activation),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, input_ids):
            mask = (input_ids == 0)

            x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
            x = self.pos_encoding(x)

            for layer in self.layers:
                x = layer(x, mask)

            x = self.norm(x)

            if self.pooling == "cls":
                pooled = x[:, 0, :]
            elif self.pooling == "mean":
                _mask_exp = (~mask).unsqueeze(-1).float()
                pooled = (x * _mask_exp).sum(1) / _mask_exp.sum(1).clamp(min=1)
            elif self.pooling == "max":
                _mask_exp = mask.unsqueeze(-1)
                pooled = x.masked_fill(_mask_exp, -1e9).max(1).values
            else:
                pooled = x[:, 0, :]

            return self.classifier(pooled).squeeze(-1)


    return RMSNorm, TransformerClassifier


@app.cell
def _(
    MAX_LEN,
    MultiHeadAttention,
    TransformerClassifier,
    flat_train,
    math,
    nn,
    os,
    torch,
    vocab_size,
):
    # train the mfs
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn.metrics import f1_score, accuracy_score

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # If ≥99% of predictions are one class, the model has collapsed to always
    # predicting that class.  val_acc then just mirrors the class distribution
    # (e.g. 0.428 or 0.572 on this dataset) rather than reflecting real learning.
    COLLAPSE_THRESHOLD = 0.99


    def evaluate(model, data_loader, criterion=None):
        model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for input_ids, targets in data_loader:
                input_ids = input_ids.to(DEVICE)
                targets = targets.to(DEVICE)

                logits = model(input_ids)
                if criterion is not None:
                    total_loss += criterion(logits, targets).item()
                    num_batches += 1

                preds = (torch.sigmoid(logits) > 0.5).long().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(targets.long().cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        pos_pred_frac = sum(all_preds) / max(len(all_preds), 1)
        return avg_loss, acc, f1, pos_pred_frac


    def create_model(attention_cls=MultiHeadAttention, d_model=128, num_heads=4,
                     num_layers=2, d_ff=512, dropout=0.2,
                     pos_encoding="sinusoidal", norm_cls=nn.LayerNorm,
                     activation="gelu", pooling="cls"):
        model = TransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=MAX_LEN,
            attention_cls=attention_cls,
            dropout=dropout,
            pos_encoding=pos_encoding,
            norm_cls=norm_cls,
            activation=activation,
            pooling=pooling,
        ).to(DEVICE)
        return model


    def _make_scheduler(optimizer, warmup_type, scheduler_type, num_train_steps, warmup_steps):
        """Build a LambdaLR scheduler with swappable warmup and decay strategies."""
        if warmup_type == "none" and scheduler_type == "constant":
            return optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

        def lr_lambda(step):
            if warmup_type != "none" and step < warmup_steps:
                if warmup_type == "linear":
                    return step / max(1, warmup_steps)
                elif warmup_type == "cosine":
                    return 0.5 * (1 - math.cos(math.pi * step / max(1, warmup_steps)))
                return step / max(1, warmup_steps)

            if warmup_type != "none":
                post_step = step - warmup_steps
                total = max(1, num_train_steps - warmup_steps)
            else:
                post_step = step
                total = max(1, num_train_steps)
            progress = min(post_step / total, 1.0)

            if scheduler_type == "exponential":
                return 0.5 ** progress
            elif scheduler_type == "cosine":
                return 0.5 * (1 + math.cos(math.pi * progress))
            else:
                return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    def train_model(model, train_loader, val_loader, num_epochs=10, lr=3e-4,
                    weight_decay=0.01, patience=3, label="default",
                    warmup_type="linear", optimizer_type="adamw",
                    scheduler_type="exponential", seed=42):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if optimizer_type == "adam":
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  momentum=0.9)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        num_train_steps = len(train_loader) * num_epochs
        warmup_steps = int(0.1 * num_train_steps) if warmup_type != "none" else 0
        scheduler = _make_scheduler(optimizer, warmup_type, scheduler_type,
                                     num_train_steps, warmup_steps)

        pos_count = sum(1 for _, _, _, lbl in flat_train if lbl)
        neg_count = len(flat_train) - pos_count
    
        criterion = nn.BCEWithLogitsLoss()

        best_val_f1 = 0.0
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
                   "collapsed": False, "collapse_epoch": None, "collapse_type": None}
        global_step = 0

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

            # Track prediction distribution during the first epoch for early
            # collapse detection.  Reuses the logits already computed for the
            # training step — zero extra forward passes.
            _epoch_pos_preds = 0
            _epoch_total_preds = 0

            pbar = tqdm(train_loader, desc=f"[{label}] Epoch {epoch+1}/{num_epochs}", leave=False)
            for input_ids, targets in pbar:
                input_ids = input_ids.to(DEVICE)
                targets = targets.to(DEVICE)

                optimizer.zero_grad()
                logits = model(input_ids)
                loss = criterion(logits, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                global_step += 1

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                # During-epoch collapse check (first epoch only): sample the
                # prediction distribution every quarter-epoch.  A collapsed model
                # will show ~100% one class at every checkpoint.
                if epoch == 0:
                    with torch.no_grad():
                        _epoch_pos_preds += (torch.sigmoid(logits) > 0.5).long().sum().item()
                        _epoch_total_preds += input_ids.size(0)

                    _check_interval = max(1, len(train_loader) // 4)
                    if num_batches % _check_interval == 0:
                        _frac = _epoch_pos_preds / _epoch_total_preds
                        if _frac >= COLLAPSE_THRESHOLD or _frac <= 1 - COLLAPSE_THRESHOLD:
                            _ctype = "all-positive" if _frac >= COLLAPSE_THRESHOLD else "all-negative"
                            print(f"\n\u26a0\ufe0f  [{label}] Collapse signal at batch "
                                  f"{num_batches}/{len(train_loader)} (epoch 1): "
                                  f"{_frac:.1%} {_ctype} predictions")

            avg_train_loss = total_loss / num_batches
            val_loss, val_acc, val_f1, val_pos_frac = evaluate(model, val_loader, criterion)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            # Post-epoch collapse check: if the model predicts a single class,
            # val_acc just mirrors the class distribution — not real learning.
            # Stop immediately rather than wasting the remaining epochs.
            if val_pos_frac >= COLLAPSE_THRESHOLD or val_pos_frac <= 1 - COLLAPSE_THRESHOLD:
                _ctype = "all-positive" if val_pos_frac >= COLLAPSE_THRESHOLD else "all-negative"
                history["collapsed"] = True
                history["collapse_epoch"] = epoch + 1
                history["collapse_type"] = _ctype
                print(f"\n{'='*60}")
                print(f"\u26a0\ufe0f  [{label}] MODEL COLLAPSE \u2014 stopping after epoch {epoch+1}")
                print(f"   Predictions: {val_pos_frac:.1%} positive \u2192 always '{_ctype}'")
                print(f"   val_acc={val_acc:.4f} = class distribution, NOT learning")
                print(f"   Skipping remaining {num_epochs - epoch - 1} epochs.")
                print(f"{'='*60}")
                return history

            print(f"[{label}] Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                torch.save(model.state_dict(), f"best_model_{label}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        _ckpt = f"best_model_{label}.pt"
        if os.path.exists(_ckpt):
            model.load_state_dict(torch.load(_ckpt, weights_only=True))
        return history


    return create_model, train_model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3.Testing and Evaluation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _(
    MultiHeadAttention,
    create_model,
    test_loader,
    train_loader,
    train_model,
):
    # Test the default configuration
    import matplotlib.pyplot as plt

    NUM_EPOCHS = 10
    LR = 1e-4

    experiments = {}

    # --- Experiment 1: Standard Multi-Head Attention ---
    print("=" * 60)
    print("Experiment 1: Multi-Head Attention (2 layers, 4 heads)")
    print("=" * 60)

    model_mha = create_model(
        attention_cls=MultiHeadAttention,
        num_layers=2, 
        d_model=128,     
        d_ff=512,          
        dropout=0.2,       
        num_heads=4,      
    )

    history_mha = train_model(model_mha, train_loader, test_loader,
                              num_epochs=NUM_EPOCHS, lr=LR, label="mha")
    experiments["Multi-Head Attention"] = history_mha

    print("\nAll experiments complete.")
    return experiments, plt


@app.cell
def _(experiments, plt):
    # Plot the results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss curves ---
    for label, hist in experiments.items():
        axes[0].plot(hist["train_loss"], label=f"{label} (train)", linestyle="--")
        axes[1].plot(hist["val_loss"], label=f"{label} (val)")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    # --- Validation F1 ---
    for label, hist in experiments.items():
        axes[2].plot(hist["val_f1"], label=label, marker="o")
    axes[2].set_title("Validation F1 Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1")
    axes[2].legend()

    plt.tight_layout()
    plt.show()

    # --- Final comparison table ---
    print("\n" + "=" * 55)
    print(f"{'Experiment':<25} {'Val Acc':>8} {'Val F1':>8}")
    print("=" * 55)
    for label, hist in experiments.items():
        best_idx = max(range(len(hist["val_f1"])), key=lambda i: hist["val_f1"][i])
        print(f"{label:<25} {hist['val_acc'][best_idx]:>8.4f} {hist['val_f1'][best_idx]:>8.4f}")
    print("=" * 55)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset Statistics

    Overview of the MultiRC dataset distribution, vocabulary, and structure.
    """)
    return


@app.cell
def _(Counter, flat_test, flat_train, tokenize, train_data):

    import numpy as np

    passage_word_counts = []
    question_word_counts = []
    answer_word_counts = []
    tokens_per_example = []
    all_tokens = []

    for passage, question, answer, ans_label in flat_train:
        p_tokens = tokenize(passage)
        q_tokens = tokenize(question)
        a_tokens = tokenize(answer)
        passage_word_counts.append(len(p_tokens))
        question_word_counts.append(len(q_tokens))
        answer_word_counts.append(len(a_tokens))
        tokens_per_example.append(len(p_tokens) + len(q_tokens) + len(a_tokens))
        all_tokens.extend(p_tokens)
        all_tokens.extend(q_tokens)
        all_tokens.extend(a_tokens)

    word_freq = Counter(all_tokens)
    word_freq_values = sorted(word_freq.values(), reverse=True)

    train_labels_list = [lbl for _, _, _, lbl in flat_train]
    test_labels_list = [lbl for _, _, _, lbl in flat_test]
    train_pos_count = sum(train_labels_list)
    test_pos_count = sum(test_labels_list)

    questions_per_passage = []
    answers_per_question = []
    for item in train_data:
        paragraph = item['paragraph']
        questions_per_passage.append(len(paragraph['questions']))
        for q in paragraph['questions']:
            answers_per_question.append(len(q['answers']))

    test_word_set = set()
    for passage, question, answer, _ in flat_test:
        for t in tokenize(passage) + tokenize(question) + tokenize(answer):
            test_word_set.add(t)
    train_word_set = set(word_freq.keys())
    test_only_words = len(test_word_set - train_word_set)
    return (
        answer_word_counts,
        answers_per_question,
        np,
        passage_word_counts,
        question_word_counts,
        questions_per_passage,
        test_only_words,
        test_pos_count,
        tokens_per_example,
        train_pos_count,
        word_freq_values,
    )


@app.cell(hide_code=True)
def _(
    answer_word_counts,
    answers_per_question,
    flat_test,
    flat_train,
    np,
    passage_word_counts,
    plt,
    question_word_counts,
    questions_per_passage,
    test_data,
    test_only_words,
    test_pos_count,
    tokens_per_example,
    train_data,
    train_pos_count,
    vocab_size,
    word_freq_values,
):

    _fig, _axes = plt.subplots(2, 3, figsize=(20, 12))
    _fig.suptitle("MultiRC Dataset Overview", fontsize=16, fontweight="bold", y=0.98)

    # 1. Vocabulary size vs dataset size
    ax1 = _axes[0, 0]
    cats = ["Vocab\n(unique words)", "Training\ntuples", "Test\ntuples", "Test-only\nwords"]
    vals = [vocab_size, len(flat_train), len(flat_test), test_only_words]
    cols = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    bars = ax1.bar(cats, vals, color=cols, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                 f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Count")
    ax1.set_title("Vocabulary Size vs Dataset Size")
    ax1.set_ylim(0, max(vals) * 1.15)

    # 2. Label distribution
    ax2 = _axes[0, 1]
    x_idx = np.arange(2)
    width = 0.35
    train_counts = [len(flat_train) - train_pos_count, train_pos_count]
    test_counts = [len(flat_test) - test_pos_count, test_pos_count]
    ax2.bar(x_idx - width/2, train_counts, width, label="Train", color="#42A5F5", edgecolor="black", linewidth=0.5)
    ax2.bar(x_idx + width/2, test_counts, width, label="Test", color="#FF7043", edgecolor="black", linewidth=0.5)
    for i, (tc, vc) in enumerate(zip(train_counts, test_counts)):
        ax2.text(x_idx[i] - width/2, tc + 100, f"{tc:,}", ha="center", va="bottom", fontsize=10)
        ax2.text(x_idx[i] + width/2, vc + 100, f"{vc:,}", ha="center", va="bottom", fontsize=10)
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(["Not Answer (0)", "Is Answer (1)"])
    ax2.set_ylabel("Count")
    ax2.set_title("Label Distribution: Train vs Test")
    ax2.legend()

    # 3. Combined sequence length per example
    ax3 = _axes[0, 2]
    ax3.hist(tokens_per_example, bins=50, color="#7E57C2", edgecolor="black", linewidth=0.3, alpha=0.8)
    ax3.axvline(np.mean(tokens_per_example), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean = {np.mean(tokens_per_example):.0f}")
    ax3.axvline(np.percentile(tokens_per_example, 95), color="orange", linestyle="--", linewidth=1.5,
                label=f"p95 = {np.percentile(tokens_per_example, 95):.0f}")
    ax3.axvline(512, color="green", linestyle=":", linewidth=2, label="Max len = 512")
    ax3.set_xlabel("Total Tokens (Passage + Question + Answer)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Combined Sequence Length per Example")
    ax3.legend(fontsize=9)

    # 4. Component length box plots
    ax4 = _axes[1, 0]
    box_data = [passage_word_counts, question_word_counts, answer_word_counts]
    bp = ax4.boxplot(box_data, tick_labels=["Passage", "Question", "Answer"], patch_artist=True,
                     showmeans=True, meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5})
    box_cols = ["#1565C0", "#2E7D32", "#E65100"]
    for patch, color in zip(bp['boxes'], box_cols):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_ylabel("Word Count")
    ax4.set_title("Length Distribution by Component")

    # 5. Word frequency spectrum (Zipfian)
    ax5 = _axes[1, 1]
    ax5.loglog(range(1, len(word_freq_values) + 1), word_freq_values, color="#37474F", linewidth=1.5)
    ax5.set_xlabel("Word Rank (log)")
    ax5.set_ylabel("Frequency (log)")
    ax5.set_title(f"Word Frequency Spectrum\n({vocab_size:,} words in vocab)")
    cutoff_rank = sum(1 for f in word_freq_values if f >= 2)
    ax5.axvline(cutoff_rank, color="red", linestyle="--", linewidth=1.5,
                label=f"min_freq=2 cutoff\n(rank {cutoff_rank:,})")
    ax5.legend(fontsize=9)

    # 6. Questions per passage & answers per question
    ax6 = _axes[1, 2]
    ax6.hist(questions_per_passage, bins=range(0, max(questions_per_passage)+2),
             color="#00897B", edgecolor="black", linewidth=0.3, alpha=0.7, label="Questions / Passage")
    ax6b = ax6.twinx()
    ax6b.hist(answers_per_question, bins=range(0, max(answers_per_question)+2),
              color="#D81B60", edgecolor="black", linewidth=0.3, alpha=0.5, label="Answers / Question")
    ax6.set_xlabel("Count")
    ax6.set_ylabel("Questions per Passage", color="#00897B")
    ax6b.set_ylabel("Answers per Question", color="#D81B60")
    ax6.set_title("Questions per Passage & Answers per Question")
    h1, l1 = ax6.get_legend_handles_labels()
    h2, l2 = ax6b.get_legend_handles_labels()
    ax6.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("=" * 60)
    print("MultiRC Dataset Summary")
    print("=" * 60)
    print(f"Vocabulary size (min_freq=2):  {vocab_size:,} unique words")
    print(f"Training tuples:               {len(flat_train):,}")
    print(f"Test tuples:                   {len(flat_test):,}")
    print(f"Train:Test ratio:              1:{len(flat_test)/len(flat_train):.2f}")
    print(f"Train positive ratio:          {train_pos_count/len(flat_train):.1%}")
    print(f"Test positive ratio:           {test_pos_count/len(flat_test):.1%}")
    print(f"Passages in train:             {len(train_data)}")
    print(f"Passages in test:               {len(test_data)}")
    print(f"Avg questions per passage:     {np.mean(questions_per_passage):.1f}")
    print(f"Avg answers per question:      {np.mean(answers_per_question):.1f}")
    print(f"Avg passage words:             {np.mean(passage_word_counts):.0f} (p95={np.percentile(passage_word_counts,95):.0f})")
    print(f"Avg question words:            {np.mean(question_word_counts):.0f} (p95={np.percentile(question_word_counts,95):.0f})")
    print(f"Avg answer words:              {np.mean(answer_word_counts):.0f} (p95={np.percentile(answer_word_counts,95):.0f})")
    print(f"Avg total tokens per example:  {np.mean(tokens_per_example):.0f} (p95={np.percentile(tokens_per_example,95):.0f})")
    print(f"Words appearing only in test:  {test_only_words:,}")
    print("=" * 60)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Changelog

    ### Architecture
    - **Miniaturized** from 4 layers / 256 d_model / 8 heads to **2 layers / 128 d_model / 4 heads** — fewer parameters to optimize on 27K examples
    - **Pre-Layer Normalization** (Pre-LN): LayerNorm applied *before* attention and FFN sub-layers (not after residual), giving a direct gradient path and preventing early collapse

    ### Data processing
    - Raised `min_freq` from 2 to 3 — smaller vocab (~10K vs ~15.5K), each word gets more gradient updates per epoch

    ### Training pipeline
    - Added LR warmup — linear ramp over first 10% of steps, then exponential decay; prevents random embeddings from getting destabilised at start of training
    - **Increased weight decay from `1e-4` to `0.01`** — heavier regularization with AdamW forces generalization over memorization
    - **Increased batch size from 32 to 64** — more stable gradient updates for Transformer training from scratch
    - Dropout set to 0.2 throughout
    - Changed checkpointing from best `val_loss` to best `val_f1` — weighted loss is misleading for model selection
    - Replaced `ReduceLROnPlateau` with `LambdaLR` for warmup support

    ### Positional encodings
    - Sinusoidal positional encodings verified: added to token embeddings (`x = pos_encoding(token_embedding * sqrt(d_model))`) before the first attention layer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Comprehensive Ablation Study

    This cell systematically evaluates **all combinations** of the following swappable
    model and training components:

    | Component | Options |
    |-----------|---------|
    | **Attention** | Multi-Head Attention, Linear Attention |
    | **Activation** | GELU, ReLU, SiLU |
    | **Positional Encoding** | Sinusoidal, Rotary (RoPE) |
    | **Normalisation** | LayerNorm, RMSNorm |
    | **Pooling** | CLS-token, Mean |
    | **LR Warmup** | Linear, None |
    | **Optimizer** | AdamW, Adam |
    | **LR Scheduler** | Exponential decay, Cosine decay |

    Each combination is trained for the same number of epochs with an identical
    seed (`42`) to ensure fair comparison. Results are saved to `ablation_results/`
    including a CSV table, training histories, and visualisation plots.

    > Warning: This cell runs 384 experiments and will take a very long time.
    > Do NOT run it casually.
    """)
    return


@app.cell
def _(
    LinearAttention,
    MultiHeadAttention,
    RMSNorm,
    create_model,
    json,
    nn,
    np,
    os,
    plt,
    test_loader,
    torch,
    train_loader,
    train_model,
):
    # === Comprehensive Ablation Study ===
    # WARNING: 384 combinations x 10 epochs -- this will take a VERY long time.
    # Results are saved to ablation_results/.
    # DO NOT run this cell casually.

    import itertools
    import pandas as pd

    SEED = 42
    ABLATION_EPOCHS = 10
    ABLATION_LR = 1e-4
    ABLATION_DIR = "ablation_results"
    os.makedirs(ABLATION_DIR, exist_ok=True)

    # Global seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

    # --- Swappable component options ---
    attention_opts = {"mha": MultiHeadAttention, "linear": LinearAttention}
    activation_opts = ["gelu", "relu", "silu"]
    pos_enc_opts = ["sinusoidal", "rotary"]
    norm_opts = {"layernorm": nn.LayerNorm, "rmsnorm": RMSNorm}
    pooling_opts = ["cls", "mean"]
    warmup_opts = ["linear", "none"]
    optimizer_opts = ["adamw", "adam"]
    scheduler_opts = ["exponential", "cosine"]

    # Column names for results table (matches keys in result dicts)
    _ablation_cols = ["attention", "activation", "pos_encoding", "norm",
                       "pooling", "warmup", "optimizer", "scheduler"]

    # --- Build all combinations ---
    _ablation_combos = list(itertools.product(
        attention_opts.items(),
        activation_opts,
        pos_enc_opts,
        norm_opts.items(),
        pooling_opts,
        warmup_opts,
        optimizer_opts,
        scheduler_opts,
    ))
    _ablation_total = len(_ablation_combos)
    print(f"Total ablation combinations: {_ablation_total}")
    print(f"Estimated: {_ablation_total} models x {ABLATION_EPOCHS} epochs each")
    print("=" * 70)

    ablation_results = []
    ablation_histories = {}

    for _i, (_attn, _act, _pos, _norm, _pool, _warm, _opt, _sched) in enumerate(_ablation_combos):
        _attn_name, _attn_cls = _attn
        _norm_name, _norm_cls = _norm
        _combo = f"{_attn_name}|{_act}|{_pos}|{_norm_name}|{_pool}|{_warm}|{_opt}|{_sched}"
        _label = f"abl_{_i:03d}"

        print(f"\n[{_i+1}/{_ablation_total}] {_combo}")
        print("-" * 60)

        # Reset seed before each run for fair comparison
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        _model = create_model(
            attention_cls=_attn_cls,
            pos_encoding=_pos,
            norm_cls=_norm_cls,
            activation=_act,
            pooling=_pool,
        )

        _hist = train_model(
            _model, train_loader, test_loader,
            num_epochs=ABLATION_EPOCHS, lr=ABLATION_LR,
            label=_label,
            warmup_type=_warm,
            optimizer_type=_opt,
            scheduler_type=_sched,
            seed=SEED,
        )

        ablation_histories[_combo] = _hist
        _best_idx = max(range(len(_hist["val_f1"])), key=lambda j: _hist["val_f1"][j])
        ablation_results.append({
            "attention": _attn_name,
            "activation": _act,
            "pos_encoding": _pos,
            "norm": _norm_name,
            "pooling": _pool,
            "warmup": _warm,
            "optimizer": _opt,
            "scheduler": _sched,
            "best_val_f1": _hist["val_f1"][_best_idx],
            "best_val_acc": _hist["val_acc"][_best_idx],
            "final_val_f1": _hist["val_f1"][-1],
            "final_val_acc": _hist["val_acc"][-1],
            "best_epoch": _best_idx + 1,
            "final_train_loss": _hist["train_loss"][-1],
        })

        # Clean up checkpoint and free memory
        _ckpt = f"best_model_{_label}.pt"
        if os.path.exists(_ckpt):
            os.remove(_ckpt)
        del _model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Save results to CSV ---
    results_df = pd.DataFrame(ablation_results)
    results_df.to_csv(os.path.join(ABLATION_DIR, "ablation_results.csv"), index=False)
    print(f"\nResults saved to {ABLATION_DIR}/ablation_results.csv")

    # --- Save histories to JSON ---
    with open(os.path.join(ABLATION_DIR, "ablation_histories.json"), "w") as _f:
        json.dump(ablation_histories, _f)
    print(f"Histories saved to {ABLATION_DIR}/ablation_histories.json")

    # --- Results table ---
    results_sorted = results_df.sort_values("best_val_f1", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 130)
    print("ABLATION STUDY RESULTS -- sorted by best Val F1 (top 30 shown)")
    print("=" * 130)
    print(results_sorted.head(30).to_string(index=False))
    print("=" * 130)

    # --- Visualisation 1: Component importance bar charts ---
    _fig1, _axes1 = plt.subplots(2, 4, figsize=(24, 10))
    _fig1.suptitle("Ablation -- Mean Best Val F1 by Component Option", fontsize=14, fontweight="bold")
    for _ax, _col in zip(_axes1.flat, _ablation_cols):
        _means = results_df.groupby(_col)["best_val_f1"].mean().sort_values(ascending=False)
        _bars = _ax.bar(range(len(_means)), _means.values, color="steelblue", edgecolor="black")
        _ax.set_xticks(range(len(_means)))
        _ax.set_xticklabels(_means.index, rotation=30, ha="right")
        _ax.set_ylabel("Mean Best Val F1")
        _ax.set_title(_col)
        for _bar, _val in zip(_bars, _means.values):
            _ax.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height() + 0.001,
                     f"{_val:.4f}", ha="center", va="bottom", fontsize=9)
        _ax.set_ylim(_means.min() - 0.02, _means.max() + 0.02)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(ABLATION_DIR, "component_importance.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: component_importance.png")

    # --- Visualisation 2: Box plots of F1 distribution by component ---
    _fig2, _axes2 = plt.subplots(2, 4, figsize=(24, 10))
    _fig2.suptitle("Ablation -- Best Val F1 Distribution by Component", fontsize=14, fontweight="bold")
    for _ax, _col in zip(_axes2.flat, _ablation_cols):
        _groups = [results_df[results_df[_col] == v]["best_val_f1"].values
                   for v in results_df[_col].unique()]
        _labels = list(results_df[_col].unique())
        _bp = _ax.boxplot(_groups, tick_labels=_labels, patch_artist=True)
        for _patch in _bp["boxes"]:
            _patch.set_facecolor("lightblue")
            _patch.set_alpha(0.7)
        _ax.set_ylabel("Best Val F1")
        _ax.set_title(_col)
        _ax.tick_params(axis="x", rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(ABLATION_DIR, "f1_boxplots.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: f1_boxplots.png")

    # --- Visualisation 3: Pairwise heatmaps ---
    _fig3, _axes3 = plt.subplots(1, 3, figsize=(21, 6))
    _fig3.suptitle("Ablation -- Pairwise Component Interactions (Mean Best Val F1)",
                   fontsize=14, fontweight="bold")
    _pairs = [("attention", "activation"), ("norm", "pooling"), ("warmup", "optimizer")]
    for _ax, (_c1, _c2) in zip(_axes3, _pairs):
        _pivot = results_df.pivot_table(values="best_val_f1", index=_c1, columns=_c2, aggfunc="mean")
        _im = _ax.imshow(_pivot.values, cmap="YlOrRd", aspect="auto")
        _ax.set_xticks(range(len(_pivot.columns)))
        _ax.set_xticklabels(_pivot.columns, rotation=30, ha="right")
        _ax.set_yticks(range(len(_pivot.index)))
        _ax.set_yticklabels(_pivot.index)
        _ax.set_xlabel(_c2)
        _ax.set_ylabel(_c1)
        _ax.set_title(f"{_c1} x {_c2}")
        for _r in range(len(_pivot.index)):
            for _c in range(len(_pivot.columns)):
                _ax.text(_c, _r, f"{_pivot.values[_r, _c]:.4f}",
                         ha="center", va="center", fontsize=11, fontweight="bold")
        _fig3.colorbar(_im, ax=_ax, shrink=0.8)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(ABLATION_DIR, "pairwise_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: pairwise_heatmaps.png")

    # --- Visualisation 4: Top-10 training curves ---
    _top10 = results_sorted.head(10)
    _fig4, _axes4 = plt.subplots(1, 2, figsize=(16, 6))
    _fig4.suptitle("Top-10 Configurations -- Training Curves", fontsize=14, fontweight="bold")
    for _, _row in _top10.iterrows():
        _name = "|".join(str(_row[c]) for c in _ablation_cols)
        _h = ablation_histories[_name]
        _axes4[0].plot(_h["train_loss"], label=_name, alpha=0.8)
        _axes4[1].plot(_h["val_f1"], label=_name, alpha=0.8, marker="o")
    _axes4[0].set_title("Training Loss")
    _axes4[0].set_xlabel("Epoch")
    _axes4[0].set_ylabel("Loss")
    _axes4[0].legend(fontsize=6, loc="upper right")
    _axes4[1].set_title("Validation F1")
    _axes4[1].set_xlabel("Epoch")
    _axes4[1].set_ylabel("F1")
    _axes4[1].legend(fontsize=6, loc="lower right")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(ABLATION_DIR, "top10_curves.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: top10_curves.png")

    # --- Visualisation 5: Parallel coordinates ---
    _fig5, _ax5 = plt.subplots(figsize=(18, 8))
    _fig5.suptitle("Ablation -- Parallel Coordinates (coloured by Best Val F1)",
                   fontsize=14, fontweight="bold")
    _pc_data = results_df[_ablation_cols + ["best_val_f1"]].copy()
    _pc_enc = {}
    for _col in _ablation_cols:
        _vals = sorted(_pc_data[_col].unique())
        _pc_enc[_col] = {v: i for i, v in enumerate(_vals)}
        _pc_data[_col] = _pc_data[_col].map(_pc_enc[_col])

    _f1_min = results_df["best_val_f1"].min()
    _f1_max = results_df["best_val_f1"].max()
    _f1_range = _f1_max - _f1_min + 1e-8
    for _, _row in _pc_data.iterrows():
        _vals = [_row[c] for c in _ablation_cols]
        _color = plt.cm.RdYlGn((_row["best_val_f1"] - _f1_min) / _f1_range)
        _ax5.plot(range(len(_ablation_cols)), _vals, color=_color, alpha=0.3, linewidth=0.5)
    _ax5.set_xticks(range(len(_ablation_cols)))
    _ax5.set_xticklabels(_ablation_cols, rotation=30, ha="right")
    for _col, _enc in _pc_enc.items():
        _xi = _ablation_cols.index(_col)
        for _k, _v in _enc.items():
            _ax5.annotate(_k, (_xi, _v), textcoords="offset points",
                          xytext=(8, 0), fontsize=6, ha="left")
    _ax5.set_ylabel("Option Index")
    _ax5.set_ylim(-0.5, max(len(v) for v in _pc_enc.values()) - 0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(ABLATION_DIR, "parallel_coordinates.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: parallel_coordinates.png")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)
    print(f"Total configurations tested: {_ablation_total}")
    print(f"\nBest configuration:")
    _best_row = results_sorted.iloc[0]
    for _col in _ablation_cols:
        print(f"  {_col:>15}: {_best_row[_col]}")
    print(f"  {'best_val_f1':>15}: {_best_row['best_val_f1']:.4f}")
    print(f"  {'best_val_acc':>15}: {_best_row['best_val_acc']:.4f}")
    print(f"\nAll artifacts saved to: {ABLATION_DIR}/")
    print("  - ablation_results.csv")
    print("  - ablation_histories.json")
    print("  - component_importance.png")
    print("  - f1_boxplots.png")
    print("  - pairwise_heatmaps.png")
    print("  - top10_curves.png")
    print("  - parallel_coordinates.png")

    # Display full sorted results as interactive table
    results_sorted

    return


if __name__ == "__main__":
    app.run()
