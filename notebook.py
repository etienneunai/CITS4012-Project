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

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import gdown
    import os

    if not(os.path.exists('MultiRC/train_456-fixedIds.json') and os.path.exists('MultiRC/test_83-fixedIds.json')):
        gdown.download_folder(id="18SlXjdkhUrG_PE0aTysMypMKx41a9evH")

    train_json = 'MultiRC/train_456-fixedIds.json'
    test_json = 'MultiRC/test_83-fixedIds.json'
    return test_json, train_json


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


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


@app.cell
def _(test_json, train_json):
    import json
    with open(train_json, 'r') as f:
        train_data = json.load(f)['data']
    with open(test_json, 'r') as f:
        test_data = json.load(f)['data']

    print(train_data[1])
    return test_data, train_data


@app.cell(hide_code=True)
def _():

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
    vocab = build_vocab(flat_train, min_freq=2)

    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    MAX_LEN = 512

    def encode(passage, question, answer):
        tokens = (
            [CLS_TOKEN]
            + tokenize(passage)[:380]
            + [SEP_TOKEN]
            + tokenize(question)[:64]
            + [SEP_TOKEN]
            + tokenize(answer)[:60]
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

    BATCH_SIZE = 32

    train_dataset = MultiRCDataset(flat_train)
    test_dataset = MultiRCDataset(flat_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

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

            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

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

    # eti we gotta check this idk it
    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
            )

        def forward(self, x):
            return self.net(x)

    # idk
    class TransformerEncoderLayer(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, attention_cls, dropout=0.1):
            super().__init__()
            self.attention = attention_cls(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff, dropout)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            x = x + self.dropout(self.attention(self.norm1(x), mask))
            x = x + self.dropout(self.ffn(self.norm2(x)))
            return x

    # somethin else
    class TransformerClassifier(nn.Module):
        def __init__(
            self,
            vocab_size,
            d_model=256,
            num_heads=8,
            num_layers=4,
            d_ff=1024,
            max_len=512,
            attention_cls=MultiHeadAttention,
            dropout=0.1,
        ):
            super().__init__()
            self.d_model = d_model

            self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
            self.pos_encoding = PositionalEncoding(d_model, max_len)

            self.layers = nn.ModuleList([
                TransformerEncoderLayer(d_model, num_heads, d_ff, attention_cls, dropout)
                for _ in range(num_layers)
            ])
            self.norm = nn.LayerNorm(d_model)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
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
            cls_repr = x[:, 0, :]

            return self.classifier(cls_repr).squeeze(-1)


    return (TransformerClassifier,)


@app.cell
def _(
    MAX_LEN,
    MultiHeadAttention,
    TransformerClassifier,
    nn,
    torch,
    vocab_size,
):
    # train the mfs
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn.metrics import f1_score, accuracy_score

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

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
        return avg_loss, acc, f1

    def create_model(attention_cls=MultiHeadAttention, d_model=256, num_heads=8,
                     num_layers=4, d_ff=1024, dropout=0.1):
        model = TransformerClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_len=MAX_LEN,
            attention_cls=attention_cls,
            dropout=dropout,
        ).to(DEVICE)
        return model

    def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4,
                     weight_decay=1e-4, patience=3, label="default"):
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0.0
            num_batches = 0

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

                total_loss += loss.item()
                num_batches += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_train_loss = total_loss / num_batches
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion)

            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_f1"].append(val_f1)

            print(f"[{label}] Epoch {epoch+1}: train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f"best_model_{label}.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(f"best_model_{label}.pt", weights_only=True))
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
    LinearAttention,
    MultiHeadAttention,
    create_model,
    test_loader,
    train_loader,
    train_model,
):

    import matplotlib.pyplot as plt

    NUM_EPOCHS = 10
    LR = 1e-4

    experiments = {}

    # --- Experiment 1: Standard Multi-Head Attention ---
    print("=" * 60)
    print("Experiment 1: Multi-Head Attention (4 layers, 8 heads)")
    print("=" * 60)
    model_mha = create_model(attention_cls=MultiHeadAttention, num_layers=4, num_heads=8)
    history_mha = train_model(model_mha, train_loader, test_loader,
                              num_epochs=NUM_EPOCHS, lr=LR, label="mha")
    experiments["Multi-Head Attention"] = history_mha

    # --- Experiment 2: Linear Attention ---
    print("\n" + "=" * 60)
    print("Experiment 2: Linear Attention (4 layers, 8 heads)")
    print("=" * 60)
    model_linear = create_model(attention_cls=LinearAttention, num_layers=4, num_heads=8)
    history_linear = train_model(model_linear, train_loader, test_loader,
                                 num_epochs=NUM_EPOCHS, lr=LR, label="linear")
    experiments["Linear Attention"] = history_linear

    print("\nAll experiments complete.")

    return experiments, plt


@app.cell
def _(experiments, plt):

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


if __name__ == "__main__":
    app.run()
