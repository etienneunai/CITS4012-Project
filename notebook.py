# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets>=5.0.1",
#     "gdown==6.4.1",
#     "gensim==4.4.0",
#     "huggingface-hub>=1.5,<2.0",
#     "marimo>=0.23.3",
#     "sentencepiece==0.2.2",
#     "tqdm==4.70.1",
# ]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def hyperparameters():
    # === Hyperparameters =========================================================
    # Everything that controls the experiment, in one place.

    # --- Paths ---
    # Directories the notebook writes to are anchored at the notebook's own directory, so their
    # contents sit beside the notebook (next to TaskOutline.md, README.md, ...) and are saved
    # with the workspace. A bare relative name would instead resolve against whatever directory
    # the kernel happened to start in -- not the notebook's folder, and gone once the notebook is
    # closed and reopened.
    import os as _os
    import marimo as _mo

    _NOTEBOOK_DIR = str(_mo.notebook_dir() or _os.getcwd())

    # --- Data & tokenization ---
    # Every train word is kept: raising this can only make token-level OOV worse, and
    # 7.8% of test tokens are already unseen in the train vocabulary.
    MIN_FREQ = 1
    # Encoder budgets, set from the MultiRC token-length distribution (train split,
    # 27,243 answer instances; tokens are lowercase `\w+` with punctuation dropped, see
    # the `tokenizer` cell). Passage: mean 283, p50 279, p90 423, p95 446, p99 504,
    # max 574 -- so 448 sits at the 95th percentile and only 3.7% of instances are
    # truncated at all. Of those few, the head+tail window still holds the evidence
    # sentence for 96.7% of correct answers (head-only 62.3%, tail-only 32.9%, both
    # 1.5%; the dropped middle loses 3.3%).
    PASSAGE_MAX_TOKENS = 448
    QUESTION_MAX_TOKENS = 32  # question mean 11.4 / p95 21 tokens: truncates 0.17%
    ANSWER_MAX_TOKENS = 28    # answer mean 4.7 / p95 15 tokens: truncates 0.83%
    QA_MAX_TOKENS = 96        # safety cap only: 32 + 1 + 28 + 1 = 62, so never binds
    BATCH_SIZE = 64
    NUM_WORKERS = 0           # DataLoader workers. Padding and collate run off the main
                              # thread, so they overlap the GPU step instead of blocking it --
                              # but only if the workers can start at all. This notebook defines
                              # the dataset and `collate_batch` in a cell, so a worker that is
                              # not forked re-imports `__main__` -- the notebook launcher, not
                              # this notebook -- and cannot resolve them while unpickling: spawn
                              # raises `AttributeError: Can't get attribute 'MultiRCDataset' on
                              # <module '__mp_main__'>` and the parent dies with
                              # `BrokenPipeError` before the first batch. `_loader_workers` in
                              # the `dataset` cell carries the detail and forces 0 for any value
                              # that cannot work, so leave this at 0 while the dataset and the
                              # collate function are defined in cells. Numerically inert either
                              # way: the dataset is pre-tokenised, collate is a pure function,
                              # and the batch order comes from the sampler in the main process,
                              # so runs are bit-for-bit unchanged.
    VAL_FRACTION = 0.18
    SEED = 42

    # "head" keeps the first PASSAGE_MAX_TOKENS tokens; "head_tail" keeps the head plus
    # the last PASSAGE_TAIL_FRACTION of the budget, so the end of a long paragraph survives.
    PASSAGE_TRUNCATION = "head_tail"   # head | head_tail
    PASSAGE_TAIL_FRACTION = 0.5

    # --- Model ---
    ENCODER = "two_tower"     # two_tower | single (single = ablation baseline)
    ATTENTION = "cross"       # self | cross | inter | none  (cross/inter need two towers)
    CROSS_DIRECTION = "both"  # both | qa_to_passage
    PASSAGE_ENCODER = "bigru" # bigru | none
    POOLING = "attention"     # final | mean | max | attention
    QA_POOLING = "answer"     # answer | full
    # What the classifier sees (see `_head_input` and `_passage_match_features`):
    #   "interaction" -> [u; v; |u - v|; u * v]   (the pooled vectors, plus the comparison)
    #   "compare"     -> [|u - v|; u * v]         (every term contains the passage)
    #   "concat"      -> [u; v]
    #   "match"       -> only the passage-conditioned alignment features, so the pooled
    #                    vectors -- and with them the whole POOLING/QA_POOLING axis -- never
    #                    reach the classifier.
    # "interaction" is the configured model and the head the section-4 architecture grid is
    # swept under: an ablation over POOLING needs the pooled vectors to reach the classifier,
    # and under "match" they do not (the grid cell refuses that combination by name). "match"
    # stays in HEAD_OPTIONS as the fully passage-conditioned extreme. MATCH_FEATURES appends
    # the alignment features to every head except "match", whose input *is* them.
    HEAD = "interaction"      # concat | interaction | compare | match
    MATCH_FEATURES = True
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_LAYERS = 1            # BiGRU layers; dropout applies between them when > 1
    DROPOUT = 0.2

    # --- Optimisation ---
    # The budget *is* the LR schedule's horizon: `lr_for_epoch` anneals the cosine over
    # `num_epochs`, so the budget has to sit near where early stopping actually fires.
    # At 25 with patience 5 the runs stopped around epoch 7 with the LR still at ~93% of
    # its peak, so the cosine never annealed and the selected checkpoint was a
    # mid-schedule one. 12 lets the anneal land inside the run.
    NUM_EPOCHS = 12
    LR = 3e-4                 # short warmup, then cosine decay
    LR_WARMUP_EPOCHS = 2
    LR_DECAY = "cosine"       # cosine | constant
    LR_MIN = 1e-5             # floor of the cosine decay
    WEIGHT_DECAY = 1e-2       # L2 on weight matrices only (no biases/LayerNorm/embeddings)
    GRAD_CLIP = 1.0
    USE_TF32 = True           # Ampere+ matmul/RNN kernels; no effect on CPU

    # --- Checkpoint selection ---
    # Select on *within-question* ranking, the comparison MultiRC's metrics use.
    # Within-question AUC is the right *kind* of selection signal for MultiRC: every
    # question has a different number of correct answers, so a fixed global cut is
    # inappropriate, and the threshold-free within-question ranking is what the
    # per-question metrics rest on. It is also the only *stable* one -- per-question F1
    # on ~860 validation questions jitters ~0.03-0.08 epoch to epoch, against ~0.004 for
    # the AUC. `val_q_f1_declared` (the reported operating point, now recorded every
    # epoch) is available here if you ever want to select on the exact number the report
    # leads with instead.
    EARLY_STOP_METRIC = "val_q_auc"   # val_q_auc | val_q_ap | val_ap | val_q_f1 | val_loss | val_auc | val_f1 | val_q_f1_declared
    EARLY_STOPPING = True
    EARLY_STOP_PATIENCE = 5
    # Must sit *below* the selection metric's epoch-to-epoch movement (~0.004 on
    # val_q_auc). At 0.010 the "improved" test swallowed the genuine gains and 15 of 19
    # runs reported an epoch that was not the argmax -- by up to 0.008 of AUC, which is
    # most of the difference the ablation is trying to measure.
    EARLY_STOP_MIN_DELTA = 0.001
    EARLY_STOP_MIN_EPOCHS = 4      # never stop inside the warmup / early annealing

    # --- Decision threshold ---
    TUNE_THRESHOLD = True
    # "within_question" centres each question's candidate scores on their own mean before
    # cutting: the score level is not comparable across questions. Declared, not fitted.
    THRESHOLD_RULE = "within_question"   # within_question | prior_match | fixed
    THRESHOLD_FIXED = 0.50
    REPORT_SWEEP_DIAGNOSTIC = True       # sweep is a printed diagnostic only, never a reported number
    THRESHOLD_METRIC = "q_f1"      # sweep objective -- diagnostic only (q_f1 | f1)
    THRESHOLD_MIN = 0.10
    THRESHOLD_MAX = 0.90
    THRESHOLD_STEP = 0.01

    # --- Loss ---
    USE_CLASS_WEIGHTS = False  # skew handled by threshold calibration, not weighting
    LOSS = "within_question"   # bce | within_question
    PAIRWISE_WEIGHT = 0.5      # weight of the pairwise ranking term (within_question only)
    GROUPED_BATCHES = True     # keep a question's candidates in one training batch

    # --- Reporting ---
    N_SAMPLE_QUESTIONS = 5
    SAMPLE_SEED = 42
    N_ATTENTION_EXAMPLES = 2   # attention figures: one question right, one wrong
    N_ATTENTION_SCAN = 120     # test questions scanned for the attention evidence

    # --- Section 4: the two ablation studies ---------------------------------------------
    # There are exactly two, and they differ only in budget and seed count:
    #
    #   QUICK_ABLATION      -- a one-change-at-a-time SCREEN on a shortened budget and a
    #                          single seed, for ballpark triage. Its runs are tagged
    #                          kind="screen", so the plots show them but no reported table,
    #                          no diagnostic and no model-vs-baseline row does.
    #   RUN_ABLATION_STUDY  -- the FINAL, reported study: the full budget over
    #                          ABLATION_SEEDS, covering the architecture factorial
    #                          (ENCODER_OPTIONS x ATTENTION_OPTIONS x POOLING_OPTIONS) *and*
    #                          every COMPONENT_VARIANTS entry (head, objective, passage
    #                          truncation, passage encoder, cross-attention direction,
    #                          question/answer pooling, dropout, pairwise weight and the
    #                          alignment features). The component list is in the
    #                          `ablation_components` cell; both studies read it, so the
    #                          screen and the final study can never drift apart.
    RUN_ABLATION_STUDY = False
    ENCODER_OPTIONS = ["two_tower", "single"]
    ATTENTION_OPTIONS = ["self", "cross", "inter", "none"]
    POOLING_OPTIONS = ["final", "mean", "max", "attention"]

    # The quick screen, and its shortened budget. A screen, not a result.
    QUICK_ABLATION = True
    QUICK_ABLATION_EPOCHS = 8

    # --- Section 4: the final study's budget and seeds ---
    ABLATION_EPOCHS = 12           # ceiling *and* the cosine's horizon: early stopping
                                   # ends most runs at ~7, with the LR then down to ~20%
                                   # of its peak instead of ~93%
    ABLATION_SEEDS = [42, 43, 44]  # includes SEED, so the study subsumes the configured run
    ABLATION_CACHE_DIR = _os.path.join(_NOTEBOOK_DIR, "ablation_cache")

    # The head and the objective axes. Every value except the configured one becomes a
    # component run in the final study (and a screen row in the quick one).
    HEAD_OPTIONS = ["interaction", "concat", "compare", "match"]
    LOSS_OPTIONS = ["within_question", "bce"]

    # --- Baselines (benchmarks only -- never read by the main model) ---
    RUN_PRETRAINED_BASELINES = True
    BASELINE_MODELS = ["glove", "fasttext", "flan_t5"]   # which of the rows below to build
    # Word-vector sources, keyed by the name each benchmark is reported under -> the
    # keyed-vectors dataset to load. Each entry yields TWO rows: a best-sentence-similarity
    # row under its own name, and a frozen-embedding BiGRU under `<name>_bigru`. Both
    # sources are 300-d, so those two frozen rows are the same network size as each other
    # (both wider than the 128-d main model, which the frozen rows are not matched to).
    GLOVE_DIM = 300           # glove-wiki-gigaword-<dim>: 50 | 100 | 200 | 300
    BASELINE_EMBEDDINGS = {
        "glove": f"glove-wiki-gigaword-{GLOVE_DIM}",
        # gensim ships fastText's wiki-news vectors as a *static* table (the .vec form),
        # so a word outside its ~1M-word vocabulary gets no subword fallback. What it buys
        # over GloVe-100 is much wider coverage of *this* vocabulary, which the coverage
        # table the baselines cell prints reports directly. True subword vectors for unseen
        # words would need the `fasttext` package and cc.en.300.bin.
        "fasttext": "fasttext-wiki-news-subwords-300",
    }
    LLM_NAME = "google/flan-t5-base"
    BASELINE_BATCH_SIZE = 32
    PROMPT_MAX_TOKENS = 512    # flan_t5 prompt window; the passage absorbs the truncation

    # Frozen-embedding BiGRU baselines: the SAME architecture as the main model, but the
    # embedding table is filled from the pretrained vectors and frozen. Everything above the
    # embedding layer is learned on MultiRC, the vectors are not -- so this asks whether the
    # vectors themselves carry signal, through a readout that has seen the task. That is the
    # fair use of GloVe that the cosine rows above are not.
    RUN_EMBEDDING_BIGRU_BASELINES = True

    # --- Persistence --------------------------------------------------------------
    # The study is deterministic: SEED drives the split, the weight initialisation and the
    # batch order, so an export is the same result the notebook would compute. It exists so
    # section 3 and the baselines cell can be read back instead of retrained.
    #   SAVE_RESULTS = True -> the `save_results` cell (the last cell) writes RESULTS_DIR/
    #   LOAD_RESULTS = True -> section 3 and the baselines cell read RESULTS_DIR/ instead
    # RESULTS_DIR sits beside the notebook (see the Paths block at the top of this cell), so the
    # export is part of the saved workspace and survives closing and reopening the notebook.
    RESULTS_DIR = _os.path.join(_NOTEBOOK_DIR, "results_export")
    SAVE_RESULTS = True
    LOAD_RESULTS = False
    return (
        ABLATION_CACHE_DIR,
        ABLATION_EPOCHS,
        ABLATION_SEEDS,
        ANSWER_MAX_TOKENS,
        ATTENTION,
        ATTENTION_OPTIONS,
        BASELINE_BATCH_SIZE,
        BASELINE_EMBEDDINGS,
        BASELINE_MODELS,
        BATCH_SIZE,
        CROSS_DIRECTION,
        DROPOUT,
        D_MODEL,
        EARLY_STOPPING,
        EARLY_STOP_METRIC,
        EARLY_STOP_MIN_DELTA,
        EARLY_STOP_MIN_EPOCHS,
        EARLY_STOP_PATIENCE,
        ENCODER,
        ENCODER_OPTIONS,
        GRAD_CLIP,
        GROUPED_BATCHES,
        HEAD,
        HEAD_OPTIONS,
        LLM_NAME,
        LOAD_RESULTS,
        LOSS,
        LOSS_OPTIONS,
        LR,
        LR_DECAY,
        LR_MIN,
        LR_WARMUP_EPOCHS,
        MATCH_FEATURES,
        MIN_FREQ,
        NUM_EPOCHS,
        NUM_HEADS,
        NUM_LAYERS,
        NUM_WORKERS,
        N_ATTENTION_SCAN,
        N_SAMPLE_QUESTIONS,
        PAIRWISE_WEIGHT,
        PASSAGE_ENCODER,
        PASSAGE_MAX_TOKENS,
        PASSAGE_TAIL_FRACTION,
        PASSAGE_TRUNCATION,
        POOLING,
        POOLING_OPTIONS,
        PROMPT_MAX_TOKENS,
        QA_MAX_TOKENS,
        QA_POOLING,
        QUESTION_MAX_TOKENS,
        QUICK_ABLATION,
        QUICK_ABLATION_EPOCHS,
        REPORT_SWEEP_DIAGNOSTIC,
        RESULTS_DIR,
        RUN_ABLATION_STUDY,
        RUN_EMBEDDING_BIGRU_BASELINES,
        RUN_PRETRAINED_BASELINES,
        SAMPLE_SEED,
        SAVE_RESULTS,
        SEED,
        THRESHOLD_FIXED,
        THRESHOLD_MAX,
        THRESHOLD_METRIC,
        THRESHOLD_MIN,
        THRESHOLD_RULE,
        THRESHOLD_STEP,
        TUNE_THRESHOLD,
        USE_CLASS_WEIGHTS,
        USE_TF32,
        VAL_FRACTION,
        WEIGHT_DECAY,
    )


@app.cell
def _():
    import gdown
    import os
    import marimo as mo

    if not(os.path.exists('MultiRC/train_456-fixedIds.json') and os.path.exists('MultiRC/test_83-fixedIds.json')):
        gdown.download_folder(id="18SlXjdkhUrG_PE0aTysMypMKx41a9evH")

    train_json = 'MultiRC/train_456-fixedIds.json'
    test_json = 'MultiRC/test_83-fixedIds.json'
    return mo, test_json, train_json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Readme
    *If there is something to be noted for the marker, please mention here.*

    *If you are planning to implement a program with Object Oriented Programming style, please put those the bottom of this ipynb file*

    ## Current configuration

    What this notebook runs, in one place. Every value is set in the hyperparameters cell, and
    the notebook prints the settings beside the numbers they produce.

    **Task.** MultiRC. A passage is paired with a question, and every candidate answer is
    labelled supported or not. Results are reported with MultiRC's per-question metrics beside
    the answer-level ones, and with the threshold-free within-question ranking metrics the
    checkpoints are selected on.

    **Model (trained from scratch).** Two BiGRU towers -- one over the passage (sentences
    joined by `<sep>`, so the multi-sentence structure the task is built around survives), one
    over `question <sep> answer <sep>` -- related by cross-attention in both directions
    (ablated against self-attention, ESIM-style inter-attention, and no attention at all).
    Each tower is summarised by attention pooling and the classifier reads an interaction
    head, `[u; v; |u - v|; u * v]`, with passage-conditioned alignment features appended.
    Embeddings are random-initialised: no pretrained vector is trained or read by the model.

    **Training.** A fixed budget with a linear warmup and a cosine learning rate (no plateau
    scheduler), early stopping and checkpoint selection both on validation within-question
    AUC, and a decision cut that is declared on validation -- after centring each question's
    candidates on their own mean -- rather than fitted to a metric.

    **Baselines (benchmarks only).** Always-yes / always-no, lexical overlap, best-sentence
    similarity from two frozen 300-d word-vector sources (GloVe and fastText), a FLAN-T5
    zero-shot yes/no prompt, and two frozen-embedding BiGRUs (the main architecture with
    each source's vectors held fixed). None of these is read by the model; the frozen
    BiGRUs are the only baselines that train, and they train *from* the frozen vectors.

    **Section 4.** A `quick_ablation` screen (one change at a time, short budget), then the
    final grid -- every `encoder x attention x pooling` architecture under the configured
    head, plus a `head x objective` factorial -- each repeated over seeds, with a summary that
    aggregates the seeds and prints the baselines underneath.
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

    # Sentences of a passage are joined by this separator in preprocessing;
    # `encode_passage` turns it into `<sep>` tokens.
    SENT_SEP = "\n"

    _SENT_MARK_RE = re.compile(r"<b>\s*Sent\s*\d+\s*:\s*</b>", re.IGNORECASE)


    def clean_sentence(text):
        """Drop remaining markup from one sentence and trim whitespace."""
        return re.sub(r"<[^>]+>", " ", text).strip()


    def split_sentences(text):
        """Split a MultiRC paragraph on its `<b>Sent N:</b>` markers.

        The markers become explicit sentence boundaries rather than being deleted,
        which preserves the sentence structure the task is built around.
        """
        sentences = []
        for part in _SENT_MARK_RE.split(text):
            cleaned = clean_sentence(part)
            if cleaned:
                sentences.append(cleaned)
        return sentences


    def clean_passage(text):
        """Passage text with its sentences separated by `SENT_SEP`."""
        return SENT_SEP.join(split_sentences(text))


    def flatten_data(data):
        """(passage, question, answer, label) tuples for every candidate answer."""
        flat = []
        for item in data:
            passage = clean_passage(item['paragraph']['text'])
            for q in item['paragraph']['questions']:
                for a in q['answers']:
                    flat.append((passage, q['question'], a['text'], a['isAnswer']))
        return flat


    return SENT_SEP, clean_passage, flatten_data, re


@app.cell
def tokenizer(re):
    # --- Tokenization ------------------------------------------------------------
    # Word-level, punctuation dropped. Punctuation was 13.6% of every token in this corpus
    # (`.` and `,` alone were 9.5%), and emitting each mark as its own token spent that much of
    # the PASSAGE_MAX_TOKENS budget on nothing: dropping it shortens an encoded passage by ~13%
    # while leaving the signal the model actually uses unchanged -- the passage-overlap feature
    # `_passage_match_features` computes moves by +0.0015 within-question AUC over 942 test
    # questions, 95% CI [-0.0014, +0.0045], i.e. inside noise.
    #
    # Stop words are deliberately *kept*. Removing them would halve the sequence length, but it
    # measurably degrades that same overlap feature (-0.0155, 95% CI [-0.0291, -0.0014]), and
    # the ablation shows the overlap features are the model's dominant mechanism
    # (`no_match_features` is the largest single effect in the screen, -0.073 qAUC). Trading
    # the signal that works for speed is the wrong trade.
    TOKEN_RE = re.compile(r"\w+")


    def tokenize(text):
        """Lowercase word tokens; punctuation is dropped, not emitted as its own token."""
        return TOKEN_RE.findall(text.lower())


    PAD_TOKEN = "<pad>"
    SEP_TOKEN = "<sep>"
    UNK_TOKEN = "<unk>"
    return PAD_TOKEN, SEP_TOKEN, UNK_TOKEN, tokenize


@app.cell
def dataset(
    ANSWER_MAX_TOKENS,
    BATCH_SIZE,
    GROUPED_BATCHES,
    MIN_FREQ,
    NUM_WORKERS,
    PAD_TOKEN,
    PASSAGE_MAX_TOKENS,
    PASSAGE_TAIL_FRACTION,
    PASSAGE_TRUNCATION,
    QA_MAX_TOKENS,
    QUESTION_MAX_TOKENS,
    SEED,
    SENT_SEP,
    SEP_TOKEN,
    UNK_TOKEN,
    VAL_FRACTION,
    flatten_data,
    test_data,
    tokenize,
    train_data,
):
    import multiprocessing

    import torch
    from torch.utils.data import Dataset, DataLoader
    from collections import Counter


    def build_vocab(flat_data, min_freq=MIN_FREQ):
        counter = Counter()
        for passage, question, answer, _ in flat_data:
            for sentence in passage.split(SENT_SEP):
                counter.update(tokenize(sentence))
            counter.update(tokenize(question))
            counter.update(tokenize(answer))
        vocab = {PAD_TOKEN: 0, SEP_TOKEN: 1, UNK_TOKEN: 2}
        for word, freq in counter.most_common():
            if freq >= min_freq:
                vocab[word] = len(vocab)
        return vocab


    flat_train = flatten_data(train_data)
    flat_test = flatten_data(test_data)
    vocab = build_vocab(flat_train)
    vocab_size = len(vocab)
    PAD_ID = vocab[PAD_TOKEN]
    SEP_ID = vocab[SEP_TOKEN]
    UNK_ID = vocab[UNK_TOKEN]
    print(f"Vocabulary size: {vocab_size}")


    def encode_passage(passage, truncation=PASSAGE_TRUNCATION):
        """Passage-tower input: sentence tokens joined by `<sep>`, then truncated.

        The `<sep>` markers keep the multi-sentence structure. An over-long passage is
        cut by `truncation`: "head" keeps the first PASSAGE_MAX_TOKENS tokens; "head_tail"
        keeps the head plus the last PASSAGE_TAIL_FRACTION of the budget. Ids are not
        padded here -- `pad_sequences` pads each batch to its own longest sequence.
        """
        ids = []
        for sentence in passage.split(SENT_SEP):
            ids.extend(vocab.get(t, UNK_ID) for t in tokenize(sentence))
            ids.append(SEP_ID)
        if len(ids) <= PASSAGE_MAX_TOKENS:
            return ids
        if truncation == "head":
            return ids[:PASSAGE_MAX_TOKENS]
        if truncation == "head_tail":
            n_tail = min(int(PASSAGE_MAX_TOKENS * PASSAGE_TAIL_FRACTION),
                         PASSAGE_MAX_TOKENS)
            n_head = PASSAGE_MAX_TOKENS - n_tail
            return ids[:n_head] + (ids[-n_tail:] if n_tail else [])
        raise ValueError(f"unknown truncation: {truncation!r} (head | head_tail)")


    def encode_qa(question, answer):
        """Question/answer-tower input: `question <sep> answer <sep>`, capped.

        Keeping the two spans in their own tower gives them a short, direct gradient
        path, and the `<sep>` markers delimit the answer span the model pools over.
        """
        tokens = (
            tokenize(question)[:QUESTION_MAX_TOKENS]
            + [SEP_TOKEN]
            + tokenize(answer)[:ANSWER_MAX_TOKENS]
            + [SEP_TOKEN]
        )
        return [vocab.get(t, UNK_ID) for t in tokens][:QA_MAX_TOKENS]


    def pad_sequences(token_lists, pad_id=PAD_ID):
        """Right-pad a list of id lists to the longest sequence in the batch."""
        max_len = max(len(ids) for ids in token_lists)
        padded = [list(ids) + [pad_id] * (max_len - len(ids)) for ids in token_lists]
        return torch.tensor(padded, dtype=torch.long)


    def collate_batch(batch):
        """Pad each batch to its own longest passage / question+answer."""
        passage_ids, qa_ids, targets, qids = zip(*batch)
        return (pad_sequences(passage_ids), pad_sequences(qa_ids),
                torch.stack(targets), torch.tensor(qids, dtype=torch.long))


    class MultiRCDataset(Dataset):
        """MultiRC tuples, tokenised once at construction.

        Each item is `(passage_ids, qa_ids, target, question_id)`. Ids stay
        variable-length so `collate_batch` can pad per batch, and `question_id` groups the
        candidates of one question for the per-question metrics. `truncation` is an
        argument because an ablation over it has to re-encode.
        """

        def __init__(self, flat_data, truncation=PASSAGE_TRUNCATION):
            self.truncation = truncation
            self.items = []
            self.labels = []
            self.qids = []       # per item, so a batch sampler can group by question
            question_ids = {}
            for passage, question, answer, label in flat_data:
                key = (passage, question)
                if key not in question_ids:
                    question_ids[key] = len(question_ids)
                self.items.append((encode_passage(passage, truncation),
                                   encode_qa(question, answer),
                                   torch.tensor(1.0 if label else 0.0, dtype=torch.float),
                                   question_ids[key]))
                self.labels.append(bool(label))
                self.qids.append(question_ids[key])

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return self.items[idx]


    class GroupedBatchSampler:
        """Batches that keep every candidate answer of a question together.

        The loss and the evaluation metric are per question, so a question's candidates
        have to arrive in the same batch. Questions are shuffled, then packed whole.
        """

        def __init__(self, qids, batch_size, seed=SEED):
            self.batch_size = batch_size
            self.seed = seed
            self.epoch = 0
            self.groups = {}
            for _index, _qid in enumerate(qids):
                self.groups.setdefault(_qid, []).append(_index)

        def __iter__(self):
            _keys = sorted(self.groups)
            _order = torch.randperm(
                len(_keys), generator=torch.Generator().manual_seed(self.seed + self.epoch)
            ).tolist()
            self.epoch += 1
            _batch = []
            for _k in _order:
                _batch.extend(self.groups[_keys[_k]])
                if len(_batch) >= self.batch_size:
                    yield _batch
                    _batch = []
            if _batch:
                yield _batch

        def __len__(self):
            _total = sum(len(_group) for _group in self.groups.values())
            return max(1, int(round(_total / max(1, self.batch_size))))


    # Split by passage: the questions of one passage share its text, so a tuple-level
    # split would leak near-duplicates into validation.
    _passages = {}
    for _i, _example in enumerate(flat_train):
        _passages.setdefault(_example[0], []).append(_i)

    _passage_keys = sorted(_passages)
    _order = torch.randperm(len(_passage_keys),
                            generator=torch.Generator().manual_seed(SEED)).tolist()
    _passage_keys = [_passage_keys[_i] for _i in _order]
    _val_keys = set(_passage_keys[: int(VAL_FRACTION * len(_passage_keys))])

    _train_examples = [flat_train[_i] for _k in _passage_keys if _k not in _val_keys for _i in _passages[_k]]
    _val_examples = [flat_train[_i] for _k in _passage_keys if _k in _val_keys for _i in _passages[_k]]


    _WORKER_NOTICE_SHOWN = [False]


    def _loader_workers(num_workers=NUM_WORKERS):
        """Worker count for the DataLoaders, forced to 0 when the workers could not start.

        `MultiRCDataset` and `collate_batch` are defined in this notebook, so they live in the
        running `__main__`. A worker that is not *forked* -- one started by `spawn` (this
        kernel's start method) or `forkserver` -- does not inherit this notebook's globals: it
        re-imports `__main__`, which is the notebook *launcher*, not this notebook, and then
        fails to resolve the names it was handed while unpickling. The parent's iterator dies
        with `AttributeError: Can't get attribute 'MultiRCDataset' on <module '__mp_main__'>`
        / `BrokenPipeError` before the first batch is ever produced. Such workers need the
        dataset and the collate function to come from a real importable module, and here they
        do not, so the loaders run in the main process instead. That costs little: the dataset
        pre-tokenises in `MultiRCDataset.__init__` and `collate_batch` only pads a batch, so
        Python-side loading is far from the bottleneck. `fork` would keep the workers alive,
        but this kernel has CUDA initialised and runs cells off the main thread, which is
        precisely what fork is unsafe with.

        Workers would not be worth having here even if they started. Measured on this machine,
        the configured model (d_model 128, 1 BiGRU layer, 2.65M params, `interaction` head) at
        the configured shapes: fetching and collating a 64-item batch in-process costs 2.0 ms,
        and one forward+backward+optimiser step costs 39 ms on the GPU -- so the host work is
        4.8% of a batch, and a full 358-batch epoch would go from 14.6 s to 13.9 s with
        *perfect* overlap. Against that, four `spawn` workers take ~12 s to serve their first
        batch in this environment (each one re-imports torch and the notebook launcher), paid
        once per loader -- and `build_loaders` builds three of them. Trading a ~5% ceiling for
        minutes of process startup is the wrong way round, which is why NUM_WORKERS is 0.

        Any `num_workers > 0` that cannot work is downgraded to 0, so raising NUM_WORKERS in
        the hyperparameters cell can never break a run this way again -- it prints one line the
        first time it has to, so the downgrade is not silent.
        """
        if not num_workers:
            return 0
        _defined_in_a_cell = (MultiRCDataset.__module__ == "__main__"
                              or collate_batch.__module__ == "__main__")
        if _defined_in_a_cell and multiprocessing.get_start_method() != "fork":
            if not _WORKER_NOTICE_SHOWN[0]:
                _WORKER_NOTICE_SHOWN[0] = True
                print(f"NOTE: NUM_WORKERS={num_workers} cannot be honoured here -- the dataset and "
                      f"`collate_batch` are defined in this notebook, and a "
                      f"{multiprocessing.get_start_method()!r} worker copies its arguments by "
                      f"reference instead of inheriting them, so it cannot import them. Loading "
                      f"in-process instead; see `_loader_workers` for the measurements.")
            return 0
        return num_workers


    def build_loaders(truncation=PASSAGE_TRUNCATION):
        """(Re)build the train/val/test datasets and loaders for a truncation mode.

        The vocab and the passage-level split are shared, but the encoded inputs are
        not, so an ablation over `PASSAGE_TRUNCATION` has to re-encode through here.
        """
        train_ds = MultiRCDataset(_train_examples, truncation)
        val_ds = MultiRCDataset(_val_examples, truncation)
        test_ds = MultiRCDataset(flat_test, truncation)
        # 0 workers means no pickling at all: the dataset, the collate function and the
        # batch sampler are used in this process, which is what `_loader_workers`
        # falls back to when the configured workers could not start.
        _workers = _loader_workers()
        _loader_kwargs = {"num_workers": _workers,
                          "pin_memory": torch.cuda.is_available(),
                          "persistent_workers": _workers > 0}
        # The training loader is the only one that needs to group by question.
        if GROUPED_BATCHES:
            _train_loader = DataLoader(
                train_ds, batch_sampler=GroupedBatchSampler(train_ds.qids, BATCH_SIZE),
                collate_fn=collate_batch, **_loader_kwargs)
        else:
            _train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                       collate_fn=collate_batch, **_loader_kwargs)
        return (train_ds, val_ds, test_ds, _train_loader,
                DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_batch, **_loader_kwargs),
                DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                           collate_fn=collate_batch, **_loader_kwargs))


    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = build_loaders()

    print(f"Tuples: {len(train_dataset)} train / {len(val_dataset)} val / {len(test_dataset)} test | DataLoader workers: {train_loader.num_workers}")

    # --- Class imbalance ---------------------------------------------------------
    # The skew is handled by *declaring* the cut on validation, not by also weighting the
    # loss (that would correct for it twice). The ratio is still reported, and feeds
    # `BCEWithLogitsLoss` in `train_model` if `USE_CLASS_WEIGHTS` is ever turned on.
    _train_pos = sum(train_dataset.labels)
    _train_neg = len(train_dataset.labels) - _train_pos
    POS_WEIGHT = torch.tensor(_train_neg / max(_train_pos, 1), dtype=torch.float)
    print(f"Train positives: {_train_pos:,}/{len(train_dataset.labels):,} "
          f"({_train_pos / len(train_dataset.labels):.1%}) -> "
          f"POS_WEIGHT = {POS_WEIGHT.item():.4f}")
    return (
        Counter,
        PAD_ID,
        POS_WEIGHT,
        SEP_ID,
        UNK_ID,
        build_loaders,
        encode_passage,
        encode_qa,
        flat_test,
        flat_train,
        pad_sequences,
        test_loader,
        torch,
        train_loader,
        val_dataset,
        val_loader,
        vocab,
        vocab_size,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Tokenization & Encoding

    A word-level tokenizer: a regex takes lowercase word tokens and drops punctuation
    (stop words are kept -- see the `tokenizer` cell), and the vocabulary is built from
    the training tuples (`MIN_FREQ` drops rare words). Each
    example feeds up to **two** encoder inputs:

    ```
    sentence <sep> sentence <sep> ...   -> passage tower
    question <sep> answer <sep>         -> question/answer tower
    ```

    MultiRC pairs each passage with several questions, and each question with several
    candidate answers. The passage is segmented into sentences, so the `<sep>` between
    sentences preserves the multi-sentence structure the task is built around: the
    evidence for an answer usually sits in one or two specific sentences. Splitting the
    passage from the question+answer also keeps the question/answer on a short, direct
    gradient path instead of burying it at the end of one long
    `passage <sep> question <sep> answer` sequence, and gives attention two distinct
    representations to relate. There is deliberately no `[CLS]` token -- a BiGRU does
    not need one, and the classifier aggregates each tower itself (see the pooling
    options below).

    The two `<sep>` markers in the question/answer tower also delimit the answer span,
    which the head pools over (`QA_POOLING`). Each tower is capped separately
    (`PASSAGE_MAX_TOKENS`, `QA_MAX_TOKENS`) and batches are padded only to the longest
    sequence *in that batch*; padded timesteps are packed out of the BiGRUs and masked
    out of attention and pooling. The `single` encoder instead concatenates the two into
    `passage <sep> question <sep> answer` for one BiGRU -- the ablation baseline for the
    two-tower split.
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
    import math
    import torch.nn as nn


    class SelfAttention(nn.Module):
        """Multi-head self-attention: every token attends to every other token.

        `capture_attention` is off during training and only turned on by the
        qualitative-analysis cell, which keeps the last forward pass's weights for plotting.
        """

        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.qkv = nn.Linear(d_model, 3 * d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.capture_attention = False   # set True by the attention-figure cell
            self.last_weights = None         # weights of the last forward pass, when kept

        def forward(self, x, mask=None):
            batch, seq_len, _ = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q = q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)

            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
            if mask is not None:  # mask is True on padding
                scores = scores.masked_fill(mask[:, None, None, :], -1e9)
            weights = self.dropout(scores.softmax(dim=-1))
            if self.capture_attention:
                self.last_weights = weights.detach()

            context = (weights @ v).transpose(1, 2).reshape(batch, seq_len, -1)
            return self.out(context)


    class CrossAttention(nn.Module):
        """Multi-head attention with separate queries and keys/values.

        Relates two spans: queries from one, keys/values from another, with `key_mask`
        marking positions a query may not attend to.
        """

        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
            self.num_heads = num_heads
            self.d_k = d_model // num_heads
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.capture_attention = False   # set True by the attention-figure cell
            self.last_weights = None         # weights of the last forward pass, when kept

        def forward(self, query, key_value, key_mask=None):
            batch, q_len, _ = query.shape
            kv_len = key_value.size(1)

            q = self.W_q(query).view(batch, q_len, self.num_heads, self.d_k).transpose(1, 2)
            k = self.W_k(key_value).view(batch, kv_len, self.num_heads, self.d_k).transpose(1, 2)
            v = self.W_v(key_value).view(batch, kv_len, self.num_heads, self.d_k).transpose(1, 2)

            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
            if key_mask is not None:
                scores = scores.masked_fill(key_mask[:, None, None, :], -1e9)
            weights = self.dropout(scores.softmax(dim=-1))
            if self.capture_attention:
                self.last_weights = weights.detach()

            context = (weights @ v).transpose(1, 2).reshape(batch, q_len, -1)
            return self.out(context)

    class InterAttention(nn.Module):
        """ESIM-style token-level soft alignment between the two towers.

        Builds the full token-pair similarity matrix between every question/answer token
        and every passage token, then feeds the match back into both towers (with a
        residual) before pooling, so a single rare token match stays visible to the
        classifier instead of being diluted into a pooled vector.
        """

        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.W = nn.Linear(d_model, d_model, bias=False)   # the bilinear match
            self.passage_out = nn.Linear(4 * d_model, d_model)
            self.qa_out = nn.Linear(4 * d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.capture_attention = False   # set True by the attention-figure cell
            self.last_weights = None         # answer->passage weights of the last forward pass

        def forward(self, qa, passage, qa_mask, passage_mask):
            similarity = torch.bmm(self.W(qa), passage.transpose(1, 2)) / math.sqrt(qa.size(-1))
            # mask is True on padding; a fully masked row or column is uniform, never NaN.
            similarity = similarity.masked_fill(qa_mask[:, :, None], -1e9)
            similarity = similarity.masked_fill(passage_mask[:, None, :], -1e9)
            qa_to_passage = similarity.softmax(dim=-1)                   # per qa token: over passage
            if self.capture_attention:
                self.last_weights = qa_to_passage.detach()
            passage_to_qa = similarity.transpose(1, 2).softmax(dim=-1)   # per passage token: over qa
            passage_align = torch.bmm(qa_to_passage, passage)            # what each qa token matches
            qa_align = torch.bmm(passage_to_qa, qa)                      # what each passage token matches
            passage = passage + self.dropout(self.passage_out(torch.cat(
                [passage, qa_align, (passage - qa_align).abs(), passage * qa_align], dim=-1)))
            qa = qa + self.dropout(self.qa_out(torch.cat(
                [qa, passage_align, (qa - passage_align).abs(), qa * passage_align], dim=-1)))
            return qa, passage


    return CrossAttention, InterAttention, SelfAttention, math, nn


@app.cell
def _(
    CrossAttention,
    InterAttention,
    PAD_ID,
    SEP_ID,
    SelfAttention,
    UNK_ID,
    nn,
    torch,
):
    def run_bigru(gru, x, mask):
        """Run a BiGRU over a padded batch, packing the padding out.

        Returns the per-timestep outputs and the final hidden state `h_n`, whose last
        layer holds (forward, backward) states on rows -2 and -1.
        """
        lengths = (~mask).sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True,
                                                   enforce_sorted=False)
        packed_out, h_n = gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True,
                                                  total_length=x.size(1))
        return out, h_n


    def masked_mean(x, mask):
        """Mean of the selected (non-masked) timesteps."""
        keep = (~mask).unsqueeze(-1).float()
        return (x * keep).sum(1) / keep.sum(1).clamp(min=1)


    def masked_max(x, mask):
        """Elementwise max over the selected (non-masked) timesteps."""
        return x.masked_fill(mask.unsqueeze(-1), -1e9).max(dim=1).values


    class AttentionPooling(nn.Module):
        """Attention-weighted sum of a tower's encoder states.

        An additive (Bahdanau-style) score is computed per timestep; the softmax weights
        are the model's own saliency, so the qualitative cell turns `capture_attention`
        on to plot where the pooled vector came from.
        """

        def __init__(self, d_model, dropout=0.1):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1),
            )
            self.dropout = nn.Dropout(dropout)
            self.capture_attention = False   # set True by the attention-figure cell
            self.last_weights = None         # weights of the last forward pass, when kept

        def forward(self, x, mask):
            scores = self.score(x).squeeze(-1)
            scores = scores.masked_fill(mask, -1e9)  # mask is True on padding
            weights = self.dropout(scores.softmax(dim=-1))
            if self.capture_attention:
                self.last_weights = weights.detach()
            return (weights.unsqueeze(-1) * x).sum(dim=1)


    POOLINGS = ("final", "mean", "max", "attention")
    ENCODERS = ("two_tower", "single")
    PASSAGE_ENCODERS = ("bigru", "none")
    CROSS_DIRECTIONS = ("both", "qa_to_passage")
    ATTENTIONS = ("self", "cross", "inter", "none")
    HEADS = ("concat", "interaction", "compare", "match")
    # Numbers `_passage_match_features` appends: three proportions/means over token
    # matches plus two pooled statistics of the encoder-state match.
    MATCH_FEATURE_DIM = 6


    def pool_sequence(x, h_n, mask, pooling, attn_pool=None):
        """Aggregate one tower's padded BiGRU states into a single vector.

        `mask` is True on the timesteps to ignore. pooling: "final" (concatenated final
        hidden states), "mean", "max" or "attention".
        """
        if pooling == "final":
            return torch.cat([h_n[-2], h_n[-1]], dim=-1)
        if pooling == "mean":
            return masked_mean(x, mask)
        if pooling == "max":
            return masked_max(x, mask)
        if pooling == "attention":
            return attn_pool(x, mask)
        raise ValueError(f"unknown pooling: {pooling!r} (expected one of {POOLINGS})")


    def answer_padding_mask(qa_ids):
        """Pooling mask (True on the positions to ignore) that keeps only the answer.

        `encode_qa` lays the tower out as `question <sep> answer <sep> <pad>...`, so the
        answer is the run of positions between the first two `<sep>` markers. Pooling it
        rather than the whole tower stops the (usually longer) question from diluting the
        vector the head has to judge.
        """
        seq_len = qa_ids.size(1)
        idx = torch.arange(seq_len, device=qa_ids.device).expand_as(qa_ids)
        is_sep = qa_ids == SEP_ID
        sentinel = torch.full_like(idx, seq_len)
        first_sep = torch.where(is_sep, idx, sentinel).min(dim=1).values
        second_sep = torch.where(is_sep & (idx > first_sep[:, None]), idx,
                                 sentinel).min(dim=1).values
        is_answer = (idx > first_sep[:, None]) & (idx < second_sep[:, None])
        return ~is_answer


    def joint_sequence(passage_ids, qa_ids):
        """Concatenate the two towers' ids into one sequence for the single encoder.

        A stable sort on the padding flag moves the real tokens to the front, so all
        padding ends up trailing and `pack_padded_sequence` can handle it.
        """
        combined = torch.cat([passage_ids, qa_ids], dim=1)
        order = (combined == PAD_ID).argsort(dim=1, stable=True)
        return combined.gather(1, order)


    def _head_input(passage_vec, qa_vec, head):
        """What the classifier sees of the two pooled vectors.

        "interaction" -- `[u; v; |u - v|; u * v]`.
        "compare"     -- `[|u - v|; u * v]`: every term contains the passage, so a
                         ranking that ignores it is much harder to express.
        "concat"      -- `[u; v]`.
        """
        if head == "interaction":
            return torch.cat([passage_vec, qa_vec, (passage_vec - qa_vec).abs(),
                              passage_vec * qa_vec], dim=-1)
        if head == "compare":
            return torch.cat([(passage_vec - qa_vec).abs(), passage_vec * qa_vec], dim=-1)
        return torch.cat([passage_vec, qa_vec], dim=-1)


    def _passage_match_features(passage_ids, qa_ids, passage_mask, embedding, match_w,
                                qa_states=None, passage_states=None, state_query=None,
                                state_key=None, scale=1.0):
        """Six *passage-conditioned* alignment features.

        Every quantity contains the passage, so a classifier that receives only these
        cannot rank a question's candidates without reading it:
          [0] |answer tokens present in the passage| / |answer tokens|
          [1] the same count over |passage tokens|
          [2] max over answer positions of the best learned bilinear token match
          [3] the same, averaged over answer positions
          [4]/[5] the same two statistics on the *encoder states*, which keeps the
                  BiGRUs and attention in the gradient path of the score.

        `PAD_ID`/`UNK_ID` are excluded from the overlap sets: an unknown word carries no
        identity, so counting it as a match would invent evidence.
        """
        answer_mask = answer_padding_mask(qa_ids)
        keep_passage = (passage_ids != PAD_ID) & (passage_ids != UNK_ID)
        answer_tokens = ~answer_mask & (qa_ids != PAD_ID) & (qa_ids != UNK_ID)
        present = ((qa_ids[:, :, None] == passage_ids[:, None, :])
                   & keep_passage[:, None, :] & answer_tokens[:, :, None]).any(dim=-1)
        n_answer = (~answer_mask & (qa_ids != PAD_ID)).sum(1).clamp(min=1)
        n_passage = keep_passage.sum(1).clamp(min=1)
        similarity = torch.bmm(
            match_w(nn.functional.normalize(embedding(qa_ids), dim=-1)),
            nn.functional.normalize(embedding(passage_ids), dim=-1).transpose(1, 2))
        similarity = similarity.masked_fill(passage_mask[:, None, :], -1e9)
        per_answer = similarity.max(dim=-1).values
        best = per_answer.masked_fill(answer_mask, -1e9).max(dim=1).values
        best = torch.where((~answer_mask).any(dim=1), best, torch.zeros_like(best))
        mean_best = (per_answer * (~answer_mask)).sum(1) / (~answer_mask).sum(1).clamp(min=1)
        features = [present.sum(1).float() / n_answer,
                    present.sum(1).float() / n_passage, best, mean_best]
        if qa_states is None:
            # The single encoder has no separate towers to match between.
            zero = torch.zeros_like(best)
            return torch.cat([torch.stack(features, dim=-1), zero.unsqueeze(-1),
                              zero.unsqueeze(-1)], dim=-1)
        state_sim = torch.bmm(torch.tanh(state_query(qa_states)),
                              torch.tanh(state_key(passage_states)).transpose(1, 2)) / scale
        state_sim = state_sim.masked_fill(passage_mask[:, None, :], -1e9)
        per_state = state_sim.max(dim=-1).values
        best_state = per_state.masked_fill(answer_mask, -1e9).max(dim=1).values
        best_state = torch.where((~answer_mask).any(dim=1), best_state,
                                 torch.zeros_like(best_state))
        mean_state = (per_state * (~answer_mask)).sum(1) / (~answer_mask).sum(1).clamp(min=1)
        return torch.cat([torch.stack(features, dim=-1), best_state.unsqueeze(-1),
                          mean_state.unsqueeze(-1)], dim=-1)


    class BiGRUClassifier(nn.Module):
        """BiGRU classifier with a pluggable encoder, attention, pooling and head.

        encoder "two_tower" (main model): the passage and the `question <sep> answer`
        span are encoded by separate BiGRUs, then related and pooled. `attention` picks
        the mechanism ("self", "cross" in the direction set by `cross_direction`, the
        ESIM "inter" alignment, or "none"). `pooling` aggregates the passage tower and
        `qa_pooling` the question/answer tower ("answer" = the answer span only).

        encoder "single" (ablation baseline): one BiGRU over
        `passage <sep> question <sep> answer`. Only "self"/"none" attention apply, and
        `qa_pooling`/`head` are ignored (a single pooled vector).
        """

        def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=1,
                     encoder="two_tower", attention="cross", pooling="mean",
                     qa_pooling="answer", passage_encoder="bigru",
                     cross_direction="both", head="concat", dropout=0.2,
                     match_features=False):
            super().__init__()
            if encoder not in ENCODERS:
                raise ValueError(f"unknown encoder: {encoder!r} (expected one of {ENCODERS})")
            if attention not in ATTENTIONS:
                raise ValueError(f"unknown attention: {attention!r} (expected one of {ATTENTIONS})")
            if passage_encoder not in PASSAGE_ENCODERS:
                raise ValueError(f"unknown passage_encoder: {passage_encoder!r} "
                                 f"(expected one of {PASSAGE_ENCODERS})")
            if head not in HEADS:
                raise ValueError(f"unknown head: {head!r} (expected one of {HEADS})")
            if head == "match" and not match_features:
                raise ValueError("head='match' *is* the match features, so it needs "
                                 "match_features=True")
            if cross_direction not in CROSS_DIRECTIONS:
                raise ValueError(f"unknown cross_direction: {cross_direction!r} "
                                 f"(expected one of {CROSS_DIRECTIONS})")
            if encoder == "single" and attention in ("cross", "inter"):
                raise ValueError("encoder='single' has one tower, so cross-attention "
                                 "and inter-attention are undefined; use "
                                 "attention='self' or 'none'")
            if encoder == "two_tower" and passage_encoder == "none" and pooling == "final":
                raise ValueError("pooling='final' reads the passage BiGRU's final state, "
                                 "so it needs passage_encoder='bigru'")
            self.encoder_name = encoder
            self.attention_name = attention
            self.pooling_name = pooling
            self.qa_pooling = qa_pooling
            self.passage_encoder_name = passage_encoder
            self.cross_direction = cross_direction
            self.head_name = head

            self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
            self.emb_dropout = nn.Dropout(dropout)
            # The bilinear token match starts at the identity (the training-free
            # "max cosine over the embedding table" readout) and is sharpened from there.
            self.match_features = bool(match_features)
            self.match_w = nn.Linear(d_model, d_model, bias=False) if self.match_features else None
            self.match_state_q = nn.Linear(d_model, d_model) if self.match_features else None
            self.match_state_k = nn.Linear(d_model, d_model) if self.match_features else None
            self.match_scale = float(d_model) ** 0.5
            if self.match_w is not None:
                with torch.no_grad():
                    self.match_w.weight.copy_(torch.eye(d_model))
            # PyTorch only applies `dropout` between stacked recurrent layers.
            gru_dropout = dropout if num_layers > 1 else 0.0

            self.inter = None
            if encoder == "two_tower":
                self.passage_gru = (nn.GRU(d_model, d_model // 2, num_layers=num_layers,
                                           batch_first=True, bidirectional=True,
                                           dropout=gru_dropout)
                                    if passage_encoder == "bigru" else None)
                self.qa_gru = nn.GRU(d_model, d_model // 2, num_layers=num_layers,
                                     batch_first=True, bidirectional=True,
                                     dropout=gru_dropout)
                self.passage_norm = nn.LayerNorm(d_model)
                self.qa_norm = nn.LayerNorm(d_model)
                self.passage_self_attn = SelfAttention(d_model, num_heads, dropout) if attention == "self" else None
                self.qa_self_attn = SelfAttention(d_model, num_heads, dropout) if attention == "self" else None
                self.passage_to_qa = (CrossAttention(d_model, num_heads, dropout)
                                      if attention == "cross" and cross_direction == "both" else None)
                self.qa_to_passage = (CrossAttention(d_model, num_heads, dropout)
                                      if attention == "cross" else None)
                self.inter = (InterAttention(d_model, dropout)
                              if attention == "inter" else None)
                self.passage_pool = AttentionPooling(d_model, dropout) if pooling == "attention" else None
                self.qa_pool = AttentionPooling(d_model, dropout) if pooling == "attention" else None
                if head == "match":
                    head_in = MATCH_FEATURE_DIM
                else:
                    head_in = 4 * d_model if head == "interaction" else 2 * d_model
                    if self.match_features:
                        head_in += MATCH_FEATURE_DIM
            else:
                self.joint_gru = nn.GRU(d_model, d_model // 2, num_layers=num_layers,
                                        batch_first=True, bidirectional=True,
                                        dropout=gru_dropout)
                self.joint_norm = nn.LayerNorm(d_model)
                self.joint_self_attn = SelfAttention(d_model, num_heads, dropout) if attention == "self" else None
                self.joint_pool = AttentionPooling(d_model, dropout) if pooling == "attention" else None
                head_in = d_model + (MATCH_FEATURE_DIM if self.match_features else 0)

            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(head_in, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )

        def forward(self, passage_ids, qa_ids):
            if self.encoder_name == "single":
                return self._forward_single(passage_ids, qa_ids)
            return self._forward_two_tower(passage_ids, qa_ids)

        def _forward_two_tower(self, passage_ids, qa_ids):
            passage_mask = passage_ids == PAD_ID  # True on padding
            qa_mask = qa_ids == PAD_ID

            passage_emb = self.emb_dropout(self.embedding(passage_ids))
            if self.passage_gru is not None:
                passage, passage_h = run_bigru(self.passage_gru, passage_emb,
                                               passage_mask)
            else:
                # No passage BiGRU: the passage reaches attention/pooling as embeddings.
                passage, passage_h = passage_emb, None
            passage = self.dropout(passage)

            qa, qa_h = run_bigru(
                self.qa_gru, self.emb_dropout(self.embedding(qa_ids)), qa_mask)
            qa = self.dropout(qa)

            if self.passage_self_attn is not None:
                passage = passage + self.dropout(
                    self.passage_self_attn(self.passage_norm(passage), passage_mask))
                qa = qa + self.dropout(self.qa_self_attn(self.qa_norm(qa), qa_mask))
            elif self.inter is not None:
                qa, passage = self.inter(self.qa_norm(qa), self.passage_norm(passage),
                                         qa_mask, passage_mask)
            elif self.qa_to_passage is not None:
                # passage <-> question/answer, with a residual so the BiGRU states keep
                # their signal. With cross_direction="qa_to_passage" only the second line runs.
                p_norm = self.passage_norm(passage)
                q_norm = self.qa_norm(qa)
                if self.passage_to_qa is not None:
                    passage = passage + self.dropout(
                        self.passage_to_qa(p_norm, q_norm, key_mask=qa_mask))
                qa = qa + self.dropout(
                    self.qa_to_passage(q_norm, p_norm, key_mask=passage_mask))

            # The head judges the answer, so pool the question/answer tower over the answer span.
            qa_pool_mask = answer_padding_mask(qa_ids) if self.qa_pooling == "answer" else qa_mask
            passage_vec = pool_sequence(passage, passage_h, passage_mask, self.pooling_name,
                                        self.passage_pool)
            qa_vec = pool_sequence(qa, qa_h, qa_pool_mask, self.pooling_name, self.qa_pool)
            if self.match_features:
                # `head="match"` uses *only* these, so the answer-prior route is closed.
                _match = _passage_match_features(
                    passage_ids, qa_ids, passage_mask, self.embedding, self.match_w,
                    qa_states=qa, passage_states=passage, state_query=self.match_state_q,
                    state_key=self.match_state_k, scale=self.match_scale)
                pooled = (_match if self.head_name == "match" else
                          torch.cat([_head_input(passage_vec, qa_vec, self.head_name),
                                     _match], dim=-1))
            else:
                pooled = _head_input(passage_vec, qa_vec, self.head_name)
            return self.classifier(self.dropout(pooled)).squeeze(-1)

        def _forward_single(self, passage_ids, qa_ids):
            ids = joint_sequence(passage_ids, qa_ids)
            mask = ids == PAD_ID
            states, h_n = run_bigru(self.joint_gru, self.emb_dropout(self.embedding(ids)),
                                    mask)
            states = self.dropout(states)
            if self.joint_self_attn is not None:
                states = states + self.dropout(
                    self.joint_self_attn(self.joint_norm(states), mask))
            pooled = pool_sequence(states, h_n, mask, self.pooling_name, self.joint_pool)
            if self.match_features:
                pooled = torch.cat([pooled, _passage_match_features(
                    passage_ids, qa_ids, passage_ids == PAD_ID, self.embedding,
                    self.match_w)], dim=-1)
            return self.classifier(self.dropout(pooled)).squeeze(-1)


    return (BiGRUClassifier,)


@app.cell
def _(
    ATTENTION,
    BiGRUClassifier,
    CROSS_DIRECTION,
    DROPOUT,
    D_MODEL,
    EARLY_STOPPING,
    EARLY_STOP_METRIC,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_MIN_EPOCHS,
    EARLY_STOP_PATIENCE,
    ENCODER,
    GRAD_CLIP,
    HEAD,
    LOSS,
    LR,
    LR_DECAY,
    LR_MIN,
    LR_WARMUP_EPOCHS,
    MATCH_FEATURES,
    NUM_EPOCHS,
    NUM_HEADS,
    NUM_LAYERS,
    PAIRWISE_WEIGHT,
    PASSAGE_ENCODER,
    POOLING,
    POS_WEIGHT,
    QA_POOLING,
    REPORT_SWEEP_DIAGNOSTIC,
    SEED,
    THRESHOLD_FIXED,
    THRESHOLD_MAX,
    THRESHOLD_METRIC,
    THRESHOLD_MIN,
    THRESHOLD_RULE,
    THRESHOLD_STEP,
    TUNE_THRESHOLD,
    USE_CLASS_WEIGHTS,
    USE_TF32,
    WEIGHT_DECAY,
    math,
    nn,
    torch,
    vocab_size,
):
    import torch.optim as optim
    from tqdm import tqdm
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 average_precision_score)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda" and USE_TF32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("TF32 enabled for matmul + cuDNN")
        except Exception as _exc:  # depends on the torch build
            print(f"TF32 not available: {type(_exc).__name__}: {_exc}")


    def per_question_metrics(qids, labels, preds, scores=None):
        """MultiRC's canonical evaluation: a question is correct only when *every*
        candidate answer under it is labelled correctly.

        `q_em` and `q_f1` are averaged over questions. When `scores` is passed, the
        threshold-free within-question ranking metrics `q_auc` and `q_ap` are added
        (they compare only the candidates that share a question). A question whose
        candidates all share one label has no comparison and is left out. The
        per-question values are returned too, because a mean over questions is not a
        mean over independent samples (a passage's questions share its text).
        """
        groups = {}
        for _qid, _gold, _pred in zip(qids, labels, preds):
            groups.setdefault(_qid, []).append((_gold, _pred))
        _by_qid = {}
        if scores is not None:
            for _qid, _gold, _score in zip(qids, labels, scores):
                _by_qid.setdefault(_qid, []).append((_gold, _score))

        em_total = f1_total = 0.0
        for _rows in groups.values():
            em_total += float(all(_gold == _pred for _gold, _pred in _rows))
            _gold_set = {_i for _i, (_gold, _) in enumerate(_rows) if _gold == 1}
            _pred_set = {_i for _i, (_, _pred) in enumerate(_rows) if _pred == 1}
            if not _gold_set and not _pred_set:
                f1_total += 1.0
            elif _gold_set and _pred_set:
                f1_total += 2 * len(_gold_set & _pred_set) / (len(_gold_set) + len(_pred_set))
            # one set empty -> contributes 0
        n_questions = max(len(groups), 1)
        out = {"q_em": em_total / n_questions, "q_f1": f1_total / n_questions,
               "n_questions": len(groups)}
        if scores is not None:
            _auc_by_question, _ap_by_question = {}, {}
            for _qid, _rows in _by_qid.items():
                _gold_vec = [_gold for _gold, _ in _rows]
                _score_vec = [_score for _, _score in _rows]
                if len(set(_gold_vec)) < 2:
                    continue  # no within-question comparison to score
                _auc_by_question[_qid] = roc_auc_score(_gold_vec, _score_vec)
                _ap_by_question[_qid] = average_precision_score(_gold_vec, _score_vec)
            _aucs = list(_auc_by_question.values())
            _aps = list(_ap_by_question.values())
            out["q_auc"] = sum(_aucs) / len(_aucs) if _aucs else 0.5
            out["q_ap"] = sum(_aps) / len(_aps) if _aps else 0.0
            out["n_questions_ranked"] = len(_aucs)
            out["q_auc_by_question"] = _auc_by_question
            out["q_ap_by_question"] = _ap_by_question
        return out


    def evaluate(model, data_loader, threshold=0.5, rule=THRESHOLD_RULE):
        """Answer-level and per-question metrics for a whole split.

        `loss` is the plain unweighted BCE so validation curves stay comparable; `ap`
        and `auc` are answer-level ranking metrics; `q_em`/`q_f1` are the per-question
        metrics and `q_auc`/`q_ap` their threshold-free within-question counterparts.
        `threshold` cuts the scores that `rule` produces (`decision_scores`), so the
        thresholded numbers describe the operating point actually being reported.
        """
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        labels, scores, qids = [], [], []
        total_loss, num_batches = 0.0, 0

        # Keep the split's logits on the GPU and hop to the host once, instead of
        # syncing (`.cpu().tolist()`, `.item()`) on every batch.
        _logits, _labels, _qids = [], [], []
        with torch.no_grad():
            for passage_ids, qa_ids, targets, batch_qids in data_loader:
                passage_ids = passage_ids.to(DEVICE, non_blocking=True)
                qa_ids = qa_ids.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                logits = model(passage_ids, qa_ids)
                total_loss = total_loss + criterion(logits, targets).detach()
                num_batches += 1
                _logits.append(logits.detach())
                _labels.append(targets.detach())
                _qids.append(batch_qids)
        if _logits:
            scores = torch.sigmoid(torch.cat(_logits)).cpu().tolist()
            labels = torch.cat(_labels).long().cpu().tolist()
            qids = torch.cat(_qids).tolist()

        preds = [int(_s > threshold) for _s in decision_scores(scores, qids, rule)]
        metrics = {
            "loss": float(total_loss / max(num_batches, 1)),
            "rule": rule,
            "acc": accuracy_score(labels, preds),
            "yes_rate": sum(preds) / max(len(preds), 1),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5,
            "ap": average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.0,
            "labels": labels,
            "scores": scores,
            "qids": qids,
        }
        metrics.update(per_question_metrics(qids, labels, preds, scores))
        return metrics


    def predicted_positive_rate(scores, threshold):
        """Share of `scores` above `threshold`, so a degenerate cut is visible."""
        if not scores:
            return 0.0
        return sum(1 for _s in scores if _s > threshold) / len(scores)


    def centre_within_question(scores, qids):
        """Subtract each question's mean score, so its candidates are comparable.

        The score level is not comparable across questions, which is what drives a
        single global cut to all-no / all-yes on some questions. Centring removes that
        offset and nothing else: it never looks at a label, and it cannot change any
        within-question ordering, so `q_auc`/`q_ap` are untouched.
        """
        _totals, _counts = {}, {}
        for _qid, _score in zip(qids, scores):
            _totals[_qid] = _totals.get(_qid, 0.0) + _score
            _counts[_qid] = _counts.get(_qid, 0) + 1
        return [_score - _totals[_qid] / _counts[_qid]
                for _score, _qid in zip(scores, qids)]


    def decision_scores(scores, qids, rule=THRESHOLD_RULE):
        """The scores a decision rule is applied to, after any per-group transform."""
        if rule == "within_question":
            if qids is None:
                raise ValueError("THRESHOLD_RULE='within_question' needs the per-answer "
                                 "question ids to centre within")
            return centre_within_question(scores, qids)
        return list(scores)


    def declared_threshold(labels, scores, qids=None, rule=THRESHOLD_RULE, fixed=THRESHOLD_FIXED):
        """The decision cut used for every reported number: declared, not fitted.

        Under the unweighted loss the scores need a *shift*, not a search. The cut is the
        validation score quantile at which the predicted-positive rate matches the
        validation positive rate, taken from the top of the sorted scores so ties can
        only make it more conservative. A metric-maximising sweep is deliberately not
        used here -- see `sweep_threshold`.
        """
        if rule == "fixed":
            return float(fixed)
        if rule not in ("prior_match", "within_question"):
            raise ValueError(f"unknown threshold rule: {rule!r} "
                             f"(within_question | prior_match | fixed)")
        _decision = decision_scores(scores, qids, rule)
        _rate = float(sum(labels)) / max(len(labels), 1)
        _descending = sorted(_decision, reverse=True)
        _k = min(int(round(_rate * len(_descending))), len(_descending) - 1)
        return float(_descending[_k])


    def sweep_threshold(qids, labels, scores, metric=THRESHOLD_METRIC,
                        lo=THRESHOLD_MIN, hi=THRESHOLD_MAX, step=THRESHOLD_STEP):
        """Threshold maximising `metric` on `labels`/`scores`, swept over [lo, hi].

        DIAGNOSTIC ONLY: `q_f1` is maximised by predicting yes for every answer when the
        ranking is weak, so the sweep walks to the top of the range. Nothing reported here.
        """
        n_steps = int(round((hi - lo) / step))
        grid = [lo + _i * step for _i in range(n_steps + 1)]
        values = []
        for _t in grid:
            _preds = [int(s > _t) for s in scores]
            if metric == "q_f1":
                values.append(per_question_metrics(qids, labels, _preds)["q_f1"])
            else:
                values.append(f1_score(labels, _preds, zero_division=0))
        best = max(range(len(grid)), key=lambda _i: values[_i])
        return float(grid[best]), float(values[best])


    LOSSES = ("bce", "within_question")
    LOSS_TAGS = {"bce": "bce", "within_question": "wq"}
    # Every head needs its own tag: a run is registered under the label its tag produces, so
    # two heads sharing a tag would silently overwrite each other in `experiments`/`models`
    # (and in the on-disk cache), and the head x objective ablation would drop runs.
    HEAD_TAGS = {"concat": "cc", "interaction": "ix", "compare": "cmp", "match": "mt"}


    def variant_tag(head=HEAD, loss=LOSS):
        """Short tag for the head and the objective, e.g. 'ix+wq'.

        Both change what the model learns, so both belong in the label a run is registered
        under: otherwise the summary would pool incompatible runs, and two heads sharing a
        tag would overwrite one another.
        """
        return f"{HEAD_TAGS.get(head, head)}+{LOSS_TAGS.get(loss, loss)}"


    def run_label(encoder, attention, pooling, seed=None, variant=None):
        """Registry key for one configuration: architecture + optional variant + seed."""
        key = f"{encoder}/{attention}/{pooling}"
        if variant:
            key = f"{key}/{variant}"
        return key if seed is None else f"{key}/s{seed}"


    def create_model(encoder=ENCODER, attention=ATTENTION, pooling=POOLING,
                     qa_pooling=QA_POOLING, passage_encoder=PASSAGE_ENCODER,
                     cross_direction=CROSS_DIRECTION, head=HEAD, d_model=D_MODEL,
                     num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
                     match_features=MATCH_FEATURES, seed=SEED):
        """Build the configured model, seeding its initial weights.

        Seeding happens *here* and not only in `train_model`, because weight
        initialisation consumes the RNG and `create_model` is called at different points
        in the stream; without it two runs of the same config would start differently.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        return BiGRUClassifier(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            encoder=encoder,
            attention=attention,
            pooling=pooling,
            qa_pooling=qa_pooling,
            passage_encoder=passage_encoder,
            cross_direction=cross_direction,
            head=head,
            dropout=dropout,
            match_features=match_features,
        ).to(DEVICE)


    def lr_for_epoch(epoch, num_epochs, base_lr=LR, warmup_epochs=LR_WARMUP_EPOCHS,
                     decay=LR_DECAY, min_lr=LR_MIN):
        """Learning rate for one epoch: linear warmup, then cosine decay.

        A pure function of the epoch index, so two runs with the same budget see the same
        LR at the same epoch and the comparison stays controlled.
        """
        if warmup_epochs and epoch < warmup_epochs:
            return base_lr * (epoch + 1) / warmup_epochs
        if decay == "cosine":
            span = max(1, num_epochs - warmup_epochs)
            progress = min(1.0, (epoch - warmup_epochs) / span)
            return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
        return base_lr


    def within_question_loss(logits, targets, qids, pairwise_weight=PAIRWISE_WEIGHT,
                             pos_weight=None):
        """Objective for `LOSS="within_question"`: centred BCE plus a pairwise ranking term.

        Centring each question's logits on their own mean makes the loss invariant to any
        per-question constant the model adds, so the only way down is a better *ordering*
        -- which is what `q_auc` measures. `qids` are the per-answer question ids from
        `collate_batch`, so a batch must hold a question's candidates together
        (`GROUPED_BATCHES`).
        """
        _unique, _inverse = torch.unique(qids, return_inverse=True)
        _totals = torch.zeros(len(_unique), device=logits.device, dtype=logits.dtype)
        _counts = torch.zeros(len(_unique), device=logits.device, dtype=logits.dtype)
        _totals = _totals.index_add(0, _inverse, logits)
        _counts = _counts.index_add(0, _inverse, torch.ones_like(logits))
        _centred = logits - (_totals / _counts.clamp(min=1.0))[_inverse]

        _loss = nn.functional.binary_cross_entropy_with_logits(_centred, targets,
                                                              pos_weight=pos_weight)
        if not pairwise_weight or len(_unique) < 2:
            return _loss
        _ranking, _with_pairs = None, 0
        for _group in range(len(_unique)):
            _mask = _inverse == _group
            _pos = _centred[_mask & (targets > 0.5)]
            _neg = _centred[_mask & (targets <= 0.5)]
            if _pos.numel() and _neg.numel():
                _term = nn.functional.softplus(-(_pos[:, None] - _neg[None, :])).mean()
                _ranking = _term if _ranking is None else _ranking + _term
                _with_pairs += 1
        if _with_pairs:
            _loss = _loss + pairwise_weight * (_ranking / _with_pairs)
        return _loss


    def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS,
                    lr=LR, weight_decay=WEIGHT_DECAY, label="model", seed=SEED,
                    use_class_weights=USE_CLASS_WEIGHTS,
                    early_stopping=EARLY_STOPPING,
                    early_stop_patience=EARLY_STOP_PATIENCE,
                    early_stop_min_delta=EARLY_STOP_MIN_DELTA,
                    early_stop_metric=EARLY_STOP_METRIC,
                    early_stop_min_epochs=EARLY_STOP_MIN_EPOCHS,
                    loss=LOSS, pairwise_weight=PAIRWISE_WEIGHT):
        """Train up to `num_epochs`, keeping the checkpoint with the best metric.

        The kept state is the best-`early_stop_metric` epoch, selected epoch by epoch, so
        the stopping rule never changes which checkpoint is reported -- it only avoids
        paying for epochs that cannot beat it. The cut is declared on the kept
        checkpoint's validation scores (`declared_threshold`, no metric sweep).
        """
        # Re-seed so dropout masks, shuffling and the gradient stream match across runs.
        # The loaders keep `generator=None`: a shared generator would advance across runs.
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # No weight decay on biases, LayerNorm parameters or the embedding table.
        decay, no_decay = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or name.startswith("embedding"):
                no_decay.append(param)
            else:
                decay.append(param)
        optimizer = optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=lr)

        if loss not in LOSSES:
            raise ValueError(f"unknown loss: {loss!r} (expected one of {LOSSES})")
        # `criterion=None` marks the objective that is a function of the whole batch.
        _pos_weight = POS_WEIGHT.to(DEVICE) if use_class_weights else None
        criterion = (None if loss == "within_question"
                     else nn.BCEWithLogitsLoss(pos_weight=_pos_weight))

        metric_key = (early_stop_metric if early_stop_metric.startswith("val_")
                      else f"val_{early_stop_metric}")
        higher_is_better = metric_key != "val_loss"

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [],
                   "val_auc": [], "val_ap": [], "val_q_em": [], "val_q_f1": [],
                   "val_q_auc": [], "val_q_ap": [], "lr": [],
                       "val_q_f1_declared": [],
                   "best_epoch": None, "best_score": None,
                   "best_val_ap": None, "best_val_q_f1": None, "best_val_auc": None,
                   "best_val_f1": None, "best_val_q_auc": None, "best_val_q_ap": None,
                       "best_val_q_f1_declared": None,
                   "stopped_epoch": None, "epochs_run": None, "budget": num_epochs,
                   "early_stopping": bool(early_stopping),
                   "loss_name": loss, "pairwise_weight": pairwise_weight,
                   "threshold": 0.5, "threshold_score": None, "threshold_q_f1": None,
                   "threshold_rule": None,
                   "threshold_yes_rate": None, "sweep_threshold": None,
                   "sweep_score": None, "sweep_yes_rate": None, "truncation": None,
                   "config": None, "seed": seed}
        best_score = -float("inf") if higher_is_better else float("inf")
        best_state = None
        epochs_without_improvement = 0

        def improved(score):
            if higher_is_better:
                return score > best_score + early_stop_min_delta
            return score < best_score - early_stop_min_delta

        for epoch in range(num_epochs):
            current_lr = lr_for_epoch(epoch, num_epochs)
            for _group in optimizer.param_groups:
                _group["lr"] = current_lr

            model.train()
            total_loss, num_batches = 0.0, 0
            pbar = tqdm(train_loader, desc=f"[{label}] epoch {epoch+1}/{num_epochs}",
                        leave=False)
            for passage_ids, qa_ids, targets, batch_qids in pbar:
                passage_ids = passage_ids.to(DEVICE, non_blocking=True)
                qa_ids = qa_ids.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)
                batch_qids = batch_qids.to(DEVICE, non_blocking=True)

                optimizer.zero_grad()
                logits = model(passage_ids, qa_ids)
                if criterion is None:
                    batch_loss = within_question_loss(logits, targets, batch_qids,
                                                      pairwise_weight, _pos_weight)
                else:
                    batch_loss = criterion(logits, targets)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

                # Accumulate on the GPU: `.item()` forces a device sync, so pay for it
                # once per epoch rather than twice per batch. `total_loss` is a tensor
                # here and is read with the single float() below.
                total_loss = total_loss + batch_loss.detach()
                num_batches += 1
                if num_batches % 20 == 0:   # bar kept; only the postfix is throttled
                    pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            train_loss = float(total_loss / max(num_batches, 1))
            # The per-epoch curve is read at a fixed cut on the raw scores so the epochs
            # stay comparable; the reported numbers use the run's declared rule.
            val = evaluate(model, val_loader, rule="prior_match")

            # The operating point the *report* uses: the declared cut (prior-matched, on
            # within-question-centred scores -- see `declared_threshold`), not the arbitrary
            # 0.5 cut the curves above are read at. It costs no extra forward pass (`val`
            # already carries labels/scores/qids), and it makes the trajectory of the metric
            # we report visible -- so `EARLY_STOP_METRIC` can be pointed at the reported
            # number if ranking and calibration ever disagree.
            _cut = declared_threshold(val["labels"], val["scores"], val["qids"])
            _declared_preds = [int(_s > _cut) for _s in
                               decision_scores(val["scores"], val["qids"], THRESHOLD_RULE)]
            val["q_f1_declared"] = per_question_metrics(
                val["qids"], val["labels"], _declared_preds)["q_f1"]

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val["loss"])
            history["val_acc"].append(val["acc"])
            history["val_f1"].append(val["f1"])
            history["val_auc"].append(val["auc"])
            history["val_ap"].append(val["ap"])
            history["val_q_em"].append(val["q_em"])
            history["val_q_f1"].append(val["q_f1"])
            history["val_q_auc"].append(val["q_auc"])
            history["val_q_ap"].append(val["q_ap"])
            history["val_q_f1_declared"].append(val["q_f1_declared"])
            history["lr"].append(current_lr)

            print(f"[{label}] epoch {epoch+1}: train_loss={history['train_loss'][-1]:.4f} "
                  f"val_loss={val['loss']:.4f} val_ap={val['ap']:.4f} "
                  f"val_q_f1={val['q_f1']:.4f} val_auc={val['auc']:.4f} "
                  f"q_f1@declared={val['q_f1_declared']:.4f} "
                  f"lr={current_lr:.2e}")

            score = val[metric_key[4:]]
            if improved(score):
                best_score = score
                history["best_epoch"] = epoch + 1
                history["best_score"] = score
                history["best_val_ap"] = val["ap"]
                history["best_val_q_f1"] = val["q_f1"]
                history["best_val_auc"] = val["auc"]
                history["best_val_f1"] = val["f1"]
                history["best_val_q_auc"] = val["q_auc"]
                history["best_val_q_ap"] = val["q_ap"]
                history["best_val_q_f1_declared"] = val["q_f1_declared"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if (early_stopping and epochs_without_improvement >= early_stop_patience
                    and epoch + 1 >= early_stop_min_epochs):
                history["stopped_epoch"] = epoch + 1
                print(f"[{label}] early stopping at epoch {epoch+1}: no {metric_key} "
                      f"improvement for {early_stop_patience} epochs "
                      f"(best epoch {history['best_epoch']})")
                break

        history["epochs_run"] = len(history["train_loss"])

        if best_state is not None:
            model.load_state_dict(best_state)

        if early_stopping and history["stopped_epoch"] is None:
            print(f"[{label}] WARNING: the stopping rule never fired -- the run used "
                  f"the whole budget ({num_epochs} epochs), so the last epoch was "
                  f"reported rather than a chosen one.")

        if TUNE_THRESHOLD:
            val = evaluate(model, val_loader, rule="prior_match")  # kept checkpoint
            history["threshold"] = declared_threshold(val["labels"], val["scores"],
                                                     val["qids"])
            history["threshold_rule"] = THRESHOLD_RULE
            _decision = decision_scores(val["scores"], val["qids"], THRESHOLD_RULE)
            _cut_preds = [int(_s > history["threshold"]) for _s in _decision]
            history["threshold_yes_rate"] = sum(_cut_preds) / max(len(_cut_preds), 1)
            history["threshold_score"] = f1_score(val["labels"], _cut_preds,
                                                 zero_division=0)
            history["threshold_q_f1"] = per_question_metrics(
                val["qids"], val["labels"], _cut_preds)["q_f1"]
            _gold_rate = sum(val["labels"]) / max(len(val["labels"]), 1)
            print(f"[{label}] decision cut={history['threshold']:.4f} "
                  f"({THRESHOLD_RULE}, declared on validation): predicts yes for "
                  f"{history['threshold_yes_rate']:.1%} of validation answers "
                  f"(gold positive rate {_gold_rate:.1%}), val answer-level "
                  f"F1={history['threshold_score']:.4f}, val q_f1="
                  f"{history['threshold_q_f1']:.4f}")
            if REPORT_SWEEP_DIAGNOSTIC:
                _sweep_thr, _sweep_score = sweep_threshold(
                    val["qids"], val["labels"], val["scores"])
                history["sweep_threshold"] = _sweep_thr
                history["sweep_score"] = _sweep_score
                history["sweep_yes_rate"] = predicted_positive_rate(val["scores"],
                                                                   _sweep_thr)
                print(f"[{label}] (diagnostic, NOT reported) {THRESHOLD_METRIC}-"
                      f"maximising sweep: cut={_sweep_thr:.2f}, val "
                      f"{THRESHOLD_METRIC}={_sweep_score:.4f} at a yes-rate of "
                      f"{history['sweep_yes_rate']:.1%}.")
        return history


    return (
        DEVICE,
        accuracy_score,
        average_precision_score,
        create_model,
        decision_scores,
        declared_threshold,
        evaluate,
        f1_score,
        per_question_metrics,
        roc_auc_score,
        run_label,
        train_model,
        variant_tag,
    )


@app.cell
def persistence_notes(mo):
    mo.md(r"""
    ## Saving and reloading results

    The study is deterministic -- one `SEED` drives the train/validation split, the weight
    initialisation and the batch order -- so the notebook always computes the same numbers.
    That means they can be written out once and read back in, rather than retrained. Both
    switches are in the hyperparameters cell, with the rest of the configuration:

    - **`SAVE_RESULTS = True`** makes the `save_results` cell at the bottom of the notebook write
      `RESULTS_DIR/`: every run's history, the baselines, one weights file per trained model, and
      a manifest of the constructor arguments and the split/vocab it was made under. `RESULTS_DIR`
      is anchored at the notebook's own directory, so the export lands beside the notebook
      (`TaskOutline.md`, `README.md`, ...) and is saved with the workspace.
    - **`LOAD_RESULTS = True`** makes section 3 and the `baselines` cell read that export instead
      of training. The loss/ranking plots, the sample-answer cell, the attention figures, the
      ablation summary and the model-vs-baselines comparison all work unchanged off a loaded
      export, because they read `experiments` / `models` / `baseline_results`.
    - A loaded export is **refused** if its `vocab_size` differs from this notebook's, so a stale
      file cannot silently produce wrong numbers. The manifest also records `SEED`,
      `VAL_FRACTION`, `MIN_FREQ` and the per-run architecture.
    - Keep `SAVE_RESULTS` off while `LOAD_RESULTS` is on, so a loaded (not retrained) session
      does not overwrite the export with itself.
    - **A crashed ablation grid is resumed, not restarted.** The section-4 grid writes every
      finished run to `ABLATION_CACHE_DIR/` as it goes -- history, weights and the settings it was
      trained under -- so re-running that cell reads back whatever is already on disk and trains
      only what is left; a crash costs the single run that was in flight. Each record is written
      through a temporary file and moved into place, so a half-written one can never be mistaken
      for a finished run. `load_ablation_runs()` reads those records straight into `experiments` /
      `models` / `test_loaders` if you would rather skip the grid cell entirely.
    """)
    return


@app.cell
def results_io(
    ABLATION_CACHE_DIR,
    ATTENTION,
    BATCH_SIZE,
    CROSS_DIRECTION,
    DEVICE,
    DROPOUT,
    D_MODEL,
    EARLY_STOPPING,
    EARLY_STOP_METRIC,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_MIN_EPOCHS,
    EARLY_STOP_PATIENCE,
    ENCODER,
    GRAD_CLIP,
    GROUPED_BATCHES,
    HEAD,
    LOSS,
    LR,
    LR_DECAY,
    LR_MIN,
    LR_WARMUP_EPOCHS,
    MATCH_FEATURES,
    MIN_FREQ,
    NUM_HEADS,
    NUM_LAYERS,
    PAIRWISE_WEIGHT,
    PASSAGE_ENCODER,
    PASSAGE_MAX_TOKENS,
    PASSAGE_TRUNCATION,
    POOLING,
    QA_MAX_TOKENS,
    QA_POOLING,
    RESULTS_DIR,
    SEED,
    THRESHOLD_RULE,
    USE_CLASS_WEIGHTS,
    VAL_FRACTION,
    WEIGHT_DECAY,
    build_loaders,
    create_model,
    test_loader,
    torch,
    vocab_size,
):
    # --- Persistence: export the trained runs, and load them back ---------------------
    # The study is deterministic -- one SEED drives the split, the weight initialisation and
    # the batch order -- so the numbers this notebook computes are reproducible. Saving them
    # lets section 3 and the baselines cell be read back instead of retrained:
    #
    #   SAVE_RESULTS = True  (hyperparameters cell) -> the `save_results` cell writes RESULTS_DIR/
    #   LOAD_RESULTS = True  (hyperparameters cell) -> section 3 and the baselines cell read it
    #
    # The export holds the run histories, the baselines, one weights file per trained model,
    # and a manifest of the constructor arguments and the split/vocab it was made under, so a
    # model rebuilds exactly and a stale export is refused instead of silently scoring wrong.
    import json as _json
    import os as _os

    _MANIFEST = "manifest.json"
    _EXPERIMENTS_FILE = "experiments.json"
    _BASELINES_FILE = "baselines.json"
    _WEIGHTS_DIR = "weights"


    def _json_ready(value):
        """JSON fallback: numpy scalars are not serialisable, their float is."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)


    def _model_spec(model):
        """Constructor arguments that rebuild `model` exactly.

        Read off the object, so the export stays valid if the hyperparameters cell is edited
        later; `num_heads`, `num_layers` and `dropout` are notebook-wide, so they come from
        those.
        """
        return {
            "encoder": model.encoder_name,
            "attention": model.attention_name,
            "pooling": model.pooling_name,
            "qa_pooling": model.qa_pooling,
            "passage_encoder": model.passage_encoder_name,
            "cross_direction": model.cross_direction,
            "head": model.head_name,
            "match_features": bool(model.match_features),
            "d_model": int(model.embedding.embedding_dim),
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
        }


    def _run_stem(label):
        """A filesystem-safe stem for one run label, shared by every stored artefact."""
        return "".join(_c if (_c.isalnum() or _c in "-_.") else "_" for _c in label)


    def _weights_name(label):
        """A filesystem-safe filename for one run label."""
        return _run_stem(label) + ".pt"


    def export_results(directory=RESULTS_DIR, experiments=None, models=None,
                       baseline_results=None, baseline_errors=None):
        """Write the runs, the model weights and the baselines under `directory`.

        The `save_results` cell calls this with the notebook's own names; the arguments default
        to those names so it can also be called by hand.
        """
        experiments = globals().get("experiments", {}) if experiments is None else experiments
        models = globals().get("models", {}) if models is None else models
        baseline_results = (globals().get("baseline_results", {}) if baseline_results is None
                            else baseline_results)
        baseline_errors = (globals().get("baseline_errors", {}) if baseline_errors is None
                           else baseline_errors)
        _os.makedirs(_os.path.join(directory, _WEIGHTS_DIR), exist_ok=True)
        specs = {}
        for _label, _model in models.items():
            specs[_label] = _model_spec(_model)
            torch.save(_model.state_dict(),
                       _os.path.join(directory, _WEIGHTS_DIR, _weights_name(_label)))
        with open(_os.path.join(directory, _EXPERIMENTS_FILE), "w") as _fh:
            _json.dump(experiments, _fh, indent=1, default=_json_ready)
        with open(_os.path.join(directory, _BASELINES_FILE), "w") as _fh:
            _json.dump({"results": baseline_results, "errors": baseline_errors}, _fh,
                       indent=1, default=_json_ready)
        _manifest = {
            "vocab_size": int(vocab_size),
            "passage_truncation": PASSAGE_TRUNCATION,
            "split_seed": SEED,
            "val_fraction": VAL_FRACTION,
            "min_freq": MIN_FREQ,
            "runs": sorted(experiments),
            "models": specs,
            "baselines": sorted(baseline_results),
        }
        with open(_os.path.join(directory, _MANIFEST), "w") as _fh:
            _json.dump(_manifest, _fh, indent=1, default=_json_ready)
        print(f"Exported {len(experiments)} run(s), {len(specs)} model(s) and "
              f"{len(baseline_results)} baseline(s) to {directory}/")
        print(f"  {_EXPERIMENTS_FILE}  {_BASELINES_FILE}  {_MANIFEST}  "
              f"{_WEIGHTS_DIR}/ ({len(specs)} file(s))")
        print("  Set LOAD_RESULTS = True in the hyperparameters cell to read it back.")
        return directory


    def _int_keys(obj):
        """JSON turns integer dict keys into strings; the paired tests index qids as ints."""
        if isinstance(obj, dict):
            return {int(_k) if isinstance(_k, str) and _k.lstrip("-").isdigit() else _k:
                    _int_keys(_v) for _k, _v in obj.items()}
        if isinstance(obj, list):
            return [_int_keys(_v) for _v in obj]
        return obj


    def _read_export(directory=RESULTS_DIR):
        """(manifest, experiments, baseline_results, baseline_errors) from `directory`.

        Refuses an export made under a different vocabulary: the model's embedding table would
        not line up and every number would be quietly wrong.
        """
        with open(_os.path.join(directory, _MANIFEST)) as _fh:
            _manifest = _json.load(_fh)
        if _manifest.get("vocab_size") != int(vocab_size):
            raise RuntimeError(
                f"the export in {directory}/ was made with vocab_size "
                f"{_manifest.get('vocab_size')}, this notebook has {vocab_size}; re-export, or "
                f"match MIN_FREQ / the split (VAL_FRACTION, SEED) first")
        with open(_os.path.join(directory, _EXPERIMENTS_FILE)) as _fh:
            experiments = _json.load(_fh)
        with open(_os.path.join(directory, _BASELINES_FILE)) as _fh:
            _bundle = _json.load(_fh)
        return (_manifest, experiments, _int_keys(_bundle.get("results", {})),
                _bundle.get("errors", {}))


    def load_baselines(directory=RESULTS_DIR):
        """(baseline_results, baseline_errors) from an export; the baselines cell uses this."""
        _manifest, _experiments, baseline_results, baseline_errors = _read_export(directory)
        return baseline_results, baseline_errors


    def load_runs(directory=RESULTS_DIR):
        """(experiments, models, test_loaders) from an export, with every model rebuilt.

        Each model comes back from its recorded constructor arguments plus its weights, so the
        plots, the sample-answer cell, the attention figures and the summary table all work off
        a loaded export exactly as off a trained one. A run whose PASSAGE_TRUNCATION differs
        from the notebook default gets its own test loader rebuilt.
        """
        _manifest, experiments, _baselines, _errors = _read_export(directory)
        models, test_loaders, _loaders = {}, {}, {}
        for _label, _spec in _manifest["models"].items():
            _model = create_model(**_spec)
            _model.load_state_dict(torch.load(
                _os.path.join(directory, _WEIGHTS_DIR, _weights_name(_label)),
                map_location=DEVICE))
            _model.eval()
            models[_label] = _model
            _trunc = (experiments.get(_label, {}).get("truncation")
                      or _manifest.get("passage_truncation") or PASSAGE_TRUNCATION)
            if _trunc not in _loaders:
                _loaders[_trunc] = (test_loader if _trunc == PASSAGE_TRUNCATION
                                    else build_loaders(_trunc)[5])
            test_loaders[_label] = _loaders[_trunc]
        print(f"Loaded {len(experiments)} run(s), {len(models)} model(s) and "
              f"{len(_baselines)} baseline(s) from {directory}/ "
              f"(vocab_size {_manifest.get('vocab_size')}, "
              f"split seed {_manifest.get('split_seed')})")
        return experiments, models, test_loaders


    # --- Per-run durable records: the ablation grid's resume store ---------------------
    # The section-4 grid writes one record per finished run *as it finishes*, so an
    # interrupted or crashed grid is picked up where it left off instead of restarted, and a
    # run that was already paid for is never retrained. Layout, under `ABLATION_CACHE_DIR/`:
    #
    #   <stem>.json   {"label", "model", "train", "history", "fingerprint"}
    #   <stem>.pt     the model's `state_dict`
    #
    # `model` holds the constructor arguments (`create_model(**record["model"])` rebuilds it)
    # and `fingerprint` the settings the run was trained under, so a record that predates a
    # hyperparameter change is refused rather than silently reused. Every write goes to a
    # `.partial` file that is moved into place with `os.replace`, so a crash mid-write can
    # never leave a truncated file that later reads back as a finished run.

    def run_paths(label, directory=ABLATION_CACHE_DIR):
        """(metadata path, weights path) for one run's durable record."""
        _stem = _run_stem(label)
        return (_os.path.join(directory, _stem + ".json"),
                _os.path.join(directory, _stem + ".pt"))


    def _write_atomically(_write, _path):
        """Write through a temporary file, then move it into place in one operation."""
        _os.makedirs(_os.path.dirname(_path), exist_ok=True)
        _tmp = _path + ".partial"
        _write(_tmp)
        _os.replace(_tmp, _path)


    def _dump_json(_obj, _path):
        with open(_path, "w") as _fh:
            _json.dump(_obj, _fh, indent=1, default=_json_ready)


    def run_fingerprint(epochs, model_kwargs=None, train_kwargs=None, overrides=None):
        """Every setting a stored run must have been trained under to be reusable.

        Enumerates what a run inherits from the hyperparameters cell -- the encoding limits, the
        split, the schedule, the stopping rule, the architecture defaults and the objective --
        plus this run's own model/training overrides. `overrides` carries the raw variant dict
        for a run defined by one (the quick screen), so a setting that is not a
        `create_model`/`train_model` keyword at all -- the passage truncation, say -- is still
        part of the check. The ablation grid and the quick screen both fingerprint through here;
        a record whose fingerprint does not match is refused rather than reused, so editing the
        configuration can never silently mix runs that were trained under different settings.
        """
        _m, _t = model_kwargs or {}, train_kwargs or {}
        return {
            # budget, data and schedule
            "epochs": epochs, "batch": BATCH_SIZE, "vocab": vocab_size,
            "lr": LR, "warmup": LR_WARMUP_EPOCHS, "decay": LR_DECAY, "lr_min": LR_MIN,
            "wd": WEIGHT_DECAY, "clip": GRAD_CLIP,
            "truncation": PASSAGE_TRUNCATION, "passage_tokens": PASSAGE_MAX_TOKENS,
            "qa_tokens": QA_MAX_TOKENS, "min_freq": MIN_FREQ,
            "val_fraction": VAL_FRACTION, "split_seed": SEED,
            # stopping rule, and what the objective does
            "early": [EARLY_STOPPING, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
                      EARLY_STOP_MIN_EPOCHS, EARLY_STOP_METRIC],
            "class_weights": USE_CLASS_WEIGHTS, "rule": THRESHOLD_RULE,
            "grouped_batches": GROUPED_BATCHES,
            "loss": _t.get("loss", LOSS),
            "pairwise": _t.get("pairwise_weight", PAIRWISE_WEIGHT),
            # architecture, defaulted from the configuration and overridden per run
            "encoder": _m.get("encoder", ENCODER),
            "attention": _m.get("attention", ATTENTION),
            "pooling": _m.get("pooling", POOLING),
            "qa_pooling": _m.get("qa_pooling", QA_POOLING),
            "passage_encoder": _m.get("passage_encoder", PASSAGE_ENCODER),
            "cross_direction": _m.get("cross_direction", CROSS_DIRECTION),
            "head": _m.get("head", HEAD),
            "match_features": _m.get("match_features", MATCH_FEATURES),
            "d_model": _m.get("d_model", D_MODEL),
            "heads": _m.get("num_heads", NUM_HEADS),
            "layers": _m.get("num_layers", NUM_LAYERS),
            "dropout": _m.get("dropout", DROPOUT),
            # anything the run overrode that is not a keyword above (the truncation variant)
            "overrides": dict(overrides) if overrides else None,
        }

    def save_run(label, model, history, model_kwargs, train_kwargs, fingerprint=None,
                 directory=ABLATION_CACHE_DIR):
        """Persist one finished run. Returns the metadata path.

        Called after every run rather than once at the end, because the point is to survive a
        crash: the most a crash can then cost is the single run that was in flight.
        """
        _meta, _weights = run_paths(label, directory)
        _write_atomically(lambda _p: torch.save(model.state_dict(), _p), _weights)
        _write_atomically(
            lambda _p: _dump_json({"label": label, "model": dict(model_kwargs),
                                   "train": dict(train_kwargs), "history": history,
                                   "fingerprint": fingerprint}, _p), _meta)
        return _meta


    def saved_run_labels(directory=ABLATION_CACHE_DIR):
        """Labels of every complete record on disk, sorted.

        Both files have to be there: metadata whose weights are missing is not a run.
        """
        if not _os.path.isdir(directory):
            return []
        _labels = []
        for _name in sorted(_os.listdir(directory)):
            if not _name.endswith(".json"):
                continue
            _meta = _os.path.join(directory, _name)
            if not _os.path.exists(_meta[:-5] + ".pt"):
                continue
            try:
                with open(_meta) as _fh:
                    _record = _json.load(_fh)
            except Exception:
                continue          # truncated by a crash before writes became atomic
            if _record.get("label"):
                _labels.append(_record["label"])
        return sorted(_labels)


    def load_saved_run(label, directory=ABLATION_CACHE_DIR):
        """(history, model, fingerprint) for one stored run, or None if it is unusable.

        The fingerprint is returned rather than checked here: only the caller knows which
        settings the record has to match.
        """
        _meta, _weights = run_paths(label, directory)
        if not (_os.path.exists(_meta) and _os.path.exists(_weights)):
            return None
        try:
            with open(_meta) as _fh:
                _record = _json.load(_fh)
            _model = create_model(**_record["model"])
            _model.load_state_dict(torch.load(_weights, map_location=DEVICE))
            _model.eval()
        except Exception as _exc:
            print(f"  [resume] {label}: stored run unreadable "
                  f"({type(_exc).__name__}: {_exc}); it will be retrained")
            return None
        return _record["history"], _model, _record.get("fingerprint")


    def load_ablation_runs(directory=ABLATION_CACHE_DIR, labels=None):
        """(experiments, models, test_loaders) for the runs the grid persisted.

        Lets a crashed or interrupted grid be picked up without re-running its cell: every
        entry is exactly what the grid would have registered. `labels` narrows it to a
        subset; by default every complete record on disk is read back.
        """
        experiments, models, test_loaders = {}, {}, {}
        _loaders = {}
        for _label in (saved_run_labels(directory) if labels is None else labels):
            _stored = load_saved_run(_label, directory)
            if _stored is None:
                continue
            _history, _model, _ = _stored
            experiments[_label] = _history
            models[_label] = _model
            _truncation = _history.get("truncation") or PASSAGE_TRUNCATION
            if _truncation not in _loaders:
                _loaders[_truncation] = (test_loader if _truncation == PASSAGE_TRUNCATION
                                         else build_loaders(_truncation)[5])
            test_loaders[_label] = _loaders[_truncation]
        print(f"[resume] read {len(experiments)} stored ablation run(s) back from "
              f"{directory}/")
        return experiments, models, test_loaders


    return (
        export_results,
        load_baselines,
        load_runs,
        load_saved_run,
        run_fingerprint,
        save_run,
        saved_run_labels,
    )


@app.cell
def baselines(
    BASELINE_BATCH_SIZE,
    BASELINE_EMBEDDINGS,
    BASELINE_MODELS,
    DEVICE,
    LLM_NAME,
    LOAD_RESULTS,
    PAD_ID,
    PROMPT_MAX_TOKENS,
    RESULTS_DIR,
    RUN_EMBEDDING_BIGRU_BASELINES,
    RUN_PRETRAINED_BASELINES,
    SEED,
    SENT_SEP,
    THRESHOLD_RULE,
    VAL_FRACTION,
    accuracy_score,
    average_precision_score,
    create_model,
    decision_scores,
    declared_threshold,
    evaluate,
    f1_score,
    flat_test,
    flat_train,
    load_baselines,
    per_question_metrics,
    roc_auc_score,
    test_loader,
    tokenize,
    torch,
    train_loader,
    train_model,
    val_loader,
    vocab,
):
    # Benchmarks for the report. Nothing here is read by the model or the training loop:
    # the pretrained embeddings/checkpoints exist only to give the reported numbers a bar.
    # numpy is imported under a private alias: the dataset-statistics cell already
    # defines a public `np`, and a second public definition of the same name is a
    # multiply-defined error. Private names are cell-local, so this stays here.
    import numpy as _np

    # Force the classic HTTP download path before huggingface_hub is first imported. The
    # Xet backend cannot fetch large weight blobs from behind this sandbox's proxy, which
    # used to make flan_t5 raise at build time and silently vanish from every table.
    import os as _os

    _os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    # Confirm the setting took effect; if huggingface_hub was already imported it is too late.
    try:
        import huggingface_hub.constants as _hub_constants

        if _hub_constants.HF_HUB_DISABLE_XET:
            print("[baselines] Xet transfer disabled: the classic HTTP download path is in use")
        else:
            print("[baselines] WARNING: huggingface_hub was imported before HF_HUB_DISABLE_XET "
                  "was set, so the Xet path is still active and flan_t5 may fail to "
                  "download their weights. Restart the kernel and re-run, or "
                  "`uv pip uninstall hf-xet`.")
    except Exception:
        pass   # hub not imported yet: the setting above takes effect when it is first imported


    def _validation_tuples():
        """Validation tuples, from the same passage-level split the dataset cell builds."""
        passages = {}
        for _i, _example in enumerate(flat_train):
            passages.setdefault(_example[0], []).append(_i)
        keys = sorted(passages)
        order = torch.randperm(len(keys),
                               generator=torch.Generator().manual_seed(SEED)).tolist()
        keys = [keys[_i] for _i in order]
        val_keys = set(keys[: int(VAL_FRACTION * len(keys))])
        return [flat_train[_i] for _k in keys if _k in val_keys for _i in passages[_k]]


    def _split_arrays(flat_data):
        """(labels, per-question ids) for one split of (passage, question, answer, label)."""
        labels, qids, seen = [], [], {}
        for passage, question, _, label in flat_data:
            seen.setdefault((passage, question), len(seen))
            qids.append(seen[(passage, question)])
            labels.append(1 if label else 0)
        return labels, qids


    def _score_metrics(labels, qids, scores, threshold, rule=THRESHOLD_RULE):
        """One baseline's metrics at the same declared cut and rule the model is judged at."""
        preds = [int(_s > threshold) for _s in decision_scores(scores, qids, rule)]
        q = per_question_metrics(qids, labels, preds, scores)
        return {
            "threshold": threshold,
            "rule": rule,
            "yes_rate": sum(preds) / max(len(preds), 1),
            "acc": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds, zero_division=0),
            "auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5,
            "ap": average_precision_score(labels, scores) if len(set(labels)) > 1 else 0.0,
            "q_em": q["q_em"],
            "q_f1": q["q_f1"],
            "q_auc": q["q_auc"],
            "q_ap": q["q_ap"],
            # Per question, so the model can be compared with a baseline *paired* on the
            # same passages (baselines are deterministic, so that test is available).
            "q_auc_by_question": q["q_auc_by_question"],
        }


    def _baseline_rule(scores):
        """The rule a scorer can be judged under.

        A scorer with one distinct value (always-yes / always-no) has no within-question
        variation to centre, so it is scored on its raw value.
        """
        return "prior_match" if len(set(scores)) < 2 else THRESHOLD_RULE

    def _baseline_threshold(labels, scores, qids, rule):
        """Same declared-on-validation cut as the model, with a guard for ties.

        Overlap/similarity scores take few distinct values, so the quantile can land on a
        tie and predict nothing; the fallback cut splits the observed range.
        """
        if len(set(scores)) < 2:
            return 0.5   # 1.0 -> yes for everything (always-yes); 0.0 -> for nothing
        threshold = declared_threshold(labels, scores, qids, rule)
        _decision = decision_scores(scores, qids, rule)
        if not any(_s > threshold for _s in _decision):
            threshold = 0.5 * (min(_decision) + max(_decision))
        return threshold


    def _lexical_overlap(flat_data):
        """Fraction of the answer's tokens present in its best-matching passage sentence."""
        scores = []
        for passage, _, answer, _ in flat_data:
            tokens = set(tokenize(answer))
            if not tokens:
                scores.append(0.0)
                continue
            best = 0.0
            for sentence in passage.split(SENT_SEP):
                best = max(best, len(tokens & set(tokenize(sentence))) / len(tokens))
            scores.append(best)
        return scores


    def _max_sentence_similarity(flat_data, embed_texts):
        """Cosine between the question+answer and its best-matching passage sentence.

        Sentences and question+answer strings are embedded once each (many tuples share a
        passage), then every tuple takes the maximum similarity over its own sentences.
        """
        sentences, qa_texts = {}, {}
        for passage, question, answer, _ in flat_data:
            for sentence in passage.split(SENT_SEP):
                sentences.setdefault(sentence, len(sentences))
            qa_texts.setdefault(f"{question} {answer}", len(qa_texts))

        def _normalise(matrix):
            return matrix / _np.clip(_np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8, None)

        sentence_vecs = _normalise(embed_texts(list(sentences)))
        qa_vecs = _normalise(embed_texts(list(qa_texts)))

        scores = []
        for passage, question, answer, _ in flat_data:
            rows = [sentences[s] for s in passage.split(SENT_SEP)]
            sims = sentence_vecs[rows] @ qa_vecs[qa_texts[f"{question} {answer}"]]
            scores.append(float(sims.max()) if len(sims) else 0.0)
        return scores


    def _keyed_vector_embedder(dataset):
        """Frozen static word vectors from a gensim keyed-vectors dataset.

        Any word-level source works (GloVe, word2vec, fastText). The returned `_embed`
        maps a list of strings to vectors by averaging the source vectors of the words
        each contains -- the same rule for a sentence and for a single word (its own
        vector, when it is in the source vocabulary). Wider coverage therefore means the
        frozen rows measure the vectors rather than their random fallback rows.
        """
        import gensim.downloader as api
        vectors = api.load(dataset)

        def _embed(texts):
            out = _np.zeros((len(texts), vectors.vector_size), dtype=_np.float32)
            for i, text in enumerate(texts):
                words = [w for w in tokenize(text) if w in vectors.key_to_index]
                if words:
                    out[i] = _np.mean([vectors[w] for w in words], axis=0)
            return out

        return _embed


    def _flan_scorer():
        """P(yes) - P(no) from a small instruction-tuned seq2seq model."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_NAME).to(DEVICE).eval()
        # The readout compares one decoder logit against another, which only means anything
        # if "yes" and "no" are each a *single* token. Checked here rather than assumed: a
        # different checkpoint could score every candidate 0.5 and measure nothing.
        _yes = tokenizer.encode("yes", add_special_tokens=False)
        _no = tokenizer.encode("no", add_special_tokens=False)
        if len(_yes) != 1 or len(_no) != 1 or _yes[0] == _no[0]:
            raise ValueError(
                f"{LLM_NAME} does not tokenise 'yes'/'no' as one token each ({_yes} / "
                f"{_no}), so the single-logit readout is undefined; score the two answer "
                f"sequences over their full token log-probabilities instead")
        yes_id, no_id = _yes[0], _no[0]

        def _prompt(passage, question, answer):
            """One yes/no prompt, with the *passage* absorbing the truncation.

            The passage is inlined, so cutting the prompt from the right at a fixed window
            removes the question, the answer being asked about and the instruction -- and the
            model is then scored on a question it cannot see, which is not a fair reading of
            the benchmark. Over-long passages are therefore truncated here, head+tail, to
            whatever is left of `PROMPT_MAX_TOKENS` after the rest of the prompt, so the prompt
            always carries its question, its answer and its instruction, and the passage keeps
            its opening and its close the way the main model's truncation does.
            """
            _tail = (f"\nQuestion: {question}\nAnswer: {answer}\n"
                     f"Is the answer supported by the passage? Answer yes or no.")
            _tail_len = len(tokenizer(_tail, add_special_tokens=False)["input_ids"])
            _budget = max(16, PROMPT_MAX_TOKENS - _tail_len - 8)   # 8 for "Passage: " + extras
            _ids = tokenizer(passage, add_special_tokens=False, truncation=True,
                             max_length=_budget)["input_ids"]
            return f"Passage: {tokenizer.decode(_ids)}{_tail}"

        def _score(flat_data):
            scores = []
            with torch.no_grad():
                for i in range(0, len(flat_data), BASELINE_BATCH_SIZE):
                    batch = flat_data[i:i + BASELINE_BATCH_SIZE]
                    prompts = [_prompt(p, q, a) for p, q, a, _ in batch]
                    enc = tokenizer(prompts, padding=True, truncation=True,
                                    max_length=PROMPT_MAX_TOKENS,
                                    return_tensors="pt").to(DEVICE)
                    decoder = torch.full((len(prompts), 1),
                                         model.config.decoder_start_token_id,
                                         dtype=torch.long, device=DEVICE)
                    logits = model(**enc, decoder_input_ids=decoder).logits[:, 0, :]
                    probs = torch.softmax(
                        torch.stack([logits[:, yes_id], logits[:, no_id]], 1), 1)
                    scores.extend(probs[:, 0].cpu().tolist())
            return scores

        return _score


    def _frozen_embedding(embed_texts, name):
        """(matrix, coverage) -- the vocab embedded by a pretrained source, and what it reached.

        `embed_texts` maps a list of strings to an (n, width) array. A word the source cannot
        embed comes back all-zero, and an all-zero row is indistinguishable from padding, so
        those rows get a small seeded random vector instead. Row `PAD_ID` is forced to zero, so
        padding still carries nothing. `coverage` says how much of the vocabulary the source
        actually reached: a frozen row built mostly from its random fallback rows measures
        almost nothing, so the table at the end of this cell prints it for every source.
        """
        _words = list(vocab)
        _vecs = _np.asarray(embed_texts(_words), dtype=_np.float32)
        _width = int(_vecs.shape[1])
        _missing = ~_np.any(_vecs != 0.0, axis=1)
        _rng = _np.random.default_rng(SEED)
        _vecs[_missing] = _rng.normal(0.0, 0.02, (int(_missing.sum()), _width)).astype(_np.float32)
        _vecs[PAD_ID] = 0.0
        _coverage = {"width": _width, "covered": int((~_missing).sum()),
                     "total": len(_words), "random": int(_missing.sum())}
        print(f"  [{name}_bigru] frozen embeddings: {_width}-d, "
              f"{_coverage['covered']}/{_coverage['total']} vocab rows initialised from "
              f"{name}, {_coverage['random']} given a random row (padding forced to zero)")
        return torch.tensor(_vecs, dtype=torch.float), _coverage


    _scorers = {
        "always_yes": lambda data: [1.0] * len(data),
        "always_no": lambda data: [0.0] * len(data),
        "lexical_overlap": _lexical_overlap,
    }
    # A baseline that raises here is *recorded*, not merely printed: a missing table row
    # must never read as a baseline that was beaten.
    baseline_errors = {}

    # The embedders are kept, not just their similarity scorers: the frozen-embedding
    # BiGRU baselines below reuse the same vectors through a trained readout.
    _embedders = {}
    baseline_results = {}
    # What share of this notebook's vocabulary each word-vector source actually reaches;
    # filled by `_frozen_embedding` and printed as a table at the end of this cell. Empty
    # on the LOAD_RESULTS path, where the baselines come from an export instead.
    embedding_coverage = {}

    if LOAD_RESULTS:
        # Read the baselines back from an export instead of paying for them again -- the
        # trained BiGRU baselines are the slow part, so this is the one that matters most.
        baseline_results, baseline_errors = load_baselines(RESULTS_DIR)
        print(f"[baselines] LOAD_RESULTS is on: read {len(baseline_results)} baseline(s) "
              f"from {RESULTS_DIR}/ instead of rebuilding them.")
    else:
        if RUN_PRETRAINED_BASELINES:
            # One word-vector embedder per entry in BASELINE_EMBEDDINGS: a similarity row
            # under its own name, and (below) a frozen-embedding BiGRU under `<name>_bigru`.
            # A source that will not load is recorded in `baseline_errors` and skipped like
            # any other baseline, so one failure cannot take the rest of the table with it.
            for _name, _dataset in BASELINE_EMBEDDINGS.items():
                if _name in BASELINE_MODELS:
                    try:
                        _embedders[_name] = _keyed_vector_embedder(_dataset)
                        _scorers[_name] = (lambda embed: lambda data: _max_sentence_similarity(data, embed))(_embedders[_name])
                    except Exception as _exc:
                        baseline_errors[_name] = f"{type(_exc).__name__}: {_exc}"
                        print(f"[baselines] skipping {_name}: {baseline_errors[_name]}")
            if "flan_t5" in BASELINE_MODELS:
                try:
                    _scorers["flan_t5"] = _flan_scorer()
                except Exception as _exc:
                    baseline_errors["flan_t5"] = f"{type(_exc).__name__}: {_exc}"
                    print(f"[baselines] skipping flan_t5: {baseline_errors['flan_t5']}")

        if baseline_errors:
            print("=" * 88)
            print("BASELINES THAT DID NOT RUN -- every table below is missing these rows:")
            for _name, _err in baseline_errors.items():
                print(f"  {_name:<10} {_err}")
            print("  A baseline that is absent is not a baseline that was beaten.")
            print("=" * 88)
        else:
            print(f"[baselines] built: {', '.join(_scorers)}")

        _val_tuples = _validation_tuples()
        baseline_results = {}
        for _name, _scorer in _scorers.items():
            _val_labels, _val_qids = _split_arrays(_val_tuples)
            _test_labels, _test_qids = _split_arrays(flat_test)
            _val_scores = _scorer(_val_tuples)
            _rule = _baseline_rule(_val_scores)
            _threshold = _baseline_threshold(_val_labels, _val_scores, _val_qids, _rule)
            baseline_results[_name] = {
                "val": _score_metrics(_val_labels, _val_qids, _val_scores, _threshold, _rule),
                "test": _score_metrics(_test_labels, _test_qids, _scorer(flat_test),
                                       _threshold, _rule),
            }
            print(f"  {_name:<16} thr={_threshold:>7.4f} ({_rule:<13})  "
                  f"val qF1={baseline_results[_name]['val']['q_f1']:.4f}  "
                  f"val qAUC={baseline_results[_name]['val']['q_auc']:.4f}  "
                  f"test qF1={baseline_results[_name]['test']['q_f1']:.4f}  "
                  f"test qAUC={baseline_results[_name]['test']['q_auc']:.4f}")
        print("Baselines:", ", ".join(baseline_results))

        # --- Frozen-embedding BiGRU baselines -----------------------------------------
        # Same architecture as the main model with the embedding table filled from a pretrained
        # source and frozen: everything above the embedding layer is learned on MultiRC, the
        # vectors are not. `d_model` follows the vectors' width (300 for both sources here), so
        # the frozen rows match *each other* but are wider than the 128-d main model: a gap
        # between a frozen row and the main model mixes vector quality with capacity, a gap
        # between the two frozen rows does not.
        if RUN_PRETRAINED_BASELINES and RUN_EMBEDDING_BIGRU_BASELINES:
            import time as _time
            if not _embedders:
                print("[baselines] no frozen-embedding BiGRU row: no word-vector "
                      "embedder built")
            for _name in list(_embedders):   # one row per embedder that did build
                _bname = f"{_name}_bigru"
                try:
                    _matrix, _coverage = _frozen_embedding(_embedders[_name], _name)
                    embedding_coverage[_name] = _coverage
                    _bmodel = create_model(d_model=_matrix.size(1))
                    with torch.no_grad():
                        _bmodel.embedding.weight.copy_(_matrix)
                    _bmodel.embedding.weight.requires_grad_(False)
                    _t0 = _time.time()
                    train_model(_bmodel, train_loader, val_loader, label=_bname)
                    _bval = evaluate(_bmodel, val_loader, rule="prior_match")
                    _bthr = declared_threshold(_bval["labels"], _bval["scores"], _bval["qids"])
                    baseline_results[_bname] = {
                        "val": evaluate(_bmodel, val_loader, threshold=_bthr),
                        "test": evaluate(_bmodel, test_loader, threshold=_bthr),
                    }
                    baseline_results[_bname]["val"]["threshold"] = _bthr
                    baseline_results[_bname]["test"]["threshold"] = _bthr
                    _btest = baseline_results[_bname]["test"]
                    print(f"  {_bname:<16} thr={_bthr:>7.4f} ({THRESHOLD_RULE:<13})  "
                          f"val qF1={baseline_results[_bname]['val']['q_f1']:.4f}  "
                          f"val qAUC={baseline_results[_bname]['val']['q_auc']:.4f}  "
                          f"test qF1={_btest['q_f1']:.4f}  test qAUC={_btest['q_auc']:.4f}  "
                          f"({_time.time() - _t0:.0f}s)")
                except Exception as _exc:
                    baseline_errors[_bname] = f"{type(_exc).__name__}: {_exc}"
                    print(f"[baselines] skipping {_bname}: {baseline_errors[_bname]}")

        if embedding_coverage:
            print("Word-vector coverage of this notebook's vocabulary (rows with no source "
                  "vector get a random one):")
            for _name, _cov in embedding_coverage.items():
                print(f"  {_name:<10} {_cov['width']:>4}-d  {_cov['covered']:,}/"
                      f"{_cov['total']:,} = {_cov['covered'] / max(_cov['total'], 1):.1%} "
                      f"from {_name}, {_cov['random']:,} random row(s)")
        print("Provenance: always_yes / always_no and lexical_overlap are training-free rules; "
              "<name> is a frozen word-vector similarity row and <name>_bigru *trains* the main "
              "architecture with those vectors held fixed; flan_t5 is a zero-shot prompted "
              "seq2seq model -- no gradient step and no fine-tuning. None of them is read by "
              "the main model.")
        print("Baselines (incl. frozen-embedding BiGRU):", ", ".join(baseline_results))
    return baseline_errors, baseline_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 3.Testing and Evaluation
    (You can add as many code blocks and text blocks as you need. However, YOU SHOULD NOT MODIFY the section title)
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt


    return (plt,)


@app.cell
def _(
    ATTENTION,
    EARLY_STOPPING,
    EARLY_STOP_METRIC,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    ENCODER,
    HEAD,
    LOAD_RESULTS,
    LOSS,
    NUM_EPOCHS,
    PAIRWISE_WEIGHT,
    PASSAGE_TRUNCATION,
    POOLING,
    RESULTS_DIR,
    SEED,
    THRESHOLD_METRIC,
    accuracy_score,
    create_model,
    f1_score,
    load_runs,
    per_question_metrics,
    run_label,
    test_loader,
    train_loader,
    train_model,
    val_dataset,
    val_loader,
    variant_tag,
):
    # Train the configured model (ENCODER / ATTENTION / POOLING in the hyperparameters cell).
    experiments = {}
    models = {}
    # Per-run test loader: a run whose PASSAGE_TRUNCATION differs must be evaluated on its
    # own encoding, so the loader is registered alongside the model.
    test_loaders = {}

    if LOAD_RESULTS:
        # Read the export instead of retraining: the study is deterministic (one SEED
        # drives the split, the initialisation and the batch order), so this is the same
        # result section 3 would produce. The plots, the sample-answer cell, the attention
        # figures and the summary all read `experiments`/`models`, so they work off it
        # unchanged.
        experiments, models, test_loaders = load_runs(RESULTS_DIR)
        print(f"LOAD_RESULTS is on: read {len(experiments)} run(s) and {len(models)} "
              f"model(s) from {RESULTS_DIR}/ instead of training -- nothing was trained "
              f"in this cell.")
    else:

        # Labelled with its seed *and* its head/objective tag, so the section-4 grids recognise
        # this run as already done -- and a run trained differently is never pooled with it.
        _config_label = run_label(ENCODER, ATTENTION, POOLING, SEED, variant=variant_tag())
        print(f"Training {_config_label}: encoder={ENCODER}, attention={ATTENTION}, "
              f"pooling={POOLING}, head={HEAD}, loss={LOSS}"
              + (f" (pairwise weight {PAIRWISE_WEIGHT})" if LOSS == "within_question" else ""))
        model = create_model()
        history = train_model(model, train_loader, val_loader, label=_config_label)
        history["truncation"] = PASSAGE_TRUNCATION
        history["config"] = run_label(ENCODER, ATTENTION, POOLING, variant=variant_tag())
        experiments[_config_label] = history
        models[_config_label] = model
        test_loaders[_config_label] = test_loader

        print(f"\nBest epoch {history['best_epoch']} of {history['epochs_run']} trained "
              f"(selected on {EARLY_STOP_METRIC}, min_delta {EARLY_STOP_MIN_DELTA}): "
              f"AP={history['best_val_ap']:.4f} "
              f"q_auc={history['best_val_q_auc']:.4f} "
              f"val_q_f1={history['best_val_q_f1']:.4f} "
              f"val_auc={history['best_val_auc']:.4f}")

        # Who ended the run? A checkpoint kept because the budget ran out is not a selected one.
        if not EARLY_STOPPING:
            print(f"Early stopping: disabled -- the run trained its whole budget "
                  f"({NUM_EPOCHS} epochs) and kept its best {EARLY_STOP_METRIC} epoch.")
        elif history["stopped_epoch"] is None:
            print(f"Early stopping: NEVER FIRED -- the run used the whole NUM_EPOCHS budget "
                  f"({NUM_EPOCHS} epochs), so the kept epoch sits at the edge of the budget "
                  f"rather than being chosen by the rule. Check the val_q_auc curve.")
        else:
            print(f"Early stopping: fired at epoch {history['stopped_epoch']} "
                  f"(patience {EARLY_STOP_PATIENCE} on {EARLY_STOP_METRIC}); kept epoch "
                  f"{history['best_epoch']}.")

        _val_labels = [int(_row[2]) for _row in val_dataset.items]
        _val_qids = [int(_row[3]) for _row in val_dataset.items]
        if history["threshold_rule"] is None:
            print("Decision cut: 0.50 (TUNE_THRESHOLD is off -- no cut was declared).")
        else:
            print(f"Decision cut: {history['threshold']:.4f}, declared by "
                  f"THRESHOLD_RULE={history['threshold_rule']} (not fitted). It predicts "
                  f"yes for {history['threshold_yes_rate']:.1%} of validation answers, "
                  f"against a gold positive rate of "
                  f"{sum(_val_labels) / len(_val_labels):.1%}; val q_f1 at the cut = "
                  f"{history.get('threshold_q_f1', float('nan')):.4f}.")
        if history.get("sweep_threshold") is not None:
            print(f"  (the {THRESHOLD_METRIC}-maximising sweep it replaced would have "
                  f"chosen {history['sweep_threshold']:.2f} -- yes for "
                  f"{history['sweep_yes_rate']:.1%} of answers. Diagnostic only.)")

        # No-model reference points on validation: predicting yes for every answer already
        # scores ~0.60 q_f1 here, so a "tuned" q_f1 of ~0.61 is not evidence of a model.
        print("Reference points on validation, with no model at all:")
        for _name, _pred in (("always-yes", 1), ("always-no", 0)):
            _preds = [_pred] * len(_val_labels)
            _q = per_question_metrics(_val_qids, _val_labels, _preds)
            print(f"  {_name:<11} F1={f1_score(_val_labels, _preds, zero_division=0):.4f}"
                  f"    q_em={_q['q_em']:.4f}    q_f1={_q['q_f1']:.4f}"
                  f"    accuracy={accuracy_score(_val_labels, _preds):.4f}")
    return experiments, models, test_loaders


@app.cell
def _(
    DEVICE,
    EARLY_STOP_METRIC,
    N_SAMPLE_QUESTIONS,
    PASSAGE_TRUNCATION,
    SAMPLE_SEED,
    SENT_SEP,
    THRESHOLD_RULE,
    decision_scores,
    encode_passage,
    encode_qa,
    experiments,
    flat_test,
    models,
    pad_sequences,
    torch,
):
    # Score every candidate answer for a few sample test questions.
    import random

    def _run_rank(_hist):
        """Rank runs on the metric they were selected on.

        Runs from before the within-question metric carry no `best_val_q_auc`, so they
        fall back to val AP rather than being ranked as zero.
        """
        return (_hist.get(f"best_{EARLY_STOP_METRIC}") or _hist.get("best_val_ap") or 0.0)


    _run_name = max(experiments, key=lambda _k: _run_rank(experiments[_k]))
    _model = models[_run_name]
    _threshold = experiments[_run_name].get("threshold", 0.5)
    # The run's own decision rule, so the printed predictions are the ones its metrics
    # describe: under `within_question` a candidate is judged against its own question.
    _rule = experiments[_run_name].get("threshold_rule") or THRESHOLD_RULE
    _truncation = experiments[_run_name].get("truncation") or PASSAGE_TRUNCATION

    _grouped = {}
    for _passage, _question, _answer, _label in flat_test:
        _grouped.setdefault((_passage, _question), []).append((_answer, bool(_label)))

    _sample_keys = random.Random(SAMPLE_SEED).sample(
        list(_grouped), min(N_SAMPLE_QUESTIONS, len(_grouped)))

    _model.eval()
    sample_results = []
    with torch.no_grad():
        for _passage, _question in _sample_keys:
            _answers = _grouped[(_passage, _question)]
            _passage_ids = pad_sequences(
                [encode_passage(_passage, _truncation) for _a, _ in _answers]).to(DEVICE)
            _qa_ids = pad_sequences([encode_qa(_question, _a) for _a, _ in _answers]).to(DEVICE)
            _scores = torch.sigmoid(_model(_passage_ids, _qa_ids)).cpu().tolist()
            # One question's candidates only, so `decision_scores` centres within that group.
            _preds = [int(_s > _threshold) for _s in
                      decision_scores(_scores, [0] * len(_scores), _rule)]
            sample_results.append({
                "question": _question,
                "passage": _passage,
                "answers": [
                    {"answer": _a, "score": _s, "predicted": _p, "gold": _g,
                     "correct": _p == _g}
                    for (_a, _g), _s, _p in zip(_answers, _scores, _preds)
                ],
            })

    _scored = _correct = _questions_right = 0
    print("=" * 72)
    print(f"Sample answers -- {_run_name} "
          f"({EARLY_STOP_METRIC}={_run_rank(experiments[_run_name]):.4f}, "
          f"cut {_threshold:.3f} on {_rule} scores, truncation {_truncation})")
    print("=" * 72)
    for _result in sample_results:
        _snippet = _result["passage"][:150].replace(SENT_SEP, " ")
        _snippet += "..." if len(_result["passage"]) > 150 else ""
        print(f"\nQ: {_result['question']}")
        print(f"   Passage: {_snippet}")
        for _row in _result["answers"]:
            _scored += 1
            _correct += _row["correct"]
            _answer = _row["answer"][:49] + "..." if len(_row["answer"]) > 52 else _row["answer"]
            _pred = "yes" if _row["predicted"] else "no "
            _gold = "yes" if _row["gold"] else "no "
            _mark = "ok" if _row["correct"] else "--"
            print(f"   {_row['score']:>7.3f}   pred={_pred}   gold={_gold}   {_mark}   {_answer}")
        _all_right = all(_row["correct"] for _row in _result["answers"])
        _questions_right += _all_right
        print(f"   -> question {'correct' if _all_right else 'incorrect'} "
              f"(per-question rule: every option must be right)")
    print("\n" + "=" * 72)
    print(f"Sample answer accuracy: {_correct}/{_scored} = {_correct / max(_scored, 1):.1%}")
    print(f"Sample per-question EM: {_questions_right}/{len(sample_results)} = "
          f"{_questions_right / max(len(sample_results), 1):.1%}")
    print("=" * 72)
    return (random,)


@app.cell
def _(
    EARLY_STOP_METRIC,
    PASSAGE_TRUNCATION,
    THRESHOLD_RULE,
    accuracy_score,
    evaluate,
    experiments,
    f1_score,
    models,
    per_question_metrics,
    plt,
    test_loader,
    test_loaders,
):
    # Six panels: two loss curves, then the ranking metrics the checkpoint is selected on.
    # The star marks the kept checkpoint, so a star at epoch 1-3 on the AP panel with a
    # later peak on q_auc is the *metric* choosing the wrong epoch, not the model failing.
    _fig, _axes = plt.subplots(2, 3, figsize=(24, 9))
    _panels = [
        (_axes[0][0], "train_loss", "Training loss", "Loss", False),
        (_axes[0][1], "val_loss", "Validation loss", "Loss", False),
        (_axes[0][2], "val_ap", "Validation PR-AUC (answer-level)", "AP", False),
        (_axes[1][0], "val_q_auc", "Validation within-question AUC", "ROC-AUC", True),
        (_axes[1][1], "val_q_ap", "Validation within-question AP", "AP", False),
        (_axes[1][2], "val_q_f1", "Validation per-question F1 (at 0.5, not the declared cut)",
         "F1", False),
    ]
    for _ax, _key, _title, _ylabel, _is_auc in _panels:
        _plotted = 0
        for _label, _hist in experiments.items():
            if _key not in _hist:
                continue  # run trained before this metric was recorded
            _ax.plot(_hist[_key], label=_label)
            if _key == EARLY_STOP_METRIC:
                _best = _hist["best_epoch"] - 1  # star = the checkpoint that was kept
                _ax.plot([_best], [_hist[_key][_best]], marker="*", markersize=16,
                         color="black")
            _plotted += 1
        _ax.set_title(f"{_title}   [n={_plotted} runs]")
        _ax.set_xlabel("Epoch")
        _ax.set_ylabel(_ylabel)
        if _is_auc:
            _ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)  # chance
        if _plotted:
            _ax.legend(fontsize=7)
        else:
            _ax.text(0.5, 0.5, "not recorded\n(this run predates the metric)",
                     ha="center", va="center", transform=_ax.transAxes, fontsize=11)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 140)
    print("Held-out test metrics, using each run's best-validation checkpoint")
    print("=" * 140)
    print(f"{'Experiment':<26}{'trunc':>11}{'sel':>8}{'thr':>7}{'yes%':>7}{'AUC':>8}"
          f"{'AP':>8}{'qAUC':>8}{'qAP':>8}{'F1@.50':>9}{'F1@cut':>9}{'qEM':>8}{'qF1':>8}")
    print("-" * 140)
    _reference = None  # labels/qids for the no-model rows (identical for every run)
    for _label, _hist in experiments.items():
        _thr = _hist.get("threshold", 0.5)
        # Each run is scored on its own encoding and its own declared rule.
        _metrics = evaluate(models[_label], test_loaders.get(_label, test_loader),
                            threshold=_thr,
                            rule=_hist.get("threshold_rule") or THRESHOLD_RULE)
        if _reference is None:
            _reference = _metrics
        _f1_half = f1_score(_metrics["labels"],
                            [int(_s > 0.5) for _s in _metrics["scores"]],
                            zero_division=0)
        print(f"{_label:<26}{str(_hist.get('truncation') or PASSAGE_TRUNCATION):>11}"
              f"{(_hist.get(f'best_{EARLY_STOP_METRIC}') or _hist['best_score']):>8.4f}"
              f"{_thr:>7.3f}"
              f"{_metrics['yes_rate'] * 100:>6.1f}%"
              f"{_metrics['auc']:>8.4f}{_metrics['ap']:>8.4f}"
              f"{_metrics['q_auc']:>8.4f}{_metrics['q_ap']:>8.4f}"
              f"{_f1_half:>9.4f}"
              f"{_metrics['f1']:>9.4f}{_metrics['q_em']:>8.4f}{_metrics['q_f1']:>8.4f}")
    print("=" * 140)
    print("Reference points on the same test split, with no model at all. Read qF1 next to "
          "the yes-rate: the F1 columns are dominated by whether the *number* of yes "
          "predictions matches the number of positives, and always-yes matches it by "
          "construction -- qEM is the column a trivial rule cannot game (~0.003):")
    for _name, _pred in (("always-yes", 1), ("always-no", 0)):
        _preds = [_pred] * len(_reference["labels"])
        _q = per_question_metrics(_reference["qids"], _reference["labels"], _preds)
        print(f"  {_name:<11} "
              f"F1={f1_score(_reference['labels'], _preds, zero_division=0):.4f}"
              f"    qEM={_q['q_em']:.4f}    qF1={_q['q_f1']:.4f}"
              f"    accuracy={accuracy_score(_reference['labels'], _preds):.4f}")
    print(f"  (this split: {sum(_reference['labels']) / len(_reference['labels']):.1%} "
          f"positive, {_reference['n_questions']} questions)")
    print("=" * 126)
    print(f"sel = validation {EARLY_STOP_METRIC} used to select the checkpoint; "
          f"'thr' = decision cut declared on validation ({THRESHOLD_RULE}), not fitted.")
    print("qEM/qF1 = MultiRC's per-question metrics (a question is correct only if every "
          "candidate answer is).")
    print("A row whose qF1 is at or below the always-yes reference has no usable decision "
          "rule on this split, whatever its validation number said.")
    return


@app.cell
def _(
    average_precision_score,
    decision_scores,
    declared_threshold,
    evaluate,
    experiments,
    models,
    np,
    per_question_metrics,
    plt,
    roc_auc_score,
    val_loader,
):
    # Diagnostics: which differences between configurations are real, and which are artefacts.
    # Answer-level AP hides the per-question quantity the task scores; the panels below lead
    # with the within-question metrics and put the decoders side by side.
    def _group(_qids):
        """{question id: indices}, so a question's candidates can be read together."""
        _out = {}
        for _i, _q in enumerate(_qids):
            _out.setdefault(_q, []).append(_i)
        return _out


    def _within_question(_labels, _scores, _qids):
        """(per-question AUCs, per-question APs) over the questions that have both labels.

        The comparisons are only ever between candidate answers of the same question.
        """
        _aucs, _aps = [], []
        for _rows in _group(_qids).values():
            _g = [_labels[_i] for _i in _rows]
            if len(set(_g)) < 2:
                continue  # a question with one label has no within-question comparison
            _s = [_scores[_i] for _i in _rows]
            _aucs.append(roc_auc_score(_g, _s))
            _aps.append(average_precision_score(_g, _s))
        return (_aucs, _aps)


    def _decoder_table(_labels, _scores, _qids):
        """q_f1/q_em/yes-rate for each declared decoder, plus the oracle-count ceiling.

        `global cut` and `within-question cut` are both legitimate. `oracle count` is not a
        method -- it needs the true positives per question -- but it is the best any
        thresholding rule could do, so it separates a ranking problem from a calibration one.
        """
        _out = {}
        for _name, _rule in (("global cut", "prior_match"),
                             ("within-question cut", "within_question")):
            _thr = declared_threshold(_labels, _scores, _qids, rule=_rule)
            _preds = [int(_s > _thr)
                      for _s in decision_scores(_scores, _qids, _rule)]
            _m = per_question_metrics(_qids, _labels, _preds)
            _out[_name] = (_m["q_f1"], _m["q_em"], sum(_preds) / max(len(_preds), 1))
        _prior = float(sum(_labels)) / max(len(_labels), 1)
        _preds = [0] * len(_labels)
        for _rows in _group(_qids).values():
            _k = int(round(_prior * len(_rows)))
            for _i in sorted(_rows, key=lambda _j: -_scores[_j])[:_k]:
                _preds[_i] = 1
        _m = per_question_metrics(_qids, _labels, _preds)
        _out["oracle count"] = (_m["q_f1"], _m["q_em"], sum(_preds) / max(len(_preds), 1))
        _m = per_question_metrics(_qids, _labels, [1] * len(_labels))
        _out["always yes"] = (_m["q_f1"], _m["q_em"], 1.0)
        return _out


    def _cut_extremes(_labels, _scores, _qids):
        """Share of questions a globally-declared cut drives to all-no / all-yes."""
        _thr = declared_threshold(_labels, _scores, _qids, rule="prior_match")
        _none = _all = 0
        _groups = _group(_qids)
        for _rows in _groups.values():
            _rate = sum(1 for _i in _rows if _scores[_i] > _thr) / len(_rows)
            _none += _rate == 0.0
            _all += _rate == 1.0
        return _none / max(len(_groups), 1), _all / max(len(_groups), 1)


    _runs = []
    for _label in sorted(experiments):
        if _label not in models:
            continue
        if experiments[_label].get("kind") == "screen":
            continue   # a shortened-budget screen is not a result to diagnose
        _metrics = evaluate(models[_label], val_loader, rule="prior_match")
        _labels, _scores, _qids = _metrics["labels"], _metrics["scores"], _metrics["qids"]
        _aucs, _aps = _within_question(_labels, _scores, _qids)
        _runs.append({
            "label": _label,
            "ap": _metrics["ap"],
            "auc": _metrics["auc"],
            "q_auc": float(np.mean(_aucs)) if _aucs else float("nan"),
            "q_ap": float(np.mean(_aps)) if _aps else float("nan"),
            "per_q_auc": _aucs,
            "decoders": _decoder_table(_labels, _scores, _qids),
            "extremes": _cut_extremes(_labels, _scores, _qids),
            "labels": _labels,
            "scores": _scores,
            "qids": _qids,
        })

    if not _runs:
        raise RuntimeError(
            "No trained runs are in memory, so there is nothing to diagnose. Train the "
            "configured model in section 3 (or the grid below) first; this cell reads "
            "`experiments` and `models`.")

    print("=" * 134)
    print("Validation: the selection metric, next to the decoders that turn a ranking into "
          "a decision")
    print("=" * 134)
    print(f"{'run':<30}{'AP':>7}{'AUC':>7}{'qAUC':>8}{'qAP':>7}  "
          f"{'qF1[global]':>11}{'qF1[within]':>12}{'qF1[oracle]':>12}{'qF1[alwaysY]':>12}  "
          f"{'qEM[within]':>11}{'no%':>6}{'yes%':>6}")
    print("-" * 134)
    for _r in _runs:
        _d = _r["decoders"]
        print(f"{_r['label']:<30}{_r['ap']:>7.4f}{_r['auc']:>7.4f}{_r['q_auc']:>8.4f}"
              f"{_r['q_ap']:>7.4f}  "
              f"{_d['global cut'][0]:>11.4f}{_d['within-question cut'][0]:>12.4f}"
              f"{_d['oracle count'][0]:>12.4f}{_d['always yes'][0]:>12.4f}  "
              f"{_d['within-question cut'][1]:>11.4f}"
              f"{_r['extremes'][0] * 100:>5.0f}%{_r['extremes'][1] * 100:>5.0f}%")
    print("-" * 134)
    print("qAUC/qAP     mean within-question AUC / AP over questions with both labels.")
    print("qF1[oracle]  q_f1 with a perfect per-question count (a CEILING, not a method);")
    print("             qF1[alwaysY] is the trivial reference.")
    print("no%/yes%     share of questions one global cut pushes to no-for-all / yes-for-all.")
    print()

    _fig, _axes = plt.subplots(2, 3, figsize=(23, 12))
    _fig.suptitle("Why the configurations looked identical: level vs ranking, "
                  "and ranking vs count", fontsize=15, fontweight="bold")

    # 1. the selection metric vs the decision-relevant metric, per run
    _ax = _axes[0][0]
    ax_ap = [_r["ap"] for _r in _runs]
    ax_q = [_r["q_auc"] for _r in _runs]
    _ax.scatter(ax_ap, ax_q, s=70, c="#1565C0", edgecolor="black", zorder=3)
    for _r in _runs:
        _ax.annotate(_r["label"].split("/")[-1], (_r["ap"], _r["q_auc"]), fontsize=7,
                     xytext=(4, 4), textcoords="offset points")
    _ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    _ax.set_xlabel("answer-level AP (the old selection metric)")
    _ax.set_ylabel("within-question AUC (what q_em/q_f1 depend on)")
    _ax.set_title("The old selection metric is not the decision metric\n"
                  "(points at the same AP differ in qAUC by more than any AP gap)")

    # 2. the distribution behind a mean within-question AUC
    _ax = _axes[0][1]
    for _r, _colour in zip(_runs[:6], plt.cm.tab10.colors):
        _ax.hist(_r["per_q_auc"], bins=np.linspace(0, 1, 21), histtype="step", linewidth=2,
                 color=_colour, label=f"{_r['label'].split('/')[-1]} "
                                      f"({_r['q_auc']:.3f})")
    _ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="chance")
    _ax.set_xlabel("per-question AUC")
    _ax.set_ylabel("questions")
    _ax.set_title("Per-question AUC for the first runs\n(mass right of 0.5 is the signal "
                  "there is)")
    _ax.legend(fontsize=7)

    # 3. the level problem, straight up
    _ax = _axes[0][2]
    _blob = _runs[0]
    _thr = declared_threshold(_blob["labels"], _blob["scores"], _blob["qids"],
                              rule="prior_match")
    _rates = [sum(1 for _i in _rows if _blob["scores"][_i] > _thr) / len(_rows)
              for _rows in _group(_blob["qids"]).values()]
    _ax.hist(_rates, bins=np.linspace(0, 1, 21), color="#C62828", edgecolor="black",
             linewidth=0.4)
    _ax.axvline(sum(_blob["labels"]) / len(_blob["labels"]), color="black", linestyle="--",
                linewidth=1.5, label="gold positive rate")
    _ax.set_xlabel("share of a question's answers the global cut calls yes")
    _ax.set_ylabel("questions")
    _ax.set_title(f"One global cut, per question ({_blob['label'].split('/')[-1]})\n"
                  f"{_blob['extremes'][0]:.0%} of questions get no-for-all, "
                  f"{_blob['extremes'][1]:.0%} get yes-for-all")
    _ax.legend(fontsize=8)

    # 4. the score level against the question's own positive rate
    _ax = _axes[1][0]
    _q_mean = [np.mean([_blob["scores"][_i] for _i in _rows])
               for _rows in _group(_blob["qids"]).values()]
    _q_rate = [np.mean([_blob["labels"][_i] for _i in _rows])
               for _rows in _group(_blob["qids"]).values()]
    _ax.scatter(_q_rate, _q_mean, s=16, alpha=0.55, c="#2E7D32", edgecolor="none")
    _ax.set_xlabel("question's gold positive rate")
    _ax.set_ylabel("question's mean model score")
    _ax.set_title("The score level is question-specific\n(correlation "
                  f"{np.corrcoef(_q_rate, _q_mean)[0, 1]:.2f}: a global cut cannot be right "
                  "for all)")

    # 5. what a cut has to work with
    _ax = _axes[1][1]
    _ax.hist([_blob["scores"][_i] for _i in range(len(_blob["labels"]))
              if _blob["labels"][_i] == 1], bins=40, alpha=0.65, color="#EF6C00",
             label="gold positive", density=True)
    _ax.hist([_blob["scores"][_i] for _i in range(len(_blob["labels"]))
              if _blob["labels"][_i] == 0], bins=40, alpha=0.55, color="#37474F",
             label="gold negative", density=True)
    _ax.set_xlabel("raw model score")
    _ax.set_ylabel("density")
    _ax.set_title("Raw score overlap\n(no cut separates these; the ranking inside a "
                  "question is what is left)")
    _ax.legend(fontsize=8)

    # 6. the size of the decoder effect, per run
    _ax = _axes[1][2]
    _x = np.arange(len(_runs))
    _ax.bar(_x - 0.21, [_r["decoders"]["global cut"][0] for _r in _runs], 0.42,
            label="global cut", color="#EF6C00", edgecolor="black", linewidth=0.4)
    _ax.bar(_x + 0.21, [_r["decoders"]["within-question cut"][0] for _r in _runs], 0.42,
            label="within-question cut", color="#1565C0", edgecolor="black", linewidth=0.4)
    _ax.axhline(_runs[0]["decoders"]["always yes"][0], color="black", linestyle="--",
                linewidth=1.5, label="always yes (degenerate)")
    _ax.set_xticks(_x)
    _ax.set_xticklabels([_r["label"].replace("two_tower/", "").replace("single/", "1t/")
                         for _r in _runs], rotation=90, fontsize=7)
    _ax.set_ylabel("validation q_f1")
    _ax.set_title("Same model, same scores, better decoder\n(the only change is where the "
                  "cut is applied)")
    _ax.legend(fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- Paired comparison: the sensitive way to compare two configurations ----------------
    # Comparing two means is the wrong test when the seed spread is large: pairing two
    # configurations on the same seed cancels the seed effect, so a difference that holds the
    # same sign across every shared seed is real and one that flips is noise.
    _paired = {}
    for _label, _hist in experiments.items():
        _cfg, _seed = _hist.get("config"), _hist.get("seed")
        if _cfg and _seed is not None and _hist.get("best_val_q_auc") is not None:
            _paired.setdefault(_cfg, {})[_seed] = _hist["best_val_q_auc"]

    if len(_paired) > 1:
        _best_cfg = max(_paired, key=lambda _c: np.mean(list(_paired[_c].values())))
        print(f"Paired by seed against {_best_cfg} "
              f"(mean val q_auc {np.mean(list(_paired[_best_cfg].values())):.4f}); "
              f"the seed effect cancels in the difference:")
        print(f"  {'configuration':<28}{'seeds':>6}{'val qAUC':>10}{'mean delta':>12}"
              f"{'worst':>9}{'best':>9}")
        for _cfg in sorted(_paired, key=lambda _c: -np.mean(list(_paired[_c].values()))):
            _shared = sorted(set(_paired[_best_cfg]) & set(_paired[_cfg]))
            if not _shared:
                continue
            _delta = [_paired[_cfg][_s] - _paired[_best_cfg][_s] for _s in _shared]
            _mark = "" if _cfg == _best_cfg else ("   <- consistent" if
                                                  (min(_delta) > 0 or max(_delta) < 0) else
                                                  "   <- flips sign: noise")
            print(f"  {_cfg:<28}{len(_shared):>6}"
                  f"{np.mean(list(_paired[_cfg].values())):>10.4f}"
                  f"{np.mean(_delta):>12.4f}{min(_delta):>9.4f}{max(_delta):>9.4f}{_mark}")
        print("  A configuration whose delta keeps one sign across every shared seed is "
              "separated by these runs; one that flips is not.")
    else:
        print("Paired comparison needs at least two configurations with the same seed and the "
              "within-question metric recorded (run the grid or the quick screen first).")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Qualitative analysis: where the attention actually goes

    The report asks for attention to be examined on examples the model gets right *and*
    examples it gets wrong, with the inputs and the expected answers shown so the figures can
    be interpreted. That is what the cell below does, for the best run in `experiments`:

    - the passage is split back into the sentences MultiRC supplies, and the sentences the
      dataset marks as **supporting** the question (`sentences_used`) are highlighted, so the
      attention can be read against the evidence rather than against an impression;
    - **answer-to-passage attention** is shown token by token and summed per sentence: with
      `attention="cross"` that is the cross-attention weight, and with `attention="inter"` it
      is the bilinear token-match distribution of the ESIM alignment -- in both cases it is
      where the model looked when it was judging the answer in front of it;
    - **pooling weights** are shown beside it, because what reaches the classifier is the
      pooled vector, not the attention: a model can look somewhere and then not use it.

    `SelfAttention`/`CrossAttention`/`AttentionPooling` each carry a `capture_attention` flag,
    off during training, switched on only for the forward passes in this cell. Nothing about
    the model or its numbers changes.

    Two things to look for, both of which the previous run could not show: whether the mass
    lands on the supporting sentence at all, and whether it lands there *more* on the question
    the model got right than on the one it got wrong. A model that scores 0.5 within a question
    has, by construction, an attention distribution that does not separate the correct answer
    from the incorrect ones sharing its passage -- and MultiRC's candidate answers for one
    question share the passage, so the figure is directly about that comparison.
    """)
    return


@app.cell
def _(
    DEVICE,
    EARLY_STOP_METRIC,
    N_ATTENTION_SCAN,
    PAD_ID,
    PASSAGE_MAX_TOKENS,
    SAMPLE_SEED,
    SENT_SEP,
    SEP_ID,
    THRESHOLD_RULE,
    clean_passage,
    decision_scores,
    encode_passage,
    encode_qa,
    experiments,
    flat_test,
    models,
    np,
    pad_sequences,
    plt,
    random,
    test_data,
    tokenize,
    torch,
    vocab,
):
    # Attention figures: one question the run gets right, one it does not. Read with the
    # printed sentence list: the bars are only interpretable next to the sentences, and next
    # to which of them the dataset marks as supporting.
    def _rank_key(_hist):
        return (_hist.get(f"best_{EARLY_STOP_METRIC}") or _hist.get("best_val_ap") or 0.0)


    _ranked = [k for k in experiments if k in models]
    _run_name = max(_ranked, key=lambda _k: _rank_key(experiments[_k]))
    _model = models[_run_name]
    _rule = experiments[_run_name].get("threshold_rule") or THRESHOLD_RULE
    _cut = experiments[_run_name].get("threshold", 0.5)
    _id_to_token = {_i: _t for _t, _i in vocab.items()}


    def _answer_rows(_ids):
        """Token positions of the answer: strictly between the first two `<sep>` markers."""
        _seps = [_i for _i, _t in enumerate(_ids) if _t == SEP_ID]
        if len(_seps) >= 2:
            _rows = list(range(_seps[0] + 1, _seps[1]))
            if _rows:
                return _rows
        return [_i for _i, _t in enumerate(_ids) if _t != PAD_ID]


    def _sentence_spans(_ids):
        """(start, stop) token range of each sentence of an encoded passage.

        `encode_passage` puts one `<sep>` after every sentence, so splitting on that marker
        recovers the sentence structure -- but only while the passage was not truncated.
        """
        _spans, _start = [], 0
        for _i, _t in enumerate(_ids):
            if _t == SEP_ID:
                _spans.append((_start, _i))
                _start = _i + 1
        if _start < len(_ids):
            _spans.append((_start, len(_ids)))
        return [(_s, _e) for _s, _e in _spans if _e > _s]


    def _untruncated_length(_text):
        """Token count `encode_passage` builds for `_text` before any truncation.

        `encode_passage` always caps its output at `PASSAGE_MAX_TOKENS`, so its result cannot
        show that a passage *was* truncated. Each sentence contributes its tokens plus one
        `<sep>`, exactly as `encode_passage` lays them out.
        """
        return sum(len(tokenize(_sentence)) + 1 for _sentence in _text.split(SENT_SEP))


    def _capture(_on):
        """Switch attention capture on/off for every attention module of the model."""
        for _m in _model.modules():
            if hasattr(_m, "capture_attention"):
                _m.capture_attention = _on


    def _forward(_passage, _question, _answers, _keep_attention):
        """Score one question's candidate answers, optionally keeping the attention maps.

        The maps are copied out immediately: the next forward pass overwrites `last_weights`.
        """
        _pids = pad_sequences([encode_passage(_passage) for _a, _ in _answers]).to(DEVICE)
        _qids = pad_sequences([encode_qa(_question, _a) for _a, _ in _answers]).to(DEVICE)
        # eval() matters for the maps, not just the scores: the attention modules apply
        # dropout to their weights, so in train mode the map would not sum to one.
        _model.eval()
        _capture(_keep_attention)
        try:
            with torch.no_grad():
                _logits = _model(_pids, _qids)
        finally:
            _capture(False)   # never leave it on: it would keep a map per batch during training
        _out = {"passage_ids": _pids[0].cpu().tolist(),
                "qa_ids": [_qids[_b].cpu().tolist() for _b in range(_qids.size(0))],
                "scores": torch.sigmoid(_logits).cpu().tolist()}
        _out["preds"] = [int(_s > _cut) for _s in
                         decision_scores(_out["scores"], [0] * len(_out["scores"]), _rule)]
        for _name, _attr in (("cross", "qa_to_passage"), ("inter", "inter"),
                             ("pool", "passage_pool"), ("self", "passage_self_attn")):
            _mod = getattr(_model, _attr, None)
            _w = None if _mod is None else getattr(_mod, "last_weights", None)
            _out[_name] = None if _w is None else _w.float().cpu().numpy()
        return _out


    _grouped = {}
    for _passage, _question, _answer, _label in flat_test:
        _grouped.setdefault((_passage, _question), []).append((_answer, bool(_label)))

    # The dataset's own supporting-sentence annotation: what the attention *should* find.
    _gold = {}
    for _item in test_data:
        _clean = clean_passage(_item["paragraph"]["text"])
        for _q in _item["paragraph"]["questions"]:
            _gold[(_clean, _q["question"])] = [int(_s) for _s in (_q.get("sentences_used") or [])]

    _scan = random.Random(SAMPLE_SEED).sample(sorted(_grouped),
                                              min(N_ATTENTION_SCAN, len(_grouped)))
    _picked = {}
    # Evidence collected over the *whole* scan rather than from the two examples below: for
    # every candidate answer of every scanned question, what share of that candidate's
    # answer -> passage attention lands on the sentences MultiRC marks as supporting, split by
    # whether the model got that candidate right. The examples that follow show *how* the
    # attention looked on one question each; this says whether that is how it looks in general,
    # and whether it lands on the evidence more often when the decision is right.
    _support_mass = {"correct": [], "incorrect": []}
    _support_chance = []
    for _key in _scan:
        _passage, _question = _key
        if _untruncated_length(_passage) > PASSAGE_MAX_TOKENS:
            continue   # truncation would break the sentence mapping below
        _answers = _grouped[_key]
        _res = _forward(_passage, _question, _answers, True)
        _slot = ("correct" if all(_p == _g for _p, (_a, _g) in zip(_res["preds"], _answers))
                 else "incorrect")
        _picked.setdefault(_slot, _key)

        _gold_here = set(_gold.get(_key, []))
        _spans_here = _sentence_spans(_res["passage_ids"])
        _map_here = _res["cross"] if _res["cross"] is not None else _res["inter"]
        if (_gold_here and _map_here is not None
                and len(_spans_here) == len(_passage.split(SENT_SEP))):
            # Chance is the share of the passage's sentences marked as supporting: attention
            # spread evenly scores this, and no more.
            _support_chance.append(len(_gold_here) / len(_spans_here))
            for _b, (_answer_text, _gold_flag) in enumerate(_answers):
                _weights = _map_here[_b].mean(axis=0)              # mean over heads
                _mass = _weights[_answer_rows(_res["qa_ids"][_b])].sum(axis=0)
                _mass = _mass / max(_mass.sum(), 1e-9)
                _support_mass["correct" if _res["preds"][_b] == _gold_flag
                              else "incorrect"].append(
                    float(sum(_mass[_spans_here[_g][0]:_spans_here[_g][1]].sum()
                              for _g in _gold_here if _g < len(_spans_here))))

    print(f"Attention figures -- {_run_name} "
          f"(cut {_cut:.3f} on {_rule} scores, decision by {EARLY_STOP_METRIC})")
    print(f"Scanned {len(_scan)} test questions for an example of each kind: "
          f"{sorted(_picked)}")
    if _support_chance:
        print(f"Attention on the dataset's supporting sentence(s): {len(_support_chance)} "
              f"question(s), {len(_support_mass['correct']) + len(_support_mass['incorrect'])} "
              f"candidate(s), chance {np.mean(_support_chance):.3f}, decided-correct "
              f"{np.mean(_support_mass['correct']) if _support_mass['correct'] else float('nan'):.3f}, "
              f"decided-wrong "
              f"{np.mean(_support_mass['incorrect']) if _support_mass['incorrect'] else float('nan'):.3f}\n")
    else:
        print("No scanned candidate had both an answer -> passage attention map and an "
              "untruncated sentence mapping, so there is no aggregate to show.\n")

    # The aggregate view of the two example figures that follow: does the attention land on the
    # evidence, and does it land there more when the model is right?
    if _support_chance and (_support_mass["correct"] or _support_mass["incorrect"]):
        _fig, _ax = plt.subplots(figsize=(11, 5))
        _bins = np.linspace(0, 1, 21)
        for _slot, _colour in (("correct", "#2E7D32"), ("incorrect", "#C62828")):
            _vals = _support_mass[_slot]
            if _vals:
                _ax.hist(_vals, bins=_bins, histtype="step", linewidth=2, color=_colour,
                         label=f"decided {_slot} (n={len(_vals)}, mean {np.mean(_vals):.3f})")
        _ax.axvline(float(np.mean(_support_chance)), color="black", linestyle="--",
                    linewidth=1.5,
                    label=f"chance = share of supporting sentences "
                          f"({float(np.mean(_support_chance)):.3f})")
        _ax.set_xlabel("share of the candidate's answer -> passage attention on the "
                       "supporting sentences")
        _ax.set_ylabel("candidates")
        _ax.set_title(f"Where the attention goes, over every candidate of {len(_scan)} scanned "
                      f"test questions ({_run_name})")
        _ax.legend(fontsize=9)
        plt.tight_layout()
        plt.show()
    elif _support_chance:
        print("This configuration has no answer -> passage attention to aggregate "
              "(attention='none'; the example figures below fall back to self-attention or "
              "show nothing).")

    for _slot in ("correct", "incorrect"):
        if _slot not in _picked:
            print(f"No {_slot} example among the scanned questions; raise the scan or change "
                  f"SAMPLE_SEED.")
            continue
        _passage, _question = _picked[_slot]
        _answers = _grouped[(_passage, _question)]
        _res = _forward(_passage, _question, _answers, True)
        _sentences = _passage.split(SENT_SEP)
        _spans = _sentence_spans(_res["passage_ids"])
        _gold_idx = set(_gold.get((_passage, _question), []))
        if len(_spans) != len(_sentences):
            print(f"  (note: {len(_spans)} encoded sentence spans vs {len(_sentences)} text "
                  f"sentences -- read the sentence numbering with care)")

        # Sentence-level masses: what the answer looked at, and what the pooling kept.
        _token_mass = None
        _source = "attention='none' in this configuration"
        if _res["cross"] is not None:
            _w = _res["cross"][0].mean(axis=0)          # (qa_len, passage_len), mean over heads
            _token_mass = _w[_answer_rows(_res["qa_ids"][0])].sum(axis=0)
            _source = "answer -> passage cross-attention"
        elif _res["self"] is not None:
            _token_mass = _res["self"][0].mean(axis=0).sum(axis=0)
            _source = "passage self-attention (mass received, answer 1)"
        elif _res["inter"] is not None:
            _w = _res["inter"][0].mean(axis=0)
            _token_mass = _w[_answer_rows(_res["qa_ids"][0])].sum(axis=0)
            _source = "answer -> passage inter-attention (bilinear token match)"
        if _token_mass is not None:
            _token_mass = _token_mass / max(_token_mass.sum(), 1e-9)
            _sent_mass = [_token_mass[_s:_e].sum() for _s, _e in _spans]
        else:
            _sent_mass = None
        _pool_mass = None
        if _res["pool"] is not None:
            _pool_mass = [_res["pool"][0][_s:_e].sum() for _s, _e in _spans]

        _fig, _axes = plt.subplots(1, 3, figsize=(23, 6.5))
        _fig.suptitle(f"{_slot} question -- {_run_name}\nQ: {_question}",
                      fontsize=13, fontweight="bold")

        # (1) token-level map, with the sentence and gold-evidence structure on top of it
        _ax = _axes[0]
        if _token_mass is not None and (_res["cross"] is not None or _res["inter"] is not None):
            _map = (_res["cross"] if _res["cross"] is not None else _res["inter"])[0].mean(axis=0)
            _ax.imshow(_map, aspect="auto", cmap="magma", origin="upper")
            _ax.set_yticks(range(len(_res["qa_ids"][0])))
            _ax.set_yticklabels([_id_to_token.get(int(_t), "pad")
                                 for _t in _res["qa_ids"][0]], fontsize=6)
            _ax.set_ylabel("question <sep> answer <sep> (rows)")
        else:
            _ax.imshow(np.zeros((2, len(_res["passage_ids"]))), aspect="auto", cmap="magma")
            _ax.text(0.5, 0.5, "no cross-attention in this configuration",
                     ha="center", va="center", transform=_ax.transAxes, fontsize=11)
        for _s, _e in _spans:
            _ax.axvline(_e - 0.5, color="white", linewidth=0.6, alpha=0.7)
        for _g in sorted(_gold_idx):
            if _g < len(_spans):
                _ax.axvspan(_spans[_g][0] - 0.5, _spans[_g][1] - 0.5, color="#00E676",
                            alpha=0.13, zorder=0)
        _ax.set_xticks([(_s + _e) / 2 for _s, _e in _spans])
        _ax.set_xticklabels([f"S{_i + 1}" for _i in range(len(_spans))], fontsize=6)
        _ax.set_xlabel(f"passage tokens (green = sentences the dataset marks as supporting)")
        _ax.set_title(f"Attention: {_source}")

        # (2) per sentence: what attention looked at vs what pooling kept
        _ax = _axes[1]
        _x = np.arange(len(_spans))
        if _sent_mass is not None:
            _ax.bar(_x - 0.2, _sent_mass, 0.4, label="attention mass (answer rows)",
                    color="#1565C0", edgecolor="black", linewidth=0.4)
        if _pool_mass is not None:
            _ax.bar(_x + 0.2, _pool_mass, 0.4, label="pooling weight (all tokens)",
                    color="#EF6C00", edgecolor="black", linewidth=0.4)
        for _g in sorted(_gold_idx):
            if _g < len(_spans):
                _ax.text(_g, _ax.get_ylim()[1] * 0.95, "gold", ha="center", fontsize=8,
                         color="#00701a", fontweight="bold")
        _ax.set_xticks(_x)
        _ax.set_xticklabels([f"S{_i + 1}" for _i in range(len(_spans))], fontsize=8)
        _ax.set_xlabel("passage sentence")
        _ax.set_ylabel("share of mass")
        _ax.set_title("Where it looked, and what it kept")
        _ax.legend(fontsize=8)

        # (3) what was actually judged
        _ax = _axes[2]
        _scores = _res["scores"]
        _colours = ["#2E7D32" if _p == _g else "#C62828"
                    for _p, (_a, _g) in zip(_res["preds"], _answers)]
        _ax.barh(range(len(_answers)), _scores, color=_colours, edgecolor="black",
                 linewidth=0.4)
        _ax.axvline(_cut, color="black", linestyle="--", linewidth=1.5,
                    label=f"declared cut {_cut:.3f}")
        _ax.set_yticks(range(len(_answers)))
        _ax.set_yticklabels([(_a[:44] + "...") if len(_a) > 47 else _a
                             for _a, _ in _answers], fontsize=8)
        _ax.set_xlabel("model score (centred within this question before the cut)")
        _ax.set_title("Candidates: green = predicted correctly, red = not")
        _ax.legend(fontsize=8)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.show()

        print(f"--- {_slot} question: {_question}")
        print(f"    answers: " + " | ".join(
            f"{'yes' if _p else 'no '}/{'yes' if _g else 'no '} {_s:.3f} "
            f"{(_a[:40] + '..') if len(_a) > 42 else _a}"
            for _s, _p, (_a, _g) in zip(_scores, _res["preds"], _answers)))
        for _i, (_s, _e) in enumerate(_spans):
            _mark = " <- gold" if _i in _gold_idx else ""
            _att = f"attn {_sent_mass[_i]:.3f}" if _sent_mass is not None else "attn n/a"
            _pool = f"pool {_pool_mass[_i]:.3f}" if _pool_mass is not None else "pool n/a"
            _text = _sentences[_i] if _i < len(_sentences) else ""
            print(f"    S{_i + 1:<3}{_att}  {_pool}{_mark}  "
                  f"{_text[:96]}{'...' if len(_text) > 96 else ''}")
        _att_gold = sum(_sent_mass[_i] for _i in _gold_idx if _i < len(_sent_mass)) \
            if _sent_mass is not None else float("nan")
        print(f"    -> attention mass on the {len(_gold_idx)} supporting sentence(s): "
              f"{_att_gold:.1%} (chance for {len(_gold_idx)}/{len(_spans)} sentences: "
              f"{len(_gold_idx) / max(len(_spans), 1):.1%})\n")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset Statistics

    Overview of the MultiRC dataset distribution, vocabulary, and structure.
    """)
    return


@app.cell
def _(
    Counter,
    SEED,
    VAL_FRACTION,
    flat_test,
    flat_train,
    tokenize,
    torch,
    train_data,
    vocab,
):

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

    # --- Unknown-token rate -------------------------------------------------------------
    # The vocabulary is built from the training split only, so a val/test token the training
    # text never contained is `<unk>` to the model. That is the direct cost of a word-level
    # tokenizer on MultiRC, and it is field-specific: the decisive rare token (a name, a
    # number, a date) usually sits in the answer, so the answer's rate is the one that caps
    # what the model can do. Used by the panel and the summary line in the cell below.
    #
    # The dataset cell keeps its split in private names, and private names do not cross
    # cells, so the validation tuples are rebuilt here with the same passage-level split
    # as that cell (and as the baselines cell): group the tuples by cleaned passage, sort
    # the passages, shuffle them with SEED, take the first VAL_FRACTION of them. The rates
    # below therefore describe the split the model is actually validated on, not a fresh
    # one. Checked against `val_dataset` for equality rather than assumed.
    _unk_passages = {}
    for _i, _example in enumerate(flat_train):
        _unk_passages.setdefault(_example[0], []).append(_i)
    _unk_keys = sorted(_unk_passages)
    _unk_order = torch.randperm(len(_unk_keys),
                                generator=torch.Generator().manual_seed(SEED)).tolist()
    _unk_keys = [_unk_keys[_j] for _j in _unk_order]
    _unk_val_keys = set(_unk_keys[: int(VAL_FRACTION * len(_unk_keys))])
    _unk_val_rows = [flat_train[_i] for _j in _unk_keys if _j in _unk_val_keys
                     for _i in _unk_passages[_j]]

    unk_rates = {}          # {split: {field: [unknown, total]}}
    for _split_name, _rows in (("val", _unk_val_rows), ("test", flat_test)):
        _counts = {_field: [0, 0] for _field in ("passage", "question", "answer")}
        for _passage, _question, _answer, _ in _rows:
            for _field, _text in (("passage", _passage), ("question", _question),
                                  ("answer", _answer)):
                for _token in tokenize(_text):
                    _counts[_field][1] += 1
                    if _token not in vocab:
                        _counts[_field][0] += 1
        unk_rates[_split_name] = _counts
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
        unk_rates,
        word_freq_values,
    )


@app.cell(hide_code=True)
def _(
    MIN_FREQ,
    PASSAGE_MAX_TOKENS,
    QA_MAX_TOKENS,
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
    unk_rates,
    vocab_size,
    word_freq_values,
):

    _fig, _axes = plt.subplots(2, 4, figsize=(26, 12))
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
    ax3.axvline(PASSAGE_MAX_TOKENS + QA_MAX_TOKENS, color="green", linestyle=":",
                linewidth=2, label=f"Encoder cap = {PASSAGE_MAX_TOKENS + QA_MAX_TOKENS}")
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
                label=f"min_freq={MIN_FREQ} cutoff\n(rank {cutoff_rank:,})")
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

    # 7. Unknown-token rate: what the training-only vocabulary does not cover
    ax7 = _axes[0, 3]
    _fields = ["passage", "question", "answer"]
    _x = np.arange(len(_fields))
    for _split_name, _offset, _colour in (("val", -0.2, "#1565C0"), ("test", 0.2, "#EF6C00")):
        _rates = [unk_rates[_split_name][_f][0] / max(unk_rates[_split_name][_f][1], 1) * 100
                  for _f in _fields]
        _bars = ax7.bar(_x + _offset, _rates, 0.4, color=_colour, edgecolor="black",
                        linewidth=0.4, label=_split_name)
        for _bar, _rate in zip(_bars, _rates):
            ax7.text(_bar.get_x() + _bar.get_width() / 2, _bar.get_height(),
                     f"{_rate:.1f}%", ha="center", va="bottom", fontsize=8)
    ax7.set_xticks(_x)
    ax7.set_xticklabels([_f.capitalize() for _f in _fields])
    ax7.set_ylabel("% of tokens -> <unk>")
    ax7.set_title(f"Unknown-token rate\n(vocabulary built from the training split only, "
                  f"min_freq={MIN_FREQ})")
    ax7.legend(fontsize=9)
    _axes[1, 3].axis("off")   # 7 panels in a 2x4 grid: the eighth slot stays empty


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("=" * 60)
    print("MultiRC Dataset Summary")
    print("=" * 60)
    print(f"Vocabulary size (min_freq={MIN_FREQ}):  {vocab_size:,} unique words")
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
    for _split_name in ("val", "test"):
        print(f"{'Unknown tokens (' + _split_name + '):':<30}"
              + "  ".join(f"{_field}={unk_rates[_split_name][_field][0] / max(unk_rates[_split_name][_field][1], 1):.2%}"
                          for _field in ("passage", "question", "answer")))
    print("=" * 60)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Ablation study

    Two studies sharing one component list (the `ablation_components` cell), so they cannot
    drift apart. They differ only in budget and in how many seeds they repeat:

    - **`quick_ablation`** (`QUICK_ABLATION`) -- a one-change-at-a-time **screen** on a
      shortened budget (`QUICK_ABLATION_EPOCHS`) and a single seed, for ballpark triage: which
      variants look worth a full-budget run, and which are dead on arrival. Its runs are tagged
      `kind="screen"`, so the plots show them but **no reported table, no diagnostic and no
      model-vs-baselines row does**. It is a screen, not a result.
    - **`final_ablation`** (`RUN_ABLATION_STUDY`) -- the **reported** study, at
      `ABLATION_EPOCHS` over `ABLATION_SEEDS`, covering every component.
    - **the summary table** -- aggregates the reported runs and the baselines into one table.
    - **the model-vs-baselines comparison** -- the same numbers as bars, with a
      passage-bootstrap interval and a paired test against each baseline.

    All of them register their runs in `experiments`/`models`/`test_loaders`, so the plots,
    the sample-answer cell, the attention figures and the summary table pick them up.

    ### The final study

    This is the comparison the report is built on, so it is controlled:

    - **One budget and one stopping rule.** Every configuration trains under the same
      schedule, with the same `EARLY_STOPPING` rule, and keeps its best-`EARLY_STOP_METRIC`
      epoch -- selected epoch by epoch, so the rule only removes epochs that could not have
      been selected and never changes which checkpoint is reported. The LR schedule is a pure
      function of the epoch index, so a run that stopped early still saw the same LR at the
      same epoch as one that ran on, and the comparison stays like-for-like.
    - **Repeated over seeds, seeds outermost.** `ABLATION_SEEDS` runs each configuration
      several times, so the spread *between* configurations can be read against the spread
      *within* one. The seed loop is on the outside, so the study is already comparable after
      the first pass. The summary prints the means and the seed spread, and the diagnostics
      cell prints a paired-by-seed comparison: a difference that keeps its sign across every
      shared seed is separated by these runs, one that flips is noise.
    - **Two blocks, covering every component.** The **architecture factorial** sweeps every
      `ENCODER_OPTIONS` x `ATTENTION_OPTIONS` x `POOLING_OPTIONS` architecture the encoder
      admits (with the shipped options, 6 encoder x attention combinations x 4 poolings = 24).
      Those three interact, so they are swept together as a cross product. The **component
      runs** then change exactly one thing at a time off the configured model, covering
      everything the factorial holds fixed: the head, the objective, the passage truncation,
      the passage encoder, the cross-attention direction, the question/answer pooling,
      dropout, the pairwise weight and the alignment features -- every entry of
      `COMPONENT_VARIANTS` in the `ablation_components` cell.
    - **A reference that costs nothing.** The configured model
      (`ENCODER`/`ATTENTION`/`POOLING` in the hyperparameters cell) is itself a member of the
      architecture factorial, so every component run has a full-budget, same-seed reference
      already trained. The summary and the diagnostics compare them **paired seed by seed**,
      which cancels the seed effect and is the sensitive test. Set those three to the
      factorial's best configuration once the study has run.
    - **The `"match"` guard.** The architecture axis is swept under the configured `HEAD`, and
      `HEAD` has to keep the pooled vectors in the classifier for the `POOLING` axis to be
      measurable: under `head="match"` the classifier reads *only* the alignment features, so
      every pooling setting would build the same function and the pooling axis would be
      several copies of one run. The shipped configuration is therefore `HEAD = "interaction"`,
      and the study refuses the `"match"` combination by name rather than printing a table of
      identical rows. `head="match"` is still ablated -- as a *component*, where it is the
      extreme that closes the answer-prior route by construction.

    ### Reading a component

    - **head** -- what the classifier is fed. `concat` gives it the two pooled vectors
      (`[u; v]`); `interaction` adds the comparison (`[u; v; |u-v|; u*v]`); `compare` drops the
      bare vectors and keeps only terms that contain the passage (`[|u-v|; u*v]`); and `match`
      hands it *only* the passage-conditioned alignment features, so nothing in its input is
      computable from the question and answer alone. They differ in how much of the
      answer <-> passage comparison reaches the classifier.
    - **objective** -- `bce` against `within_question`, which centres each question's candidate
      logits on their own mean and adds a pairwise ranking term (`PAIRWISE_WEIGHT`). The level
      of a question is a property of a passage the model has not seen at test time, so an
      objective that can lower itself by fitting it is spending capacity on something that
      cannot transfer.
    - **the knobs the factorial holds fixed** -- `head_only` (passage truncation),
      `passage_no_gru`, `qa_gathers_only` (cross-attention direction), `qa_pool_full`,
      `more_dropout`, `loss_no_pairwise` and `no_match_features`.

    Every run in the study registers in `experiments`/`models`/`test_loaders`, so the plots,
    the sample-answer cell, the attention figures and the summary table pick them up, and each
    run is cached to `ABLATION_CACHE_DIR` as it finishes, so an interrupted study resumes.

    `len(ENCODER_OPTIONS)`-derived architectures x `len(ABLATION_SEEDS)` runs plus
    `len(COMPONENT_VARIANTS)` x `len(ABLATION_SEEDS)` component runs, each at most
    `ABLATION_EPOCHS` epochs (early stopping usually ends them before the budget). Trim
    `ABLATION_SEEDS`, `POOLING_OPTIONS`, `ATTENTION_OPTIONS` or `COMPONENT_VARIANTS` if the
    budget is tight.
    """)
    return


@app.cell
def ablation_components(
    HEAD,
    HEAD_OPTIONS,
    LOSS,
    LOSS_OPTIONS,
    PASSAGE_TRUNCATION,
    build_loaders,
    test_loader,
    train_loader,
    val_loader,
):
    # --- Ablation components: the one-change-at-a-time variants --------------------------
    # Each entry changes exactly ONE thing about the configured model, so a difference is
    # attributable to that change. This list is the single source of truth for both studies in
    # this section, so the quick screen and the final study can never drift apart:
    #
    #   quick_ablation  (QUICK_ABLATION)      shortened budget, 1 seed    -> a screen, not a result
    #   final_ablation  (RUN_ABLATION_STUDY)  full budget x ABLATION_SEEDS -> the reported study
    #
    # The final study runs the architecture factorial (encoder x attention x pooling, which
    # interact, so they are swept as a cross product) plus every COMPONENT_VARIANTS entry.
    # GRID_AXIS_VARIANTS are screened by the quick study but not re-run at full budget: the
    # factorial covers those three axes far more thoroughly than a one-change-at-a-time walk.
    REFERENCE_VARIANT = ("reference", {})

    # The axes the architecture factorial already sweeps; the quick screen walks them singly.
    GRID_AXIS_VARIANTS = [
        ("attention_self", {"attention": "self"}),
        ("attention_inter", {"attention": "inter"}),
        ("attention_none", {"attention": "none"}),
        ("pool_mean", {"pooling": "mean"}),
        ("pool_max", {"pooling": "max"}),
        ("pool_final", {"pooling": "final"}),
        # "single" cannot do cross-/inter-attention, so this pair is the encoder compared at
        # equal attention (self) -- otherwise its effect would be confounded with attention.
        ("single_encoder", {"encoder": "single", "attention": "self"}),
    ]

    # Everything the architecture factorial holds fixed. These are the components the FINAL
    # study reports at full budget, one change at a time against the configured model -- which
    # is itself a row of the architecture factorial, so the reference is already trained and
    # the comparison is paired seed by seed.
    COMPONENT_VARIANTS = (
        # head: what the classifier is fed (every HEAD_OPTIONS value but the configured one)
        [("head_" + _h, {"head": _h}) for _h in HEAD_OPTIONS if _h != HEAD]
        # objective
        + [("loss_" + _l, {"loss": _l}) for _l in LOSS_OPTIONS if _l != LOSS]
        # the knobs the factorial never varies
        + [
            ("head_only", {"truncation": "head"}),
            ("passage_no_gru", {"passage_encoder": "none"}),
            ("qa_gathers_only", {"cross_direction": "qa_to_passage"}),
            ("qa_pool_full", {"qa_pooling": "full"}),
            ("more_dropout", {"dropout": 0.4}),
            ("loss_no_pairwise", {"pairwise_weight": 0.0}),
            ("no_match_features", {"head": "interaction", "match_features": False}),
            # Model capacity. Both levers are trained from scratch on ~22k instances and
            # the embedding table alone is vocab x d_model, so neither is an obvious win --
            # which is exactly why they are swept rather than guessed at. Note the frozen-
            # embedding baselines run at d_model=300, so a *capacity* difference against
            # them is already confounded by design (and said so in the baselines cell).
            ("d_model_256", {"d_model": 256}),
            ("num_layers_2", {"num_layers": 2}),
        ]
    )

    # The quick screen walks the reference, then the grid axes, then the components.
    SCREEN_VARIANTS = [REFERENCE_VARIANT] + GRID_AXIS_VARIANTS + COMPONENT_VARIANTS

    # Which override keys build the network, and which configure training. Anything else (the
    # passage truncation, say) is not a create_model/train_model keyword and is handled by the
    # caller -- including making sure it reaches the run's fingerprint.
    _MODEL_KEYS = ("encoder", "attention", "pooling", "qa_pooling", "passage_encoder",
                   "cross_direction", "head", "d_model", "num_heads", "num_layers",
                   "dropout", "match_features")


    def split_variant(override):
        """(model kwargs, training kwargs) for one variant's override dict."""
        model_over = {_k: _v for _k, _v in override.items() if _k in _MODEL_KEYS}
        train_over = {_k: _v for _k, _v in override.items()
                      if _k not in _MODEL_KEYS and _k != "truncation"}
        return model_over, train_over


    def variant_truncation(override):
        """The passage truncation one variant is encoded under."""
        return override.get("truncation", PASSAGE_TRUNCATION)


    def loaders_for_truncation(truncation):
        """(train, val, test) loaders for a truncation mode; the shared three where possible."""
        if truncation == PASSAGE_TRUNCATION:
            return train_loader, val_loader, test_loader
        _train_ds, _val_ds, _test_ds, _train_ld, _val_ld, _test_ld = build_loaders(truncation)
        return _train_ld, _val_ld, _test_ld


    return (
        COMPONENT_VARIANTS,
        SCREEN_VARIANTS,
        loaders_for_truncation,
        split_variant,
        variant_truncation,
    )


@app.cell
def quick_ablation(
    ABLATION_CACHE_DIR,
    EARLY_STOPPING,
    QUICK_ABLATION,
    QUICK_ABLATION_EPOCHS,
    SCREEN_VARIANTS,
    create_model,
    experiments,
    load_saved_run,
    loaders_for_truncation,
    models,
    run_fingerprint,
    save_run,
    saved_run_labels,
    split_variant,
    test_loaders,
    train_model,
    variant_truncation,
):
    # --- The quick screen (QUICK_ABLATION) -----------------------------------------------
    # A screen, not a result. One run per entry of SCREEN_VARIANTS (see the
    # `ablation_components` cell) on a shortened budget (QUICK_ABLATION_EPOCHS) and a single
    # seed: use it to see which variants deserve a full-budget run and which are dead on
    # arrival. The reported numbers come from the final study (`final_ablation`, gated on
    # RUN_ABLATION_STUDY), which runs the same components at the full budget over ABLATION_SEEDS.
    #
    # It carries its own reference row, `q/reference`: the configured model re-trained at the
    # screen budget, so each variant is compared against a model that had as many epochs
    # available as it did, rather than against section 3's full-budget run.
    #
    # Every finished variant is written to ABLATION_CACHE_DIR as it goes, so a crash costs the
    # run in flight and re-running this cell picks up where it left off. Runs are tagged
    # kind="screen" and register in experiments/models/test_loaders so the plots pick them up
    # -- but the reported tables, the diagnostics and the model-vs-baselines comparison leave
    # them out, because a shortened-budget, single-seed run is not a result. A variant that
    # changes `truncation` re-encodes through `build_loaders`, so it is scored on its own
    # test loader.
    if QUICK_ABLATION:
        def _screen_fingerprint(_model_over, _train_over, _override):
            """What a stored screen run must have been trained under to be reusable.

            The same shared list the final study fingerprints with, at this screen's budget,
            plus the raw variant dict -- so the settings a variant only implies (the passage
            truncation) are part of the check too, and neither editing a hyperparameter nor
            editing the variant list can reuse a stale run.
            """
            return run_fingerprint(QUICK_ABLATION_EPOCHS, _model_over, _train_over,
                                   overrides=_override)

        _screen_labels = [f"q/{_name}" for _name, _ in SCREEN_VARIANTS]
        _already_stored = set(saved_run_labels()) & set(_screen_labels)
        if _already_stored:
            print(f"resume: {len(_already_stored)} of {len(_screen_labels)} screen run(s) are "
                  f"already stored under {ABLATION_CACHE_DIR}/ and will be read back rather "
                  f"than retrained; a crash costs only the run in flight.")

        for _name, _override in SCREEN_VARIANTS:
            _label = f"q/{_name}"
            if _label in experiments:
                print(f"{_label}: already in experiments, skipping")
                continue

            _truncation = variant_truncation(_override)
            _train_ld, _val_ld, _test_ld = loaders_for_truncation(_truncation)
            _model_over, _train_over = split_variant(_override)
            _fingerprint_now = _screen_fingerprint(_model_over, _train_over, _override)

            # Resume: a variant stored by an earlier run of this cell is read back rather than
            # retrained, so a crash mid-screen costs one variant and not all of them.
            _stored = load_saved_run(_label)
            if _stored is not None:
                _stored_history, _model, _stored_fingerprint = _stored
                if _stored_fingerprint == _fingerprint_now:
                    print(f"{_label}: read back from {ABLATION_CACHE_DIR} "
                          f"(best epoch {_stored_history.get('best_epoch')}, "
                          f"epochs run {_stored_history.get('epochs_run')})")
                    experiments[_label] = _stored_history
                    models[_label] = _model
                    test_loaders[_label] = _test_ld
                    continue
                print(f"{_label}: stored run ignored -- the variant no longer matches it")

            _model = create_model(**_model_over)
            _desc = ", ".join(f"{_k}={_v}" for _k, _v in _override.items()) or "configured model"
            print("=" * 60)
            print(f"{_label}: {_desc}")
            print(f"  [truncation={_truncation}, epochs<={QUICK_ABLATION_EPOCHS}]")
            print("=" * 60)

            _history = train_model(_model, _train_ld, _val_ld,
                                   num_epochs=QUICK_ABLATION_EPOCHS, label=_label,
                                   early_stopping=EARLY_STOPPING, **_train_over)
            _history["truncation"] = _truncation
            # Tagged so the reported tables can keep it out of the results: a shortened budget
            # and a single seed make this a screen, not a finding.
            _history["kind"] = "screen"
            save_run(_label, _model, _history, _model_over, _train_over,
                     fingerprint=_fingerprint_now)
            experiments[_label] = _history
            models[_label] = _model
            test_loaders[_label] = _test_ld
            print(f"  best epoch {_history['best_epoch']} of {_history['epochs_run']}: "
                  f"qAUC={_history['best_val_q_auc']:.4f} "
                  f"AP={_history['best_val_ap']:.4f} "
                  f"qF1@{_history['threshold']:.2f}={_history['threshold_q_f1']:.4f}")

        print("\nThis is a screen: every row above is tagged kind='screen' and no reported "
              "table reads them. Re-run the plots cell to include them, then run the final "
              "study (`final_ablation`) for the numbers the report uses.")
    return


@app.cell
def final_ablation(
    ABLATION_CACHE_DIR,
    ABLATION_EPOCHS,
    ABLATION_SEEDS,
    ATTENTION_OPTIONS,
    COMPONENT_VARIANTS,
    EARLY_STOPPING,
    EARLY_STOP_METRIC,
    EARLY_STOP_PATIENCE,
    ENCODER_OPTIONS,
    HEAD,
    LOSS,
    MATCH_FEATURES,
    PAIRWISE_WEIGHT,
    PASSAGE_TRUNCATION,
    POOLING_OPTIONS,
    RUN_ABLATION_STUDY,
    create_model,
    experiments,
    load_saved_run,
    loaders_for_truncation,
    models,
    run_fingerprint,
    run_label,
    save_run,
    saved_run_labels,
    split_variant,
    test_loaders,
    train_model,
    variant_tag,
    variant_truncation,
):
    # --- Section 4: the FINAL study (RUN_ABLATION_STUDY) ----------------------------------
    # The reported, controlled ablation: two blocks, one budget, one stopping rule, run over
    # ABLATION_SEEDS seeds each.
    #
    #   1. the architecture factorial -- ENCODER_OPTIONS x ATTENTION_OPTIONS x POOLING_OPTIONS.
    #      These three interact (cross-/inter-attention need two towers; pooling only reaches
    #      the classifier under a head that reads the pooled vectors), so they are swept as a
    #      cross product rather than one change at a time.
    #   2. the component runs -- one change at a time against the configured model, covering
    #      every COMPONENT_VARIANTS entry from the `ablation_components` cell: the head, the
    #      objective, the passage truncation, the passage encoder, the cross-attention
    #      direction, the question/answer pooling, dropout, the pairwise weight and the
    #      alignment features.
    #
    # The configured model is a member of block 1, so the component runs need no reference of
    # their own: it is already trained at every seed, and the comparison is paired seed by
    # seed -- the sensitive test the diagnostics cell prints.
    #
    # The architecture axis is swept under the configured HEAD, and HEAD has to keep the pooled
    # vectors in the classifier for the POOLING axis to be measurable: head="match" feeds the
    # classifier only the alignment features, so every pooling setting would build the same
    # function (mean/max/final identical, attention pooling different only in its
    # initialisation). The guard below refuses that combination by name.
    #
    # Seeds are the *outer* loop, so the study is readable after one pass, and every finished
    # run is cached under ABLATION_CACHE_DIR (history + weights) so an interrupted study
    # resumes. A cached run is reused only if the settings it was trained under still match.
    if RUN_ABLATION_STUDY:
        import time as _time

        if HEAD == "match":
            raise ValueError(
                "RUN_ABLATION_STUDY with HEAD='match' would sweep the POOLING axis of a "
                "model whose classifier never reads the pooled vectors: head='match' feeds "
                "it only the passage-conditioned alignment features, so every pooling "
                "setting builds the same function (mean/max/final are the same run, and "
                "attention pooling differs only in its initialisation). Set HEAD to "
                "'interaction', 'compare' or 'concat' to sweep pooling, or take "
                "POOLING_OPTIONS out of this grid.")

        def _fingerprint(_model_kwargs=None, _train_kwargs=None, _overrides=None):
            """The settings a stored run must still match to be comparable.

            The list itself is shared with the quick screen and lives in the persistence cell
            (`run_fingerprint`); this only supplies this study's budget. `_overrides` carries
            the raw variant dict for a component run, so a setting that is not a
            `create_model`/`train_model` keyword at all -- the passage truncation, say -- is
            still part of the check.
            """
            return run_fingerprint(ABLATION_EPOCHS, _model_kwargs, _train_kwargs,
                                   overrides=_overrides)

        def _load_cached(label, model_kwargs, train_kwargs, overrides=None):
            """A finished run read back from the store, or None if it has to be retrained.

            The store and its layout live in the persistence cell (`save_run`,
            `load_saved_run`); this only adds the staleness check, so a record trained under
            settings that have since changed is refused rather than silently reused.
            """
            _stored = load_saved_run(label)
            if _stored is None:
                return None
            _history, _model, _stored_fingerprint = _stored
            if _stored_fingerprint != _fingerprint(model_kwargs, train_kwargs, overrides):
                print(f"  [{label}] stored run ignored: the settings changed since it "
                      f"was trained")
                return None
            return _history, _model

        def _save_cached(label, history, model, model_kwargs, train_kwargs, overrides=None):
            """Store one finished run, so a crash costs only the run in flight."""
            save_run(label, model, history, model_kwargs, train_kwargs,
                     fingerprint=_fingerprint(model_kwargs, train_kwargs, overrides))

        # --- block 1: the architecture factorial (every configuration the encoder admits) ---
        _configs = []
        for _encoder in ENCODER_OPTIONS:
            for _attention in ATTENTION_OPTIONS:
                if _encoder == "single" and _attention in ("cross", "inter"):
                    continue  # cross- and inter-attention both need two towers
                for _pooling in POOLING_OPTIONS:
                    _configs.append((_encoder, _attention, _pooling))

        # Each entry is (run label, config label shown in the table, seed, model kwargs,
        # training kwargs, passage truncation, raw override dict for the fingerprint), so both
        # blocks share one cache, one progress estimate and one loop.
        _planned = []
        for _seed in ABLATION_SEEDS:
            for (_encoder, _attention, _pooling) in _configs:
                _tag = variant_tag()
                _planned.append(
                    (run_label(_encoder, _attention, _pooling, _seed, variant=_tag),
                     run_label(_encoder, _attention, _pooling, variant=_tag), _seed,
                     {"encoder": _encoder, "attention": _attention, "pooling": _pooling,
                      "head": HEAD, "match_features": MATCH_FEATURES, "seed": _seed},
                     {"loss": LOSS, "pairwise_weight": PAIRWISE_WEIGHT},
                     PASSAGE_TRUNCATION, None))

            # --- block 2: the component runs, one change at a time off the configured model ---
            for _name, _override in COMPONENT_VARIANTS:
                _model_over, _train_over = split_variant(_override)
                _planned.append(
                    (f"component/{_name}/s{_seed}", f"component/{_name}", _seed,
                     {**_model_over, "seed": _seed}, _train_over,
                     variant_truncation(_override), _override))

        _architecture_runs = len(_configs) * len(ABLATION_SEEDS)
        _component_runs = len(COMPONENT_VARIANTS) * len(ABLATION_SEEDS)
        _missing = sum(1 for _p in _planned if _p[0] not in experiments)
        print(f"{len(_configs)} architectures + {len(COMPONENT_VARIANTS)} components = "
              f"{len(_configs) + len(COMPONENT_VARIANTS)} configurations x "
              f"{len(ABLATION_SEEDS)} seeds = {_architecture_runs + _component_runs} runs "
              f"({_architecture_runs} architecture, {_component_runs} component); "
              f"{_missing} not in memory, up to {ABLATION_EPOCHS} epochs each, early stopping "
              f"{'on' if EARLY_STOPPING else 'off'} "
              f"(patience {EARLY_STOP_PATIENCE} on {EARLY_STOP_METRIC})")

        _planned_labels = {_p[0] for _p in _planned}
        _already_stored = _planned_labels & set(saved_run_labels())
        if _already_stored:
            print(f"resume: {len(_already_stored)} of {len(_planned_labels)} planned run(s) "
                  f"are already stored under {ABLATION_CACHE_DIR}/ and will be read back "
                  f"rather than retrained ({len(_planned_labels - _already_stored)} left to "
                  f"train); a crash costs only the run in flight.")
        else:
            print(f"resume: nothing stored under {ABLATION_CACHE_DIR}/ yet. Every run is "
                  f"written there as it finishes, so re-running this cell after a crash "
                  f"picks up where it left off.")

        _trained = _reused = 0
        _elapsed = 0.0
        for (_label, _run_config, _seed, _model_kwargs, _train_kwargs, _truncation,
             _overrides) in _planned:
            if _label in experiments:
                _reused += 1
                continue

            # A component run that changes `truncation` is encoded differently, so it gets its
            # own loaders; everything else shares the notebook's three.
            _train_ld, _val_ld, _test_ld = loaders_for_truncation(_truncation)

            _cached = _load_cached(_label, _model_kwargs, _train_kwargs, _overrides)
            if _cached is not None:
                _history, _model = _cached
                print(f"{_label}: read back from {ABLATION_CACHE_DIR} "
                      f"(best epoch {_history.get('best_epoch')}, "
                      f"epochs run {_history.get('epochs_run')})")
            else:
                print("=" * 60)
                print(f"{_label}  (budget {ABLATION_EPOCHS} epochs, seed {_seed}, "
                      f"config {_run_config}, truncation={_truncation})")
                print("=" * 60)
                _t0 = _time.time()
                _model = create_model(**_model_kwargs)
                _history = train_model(_model, _train_ld, _val_ld,
                                       num_epochs=ABLATION_EPOCHS, label=_label,
                                       seed=_seed, early_stopping=EARLY_STOPPING,
                                       **_train_kwargs)
                _history["truncation"] = _truncation
                _history["config"] = _run_config
                _save_cached(_label, _history, _model, _model_kwargs, _train_kwargs,
                             _overrides)
                _elapsed += _time.time() - _t0
                _trained += 1
                print(f"  {_label}: best epoch {_history['best_epoch']} of "
                      f"{_history['epochs_run']} trained"
                      f"{' (stopped early)' if _history['stopped_epoch'] else ''}: "
                      f"q_auc={_history['best_val_q_auc']:.4f} "
                      f"q_ap={_history['best_val_q_ap']:.4f} "
                      f"val_ap={_history['best_val_ap']:.4f}")

            # A run cached from a different export may lack the bookkeeping keys; fill them
            # rather than failing, so the plots and the summary can read every row.
            _history["truncation"] = _history.get("truncation") or _truncation
            _history["config"] = _history.get("config") or _run_config
            experiments[_label] = _history
            models[_label] = _model
            test_loaders[_label] = _test_ld

            _still = sum(1 for _p in _planned if _p[0] not in experiments)
            if _trained and _still:
                _per = _elapsed / _trained
                print(f"  [progress] {_trained} trained in this cell at "
                      f"~{_per:.0f}s each (cached runs are free); {_still} run(s) left, "
                      f"~{_per * _still / 60:.0f} min at that rate")

        print(f"\nStudy done: {_trained} trained, {_reused} already in memory. "
              f"Re-run the plots cell, then the summary table.")
    return


@app.cell
def ablation_summary(
    SAMPLE_SEED,
    THRESHOLD_RULE,
    baseline_errors,
    baseline_results,
    evaluate,
    experiments,
    flat_test,
    models,
    plt,
    test_loader,
    test_loaders,
):
    # Aggregated view of the section-4 grid: mean and spread over seeds for every
    # (encoder, attention, pooling) configuration, on the held-out test split, with the
    # baselines underneath. Re-run after the grid cell.
    import statistics as _stats


    def _test_metrics(_label):
        """Test-split metrics for one run, at the run's own declared cut and rule."""
        _model = models.get(_label)
        if _model is None:
            return None
        _hist = experiments[_label]
        return evaluate(_model, test_loaders.get(_label, test_loader),
                        threshold=_hist.get("threshold", 0.5),
                        rule=_hist.get("threshold_rule") or THRESHOLD_RULE)


    _groups = {}
    for _label, _hist in experiments.items():
        if _hist.get("kind") == "screen":
            continue   # quick_ablation is a screen, not a result: not a table row
        _groups.setdefault(_hist.get("config") or _label, []).append(_label)


    def _mean_of(_values, _fallback=float("nan")):
        _vals = [_v for _v in _values if _v is not None]
        return sum(_vals) / len(_vals) if _vals else _fallback


    def _sd_of(_values):
        _vals = [_v for _v in _values if _v is not None]
        return _stats.pstdev(_vals) if len(_vals) > 1 else 0.0


    _rows = []
    for _config, _labels in _groups.items():
        _hists = [experiments[_l] for _l in _labels]
        _tests = [t for t in (_test_metrics(_l) for _l in _labels) if t]
        _rows.append({
            "config": _config,
            "n": len(_labels),
            # val_q_auc is the selection metric and the sort key; val_ap is kept because
            # runs from before the change only have it.
            "val_q_auc": _mean_of([_h.get("best_val_q_auc") for _h in _hists]),
            "val_q_auc_sd": _sd_of([_h.get("best_val_q_auc") for _h in _hists]),
            "val_ap": _mean_of([_h.get("best_val_ap") for _h in _hists]),
            # Per-run values as well as the mean: the figure below plots one point per
            # seed, which is what makes a gap smaller than the seed spread visible as
            # overlapping points instead of something inferred from the sd column.
            "val_q_auc_each": [_h.get("best_val_q_auc") for _h in _hists],
            "test_q_auc_each": [t["q_auc"] for t in _tests],
            "test_auc": _mean_of([t["auc"] for t in _tests]),
            "test_ap": _mean_of([t["ap"] for t in _tests]),
            "test_q_auc": _mean_of([t["q_auc"] for t in _tests]),
            "test_q_ap": _mean_of([t["q_ap"] for t in _tests]),
            "test_q_em": _mean_of([t["q_em"] for t in _tests]),
            "test_q_f1": _mean_of([t["q_f1"] for t in _tests]),
            "test_q_f1_sd": _sd_of([t["q_f1"] for t in _tests]),
            "yes_rate": _mean_of([t["yes_rate"] for t in _tests], 0.0),
        })


    def _rank_key(_r):
        # A configuration with no within-question metric yet sorts last, not first.
        return (-(_r["val_q_auc"] if _r["val_q_auc"] == _r["val_q_auc"] else -1.0),
                -(_r["test_q_auc"] if _r["test_q_auc"] == _r["test_q_auc"] else -1.0))


    _rows.sort(key=_rank_key)

    _header = (f"{'config':<34}{'n':>3}{'valQAUC':>9}{'sd':>7}{'tAUC':>8}{'tAP':>8}"
               f"{'tqAUC':>8}{'tqAP':>8}{'tqEM':>8}{'tqF1':>8}{'sd':>7}{'yes%':>7}")
    print("=" * len(_header))
    print("Section 4: final ablation (mean over seeds, sorted by validation within-question AUC)")
    print("=" * len(_header))
    print(_header)
    print("-" * len(_header))
    for _r in _rows:
        print(f"{_r['config']:<34}{_r['n']:>3}{_r['val_q_auc']:>9.4f}"
              f"{_r['val_q_auc_sd']:>7.4f}"
              f"{_r['test_auc']:>8.4f}{_r['test_ap']:>8.4f}{_r['test_q_auc']:>8.4f}"
              f"{_r['test_q_ap']:>8.4f}{_r['test_q_em']:>8.4f}{_r['test_q_f1']:>8.4f}"
              f"{_r['test_q_f1_sd']:>7.4f}{_r['yes_rate'] * 100:>6.1f}%")
    print("-" * len(_header))
    for _name, _res in baseline_results.items():
        _v, _t = _res["val"], _res["test"]
        print(f"{_name:<34}{'-':>3}{_v['q_auc']:>9.4f}{'-':>7}{_t['auc']:>8.4f}"
              f"{_t['ap']:>8.4f}{_t['q_auc']:>8.4f}{_t['q_ap']:>8.4f}"
              f"{_t['q_em']:>8.4f}{_t['q_f1']:>8.4f}{'-':>7}"
              f"{_t['yes_rate'] * 100:>6.1f}%")
    if baseline_errors:
        # Repeated here because this is the table the report is copied from, and a missing
        # row in a table reads as a row that was beaten.
        print("-" * len(_header))
        print(f"NOT RUN, so not compared against: {', '.join(baseline_errors)}")
        for _name, _err in baseline_errors.items():
            print(f"  {_name:<12} {_err[:96]}")
    print("=" * len(_header))
    print("valQAUC = mean best within-question validation AUC over seeds (sd across seeds),")
    print("  the metric the checkpoints are selected on; tqAUC/tqAP = the same within-question")
    print("  ranking on test; tqEM/tqF1 = MultiRC's per-question exact match and F1 at each")
    print("  run's declared cut; yes% = share of answers that cut predicts yes for.")
    print("A configuration label carries its head and objective as a tag: cc = concat, ix =")
    print("interaction, cmp = compare, mt = match, and wq = within-question objective, bce =")
    print("plain BCE (e.g. 'ix+wq'). Baseline rows are deterministic (no seed spread) and are")
    print("scored with the same declared rule.")
    print("Baseline rows differ in what they are: always_yes / always_no and lexical_overlap")
    print("are training-free rules; <name> is a frozen word-vector similarity row and")
    print("<name>_bigru *trains* the main architecture with those vectors held fixed;")
    print("flan_t5 is a zero-shot prompted seq2seq model (no gradient step, no fine-tuning).")
    print()

    # Which configurations are actually separated by these runs? A gap is only readable
    # against the spread *within* one configuration. The baselines make the same point from
    # the other side: always-yes beats every model row on qF1 while its qEM is ~0.003, because
    # qF1 is dominated by the *number* of yes predictions and qEM is not.
    if _rows:
        _best = _rows[0]
        print(f"Best by mean validation within-question AUC: {_best['config']}  "
              f"({_best['val_q_auc']:.4f} +/- {_best['val_q_auc_sd']:.4f} over "
              f"{_best['n']} seed(s), test qAUC {_best['test_q_auc']:.4f})")
        _tie = [_r["config"] for _r in _rows[1:]
                if _r["val_q_auc"] == _r["val_q_auc"]
                and abs(_r["val_q_auc"] - _best["val_q_auc"])
                <= max(_best["val_q_auc_sd"], 0.002)]
        if _tie:
            print(f"{len(_tie)} configuration(s) sit within one sd of it and are therefore "
                  f"not distinguishable by these runs:")
            print("  " + ", ".join(_tie))
        else:
            print("No other configuration is within one sd of it.")
        if any(_r["n"] < 2 for _r in _rows):
            print("At least one configuration has a single seed, so its sd is 0 by "
                  "construction: a gap against it cannot be read yet.")
        print("Run more seeds before reading any gap smaller than the sd column. The paired")
        print("comparison in the diagnostics cell is the sensitive test: it differences two")
        print("configurations seed by seed, so the seed effect cancels.")

    # The sd column as a picture. One point per seed: a gap between two configurations that is
    # smaller than the spread inside one of them is overlapping points, not a difference. The
    # chance line and the training-free string heuristic are the two references every
    # configuration has to clear before its position is worth reading.
    if _rows:
        _fig, _axes = plt.subplots(1, 2, figsize=(20, max(4.0, 0.34 * len(_rows) + 2.0)))
        for _ax, _each, _side, _title in (
                (_axes[0], "val_q_auc_each", "val",
                 "validation within-question AUC (the selection metric)"),
                (_axes[1], "test_q_auc_each", "test",
                 "held-out test within-question AUC")):
            for _y, _row in enumerate(_rows):
                _vals = [_v for _v in _row[_each] if _v is not None]
                if not _vals:
                    continue
                _ax.scatter(_vals, [_y] * len(_vals), s=45, color="#1565C0", zorder=3,
                            edgecolor="black", linewidth=0.4)
                _ax.errorbar(_mean_of(_vals), _y, xerr=_sd_of(_vals), fmt="o",
                             color="#C62828", markersize=6, capsize=3, linewidth=1.2, zorder=4)
            _ax.axvline(0.5, color="grey", linestyle=":", linewidth=1.2)
            if "lexical_overlap" in baseline_results:
                _ref = baseline_results["lexical_overlap"][_side]["q_auc"]
                _ax.axvline(_ref, color="#2E7D32", linestyle="--", linewidth=1.2,
                            label=f"lexical_overlap {_ref:.3f}")
                _ax.legend(fontsize=8, loc="lower left")
            _ax.set_yticks(range(len(_rows)))
            _ax.set_yticklabels([_r["config"] for _r in _rows], fontsize=7)
            _ax.set_xlabel(_title)
            _ax.set_title(f"{_title}\nblue = one seed, red = mean +/- 1 sd, dotted = chance",
                          fontsize=11)
        _fig.suptitle("Is a gap bigger than the seed spread?", fontsize=14, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()



    # ---------------------------------------------------------------------------------------
    # What can this split resolve, and does the model beat a string heuristic?
    # ---------------------------------------------------------------------------------------
    # A mean over questions is not a mean over independent samples: a passage's ~11 questions
    # share its text, so a question-level standard error is too small. Resampling *passages*
    # gives the honest scale, and lets a deterministic baseline be compared *paired* on the
    # same passages -- which cancels the passage difficulty both methods share.
    import random as _random

    _qid_of, _passage_of = {}, {}
    for _p, _q, _a, _l in flat_test:
        _passage_of.setdefault(_p, len(_passage_of))
        _qid_of.setdefault((_p, _q), len(_qid_of))
    # Same numbering MultiRCDataset assigns (first appearance in `flat_test`), independent of
    # truncation, so a run and a baseline share question ids.
    _qid_to_passage = {_qid_of[_k]: _passage_of[_k[0]] for _k in _qid_of}

    _metric_cache = {}


    def _cached_test_metrics(_label):
        if _label not in _metric_cache:
            _metric_cache[_label] = _test_metrics(_label)
        return _metric_cache[_label]


    def _aucs_by_passage(_auc_by_question):
        """{passage index: [per-question within-question AUC, ...]}."""
        _out = {}
        for _qid, _auc in (_auc_by_question or {}).items():
            if _qid in _qid_to_passage:
                _out.setdefault(_qid_to_passage[_qid], []).append(_auc)
        return _out


    def _mean_interval(_auc_by_question, _n=2000, _seed=SAMPLE_SEED):
        """(mean, lo, hi, n_passages, n_questions), bootstrapping passages not questions."""
        _by_passage = _aucs_by_passage(_auc_by_question)
        if len(_by_passage) < 5:
            return None
        _keys = sorted(_by_passage)
        _rng = _random.Random(_seed)
        _means = []
        for _ in range(_n):
            _vals = [_v for _k in (_keys[_rng.randrange(len(_keys))] for _ in _keys)
                     for _v in _by_passage[_k]]
            _means.append(_stats.fmean(_vals))
        _means.sort()
        return (_stats.fmean(_auc_by_question.values()), _means[int(0.025 * _n)],
                _means[int(0.975 * _n)], len(_keys), len(_auc_by_question))


    def _paired_interval(_run_auc, _base_auc, _n=2000, _seed=SAMPLE_SEED):
        """(gap, lo, hi, P(run better), n_passages) -- paired bootstrap over shared passages."""
        _shared = {}
        for _qid, _auc in _run_auc.items():
            if _qid in _base_auc:
                _shared.setdefault(_qid_to_passage[_qid], []).append((_auc, _base_auc[_qid]))
        if len(_shared) < 5:
            return None
        _keys = sorted(_shared)
        _rng = _random.Random(_seed)
        _gaps = []
        for _ in range(_n):
            _rows = [_r for _k in (_keys[_rng.randrange(len(_keys))] for _ in _keys)
                     for _r in _shared[_k]]
            _gaps.append(_stats.fmean([_a for _a, _ in _rows])
                         - _stats.fmean([_b for _, _b in _rows]))
        _gaps.sort()
        return (_stats.fmean(_gaps), _gaps[int(0.025 * _n)], _gaps[int(0.975 * _n)],
                sum(_g > 0 for _g in _gaps) / _n, len(_keys))


    # One configuration, one number: the runs of a configuration are averaged *per question*
    # first, so pooling seeds does not weight a passage by how many runs saw it.
    _config_auc = {}
    for _config, _labels in _groups.items():
        _per_run = [(_cached_test_metrics(_l) or {}).get("q_auc_by_question") for _l in _labels]
        _per_run = [_a for _a in _per_run if _a]
        if not _per_run:
            continue
        _shared = (set.intersection(*(set(_a) for _a in _per_run)) if len(_per_run) > 1
                   else set(_per_run[0]))
        _config_auc[_config] = {_q: _stats.fmean([_a[_q] for _a in _per_run]) for _q in _shared}

    _header = (f"{'config':<34}{'questions':>10}{'passages':>9}{'test qAUC':>11}"
               f"{'95% interval (resampling passages)':>36}")
    print("=" * len(_header))
    print("Held-out test within-question AUC, with the interval this split can actually resolve")
    print("=" * len(_header))
    print(_header)
    print("-" * len(_header))
    _resolution = None
    for _config, _auc in sorted(_config_auc.items(), key=lambda _kv: -_stats.fmean(_kv[1].values())):
        _ci = _mean_interval(_auc)
        if _ci is None:
            continue
        _mean, _lo, _hi, _n_passages, _n_q = _ci
        _resolution = 0.5 * (_hi - _lo) if _resolution is None else _resolution
        print(f"{_config:<34}{_n_q:>10}{_n_passages:>9}{_mean:>11.4f}{f'[{_lo:.4f}, {_hi:.4f}]':>36}")
    for _name, _res in baseline_results.items():
        _ci = _mean_interval(_res["test"].get("q_auc_by_question"))
        if _ci is None:
            continue
        _mean, _lo, _hi, _n_passages, _n_q = _ci
        print(f"{_name:<34}{_n_q:>10}{_n_passages:>9}{_mean:>11.4f}{f'[{_lo:.4f}, {_hi:.4f}]':>36}")
    print("-" * len(_header))
    if _resolution is not None:
        print(f"This split resolves differences of about +/- {_resolution:.3f} in a single mean. "
              f"The sd column above is the seed spread, not that resolution.")
    print("Overlapping intervals decide nothing; the paired test below is the sensitive one, "
          "because it removes the passage difficulty both methods share.")
    print()

    if _config_auc and baseline_results:
        _best_config = max(_config_auc, key=lambda _c: _stats.fmean(_config_auc[_c].values()))
        _run_auc = _config_auc[_best_config]
        _header = (f"{'baseline':<24}{'tqAUC':>9}{'gap':>9}{'95% interval of the gap':>28}"
                   f"{'P(model better)':>17}   verdict")
        print("=" * len(_header))
        print(f"Does the model beat the baselines? Paired on the passages both scored ({_best_config})")
        print("=" * len(_header))
        print(_header)
        print("-" * len(_header))
        for _name, _res in baseline_results.items():
            _paired = _paired_interval(_run_auc, _res["test"].get("q_auc_by_question") or {})
            if _paired is None:
                continue
            _gap, _lo, _hi, _p, _n_passages = _paired
            _verdict = ("separated" if _lo > 0 else
                        "SEPARATED, MODEL WORSE" if _hi < 0 else "NOT separated")
            print(f"{_name:<24}{_res['test']['q_auc']:>9.4f}{_gap:>+9.4f}"
                  f"{f'[{_lo:+.4f}, {_hi:+.4f}]':>28}{_p:>17.2f}   {_verdict}")
        print("-" * len(_header))
        print("A row whose interval lies below zero is a baseline the model does not beat, and")
        print("the report has to say so: `lexical_overlap` is three lines of set intersection.")
        print("`always_yes`/`always_no` have one score value and no within-question ranking at")
        print("all -- their 0.500 here is chance by construction, not a measurement.")
    return


@app.cell
def baseline_comparison(
    SAMPLE_SEED,
    THRESHOLD_RULE,
    baseline_results,
    evaluate,
    experiments,
    flat_test,
    models,
    np,
    plt,
    test_loader,
    test_loaders,
):
    # --- Model vs baselines: one clear picture -------------------------------------------
    # Puts the trained model next to every baseline on the held-out test split. Bars carry a
    # passage-bootstrap 95% interval (a passage's ~11 questions share its text, so questions
    # are not independent samples), and the paired panel removes the passage difficulty the
    # model and a baseline share -- that is the sensitive test for "does it beat lexical
    # overlap". Depends on `experiments` (section 3, the quick screen and the grid) and
    # `baseline_results`. Every trained run is shown, with the screen runs colour-separated
    # because they are probes rather than results.
    import random as _rand
    import statistics as _st

    # Include the quick-ablation screen runs in this figure, alongside the reported
    # configurations. They are shortened-budget, single-seed probes, so they get their own
    # colour and are named by their `q/...` label; set this False once the grid has produced
    # the runs the report is written from, so the figure never mixes budgets in one bar.
    INCLUDE_SCREEN_RUNS = True

    try:
        _base = baseline_results
    except NameError:
        _base = {}

    if not experiments or not _base:
        print("Run section 3 (and the baselines cell) first: this cell needs `experiments` "
              "and `baseline_results`.")
    else:
        # question id -> passage index, the same numbering MultiRCDataset assigns
        _cmp_passage, _cmp_qid = {}, {}
        for _p, _q, _a, _l in flat_test:
            _cmp_passage.setdefault(_p, len(_cmp_passage))
            _cmp_qid.setdefault((_p, _q), len(_cmp_qid))
        _cmp_q2p = {_cmp_qid[k]: _cmp_passage[k[0]] for k in _cmp_qid}

        def _cmp_by_passage(_auc_by_q):
            _out = {}
            for _q, _auc in (_auc_by_q or {}).items():
                if _q in _cmp_q2p:
                    _out.setdefault(_cmp_q2p[_q], []).append(_auc)
            return _out

        def _cmp_ci(_auc_by_q, _n=2000, _seed=SAMPLE_SEED):
            """(mean, lo, hi) of a mean within-question AUC, bootstrapping passages."""
            _by_p = _cmp_by_passage(_auc_by_q)
            if len(_by_p) < 5:
                return None
            _keys = sorted(_by_p)
            _rng = _rand.Random(_seed)
            _means = []
            for _ in range(_n):
                _vals = [_v for _k in (_keys[_rng.randrange(len(_keys))] for _ in _keys)
                         for _v in _by_p[_k]]
                _means.append(_st.fmean(_vals))
            _means.sort()
            return (_st.fmean(_auc_by_q.values()), _means[int(0.025 * _n)],
                    _means[int(0.975 * _n)])

        def _cmp_paired(_run_auc, _base_auc, _n=2000, _seed=SAMPLE_SEED):
            """(gap, lo, hi) of model - baseline, paired on the passages both scored."""
            _shared = {}
            for _q, _auc in _run_auc.items():
                if _q in _base_auc:
                    _shared.setdefault(_cmp_q2p[_q], []).append((_auc, _base_auc[_q]))
            if len(_shared) < 5:
                return None
            _keys = sorted(_shared)
            _rng = _rand.Random(_seed)
            _gaps = []
            for _ in range(_n):
                _rows = [_r for _k in (_keys[_rng.randrange(len(_keys))] for _ in _keys)
                         for _r in _shared[_k]]
                _gaps.append(_st.fmean([_a for _a, _ in _rows])
                             - _st.fmean([_b for _, _b in _rows]))
            _gaps.sort()
            return (_st.fmean(_gaps), _gaps[int(0.025 * _n)], _gaps[int(0.975 * _n)])

        # Every trained run, one bar per configuration: a configuration's seeds are averaged
        # per question first, so pooling them cannot weight a passage by how many runs saw it.
        # Screen runs are kept as their own kind -- they are what has been trained while the
        # grid has not run, and the colour keeps them from reading as full-budget results.
        _cmp_configs, _cmp_kind = {}, {}
        for _label, _hist in sorted(experiments.items()):
            _is_screen = _hist.get("kind") == "screen"
            if _is_screen and not INCLUDE_SCREEN_RUNS:
                continue
            _key = _hist.get("config") or _label
            _cmp_configs.setdefault(_key, []).append(_label)
            _cmp_kind[_key] = "screen" if _is_screen else "model"

        _cmp_qem, _cmp_qf1, _cmp_auc_q, _cmp_val = {}, {}, {}, {}
        for _config, _labels in _cmp_configs.items():
            _ms = [evaluate(models[_l], test_loaders.get(_l, test_loader),
                            threshold=experiments[_l].get("threshold", 0.5),
                            rule=experiments[_l].get("threshold_rule") or THRESHOLD_RULE)
                   for _l in _labels]
            _cmp_qem[_config] = _st.fmean([m["q_em"] for m in _ms])
            _cmp_qf1[_config] = _st.fmean([m["q_f1"] for m in _ms])
            _cmp_val[_config] = _st.fmean(
                [h.get("best_val_q_auc") or h.get("best_val_ap") or float("nan")
                 for h in (experiments[_l] for _l in _labels)])
            _per = [m["q_auc_by_question"] for m in _ms if m.get("q_auc_by_question")]
            if _per:
                _shared = (set.intersection(*(set(a) for a in _per)) if len(_per) > 1
                           else set(_per[0]))
                _cmp_auc_q[_config] = {q: _st.fmean([a[q] for a in _per]) for q in _shared}

        def _cmp_entry(_name, _kind, _auc_by_q, _qem, _val):
            _ci = _cmp_ci(_auc_by_q)
            if _ci is None:
                return None
            return {"name": _name, "kind": _kind, "auc": _ci[0],
                    "lo": _ci[1], "hi": _ci[2], "qem": _qem, "val": _val}

        _entries = []
        for _config, _auc in _cmp_auc_q.items():
            _e = _cmp_entry(_config, _cmp_kind[_config], _auc, _cmp_qem[_config],
                            _cmp_val[_config])
            if _e:
                _entries.append(_e)
        _base_auc_q = {}
        for _name, _res in _base.items():
            _base_auc_q[_name] = _res["test"].get("q_auc_by_question") or {}
            _e = _cmp_entry(_name, "baseline", _base_auc_q[_name],
                            _res["test"]["q_em"], _res["val"]["q_auc"])
            if _e:
                _entries.append(_e)
        _entries.sort(key=lambda _e: _e["auc"])

        _colours = {"model": "#1565C0", "screen": "#7FBCE8", "baseline": "#EF6C00"}
        _names = [_e["name"] for _e in _entries]
        _ypos = np.arange(len(_entries))

        # Height follows the row count, so the bar panels stay legible with every trained run
        # in them rather than only one.
        _fig, _axes = plt.subplots(2, 2, figsize=(20, max(9.0, 0.45 * len(_entries) + 5.0)))
        _fig.suptitle(f"Every trained run vs the pretrained baselines on the held-out test "
                      f"split  ({len(_entries)} rows)", fontsize=16, fontweight="bold")

        # (1) within-question AUC, the threshold-free ranking metric, with a bootstrap interval
        _ax = _axes[0][0]
        _ax.barh(_ypos, [_e["auc"] for _e in _entries],
                 color=[_colours[_e["kind"]] for _e in _entries],
                 edgecolor="black", linewidth=0.5)
        _ax.errorbar([_e["auc"] for _e in _entries], _ypos,
                     xerr=[[_e["auc"] - _e["lo"] for _e in _entries],
                           [_e["hi"] - _e["auc"] for _e in _entries]],
                     fmt="none", ecolor="black", capsize=4, linewidth=1.2)
        _ax.axvline(0.5, color="grey", linestyle=":", linewidth=1.5)
        _ax.text(0.5, len(_entries) - 0.3, " chance", color="grey", fontsize=9, va="top")
        for _y, _e in zip(_ypos, _entries):
            _ax.text(_e["hi"] + 0.005, _y, f"{_e['auc']:.3f}", va="center", fontsize=9)
        _ax.set_yticks(_ypos)
        _ax.set_yticklabels(_names, fontsize=9)
        _ax.set_xlabel("test within-question AUC  (0.5 = chance)")
        _ax.set_title("Within-question ranking\n(full budget = blue, screen = light blue, "
                      "baselines = orange, bars = 95% CI resampling passages)", fontsize=11)
        _ax.set_xlim(0.45, min(1.0, max(_e["hi"] for _e in _entries) + 0.06))

        # (2) per-question exact match -- the metric a trivial rule cannot game
        _ax = _axes[0][1]
        _ax.barh(_ypos, [_e["qem"] for _e in _entries],
                 color=[_colours[_e["kind"]] for _e in _entries],
                 edgecolor="black", linewidth=0.5)
        for _y, _e in zip(_ypos, _entries):
            _ax.text(_e["qem"] + 0.002, _y, f"{_e['qem']:.3f}", va="center", fontsize=9)
        _ax.set_yticks(_ypos)
        _ax.set_yticklabels(_names, fontsize=9)
        _ax.set_xlabel("test per-question exact match (qEM)")
        _ax.set_title("Per-question exact match\n(a question counts only if every answer is "
                      "right; always-yes is ~0.003)", fontsize=11)

        # (3) paired gap against each baseline -- the sensitive test
        _ax = _axes[1][0]
        _model_entries = [_e for _e in _entries if _e["kind"] != "baseline"]
        if _model_entries and _base_auc_q:
            _best = max(_model_entries, key=lambda _e: _e["auc"])
            _run_auc = _cmp_auc_q[_best["name"]]
            _rows_y, _row_lab = [], []
            for _name in _base:
                _p = _cmp_paired(_run_auc, _base_auc_q[_name])
                if _p is None:
                    continue
                _rows_y.append(_p)
                _row_lab.append(_name)
            if _rows_y:
                _yy = np.arange(len(_rows_y))
                _ax.errorbar([_p[0] for _p in _rows_y], _yy,
                             xerr=[[_p[0] - _p[1] for _p in _rows_y],
                                   [_p[2] - _p[0] for _p in _rows_y]],
                             fmt="o", color="#1565C0", ecolor="black", capsize=4,
                             markersize=8, linewidth=1.2)
                _ax.axvline(0.0, color="red", linestyle="--", linewidth=1.5,
                            label="no difference")
                for _y, _p in zip(_yy, _rows_y):
                    _verdict = ("model better" if _p[1] > 0 else
                                "MODEL WORSE" if _p[2] < 0 else "not separated")
                    _ax.text(_p[2] + 0.004, _y, f"{_p[0]:+.3f}  [{_p[1]:+.3f}, {_p[2]:+.3f}]  "
                             f"{_verdict}", va="center", fontsize=8)
                _ax.set_yticks(_yy)
                _ax.set_yticklabels(_row_lab, fontsize=9)
                _ax.set_xlabel("test within-question AUC gap: model - baseline")
                _ax.set_title(f"Paired on shared passages ({_best['name']})\n"
                              "an interval crossing zero is a tie", fontsize=11)
                _ax.legend(fontsize=8)

        # (4) validation vs test -- does the ranking hold up on unseen passages?
        _ax = _axes[1][1]
        _lims = [min(_e["val"] for _e in _entries) - 0.03,
                 max(max(_e["auc"] for _e in _entries), max(_e["val"] for _e in _entries)) + 0.03]
        _ax.plot(_lims, _lims, color="grey", linestyle=":", linewidth=1.5, label="y = x")
        for _kind in ("model", "screen", "baseline"):
            _sub = [_e for _e in _entries if _e["kind"] == _kind]
            if _sub:
                _ax.scatter([_e["val"] for _e in _sub], [_e["auc"] for _e in _sub],
                            s=70, c=_colours[_kind], edgecolor="black",
                            label=f"{_kind}s", zorder=3)
        for _e in _entries:
            _ax.annotate(_e["name"].split("/")[-1], (_e["val"], _e["auc"]), fontsize=7,
                         xytext=(4, 4), textcoords="offset points")
        _ax.set_xlabel("validation within-question AUC")
        _ax.set_ylabel("test within-question AUC")
        _ax.set_title("Validation vs test\na point far below the y = x line trades validation "
                      "for test", fontsize=11)
        _ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        print("=" * 100)
        print("Summary (test split)")
        print("=" * 100)
        for _e in sorted(_entries, key=lambda _e: -_e["auc"]):
            print(f"  {_e['name']:<34}{_e['kind']:<10}"
                  f"qAUC={_e['auc']:.4f} [{_e['lo']:.4f}, {_e['hi']:.4f}]   "
                  f"qEM={_e['qem']:.4f}")
        print("=" * 100)
        print("kind: 'model' = the configured model or a grid configuration (full budget); "
              "'screen' = a quick-ablation probe (QUICK_ABLATION_EPOCHS, one seed), shown "
              "because it is trained, not because it is a result.")
        if _model_entries and _base_auc_q:
            _best = max(_model_entries, key=lambda _e: _e["auc"])
            _worse = [n for n in _base
                      if (_cmp_paired(_cmp_auc_q[_best["name"]], _base_auc_q[n]) or (0, 0, 0))[2] < 0]
            _tie = [n for n in _base
                    if (_p := _cmp_paired(_cmp_auc_q[_best["name"]], _base_auc_q[n]))
                    and _p[1] <= 0 <= _p[2]]
            if _worse:
                print(f"The best trained run ({_best['name']}) is BEHIND on test: "
                      f"{', '.join(_worse)} -- the report must say so.")
            if _tie:
                print(f"The best trained run ({_best['name']}) is not separated from: "
                      f"{', '.join(_tie)} (interval crosses zero).")
            if not _worse and not _tie:
                print(f"The best trained run ({_best['name']}) is separated above every "
                      f"baseline on this split.")
    return


@app.cell
def save_results(
    RESULTS_DIR,
    SAVE_RESULTS,
    baseline_errors,
    baseline_results,
    experiments,
    export_results,
    models,
):
    # --- Export: write RESULTS_DIR/ so the tables can be read back without retraining ------
    # The run histories, the baselines, a weights file per trained model and a manifest of the
    # settings they were made under. SAVE_RESULTS and RESULTS_DIR are in the hyperparameters
    # cell, with the rest of the configuration. Keep SAVE_RESULTS off while LOAD_RESULTS is on,
    # so a loaded session does not overwrite the export with itself.
    if SAVE_RESULTS:
        export_results(RESULTS_DIR, experiments, models, baseline_results, baseline_errors)
    else:
        print(f"SAVE_RESULTS is off. Set it True in the hyperparameters cell and re-run this "
              f"cell to write {RESULTS_DIR}/, then set LOAD_RESULTS = True there to read it "
              f"back into section 3 and the baselines cell.")
    return


if __name__ == "__main__":
    app.run()
