# Connectome Sequence Modeling

A selected, sanitized portfolio project on stateful sequence prediction under strict inference constraints.

The original workspace combines data inspection, recurrent-model experiments, streaming validation, and packaging work. This public repository keeps a few reusable engineering pieces while omitting the challenge data, trained weights, submission bundles, private run logs, and account-specific details.

## Engineering focus

- Stateful inference that carries hidden state across ordered steps and resets cleanly between sequences.
- A configurable recurrent model interface that returns both predictions and updated state.
- Memory-bounded weighted-correlation accumulation across batches.
- Validation contracts for finite inputs, sequence ordering, and warm-up behavior.

## Repository contents

- `sequence_contract.py` — sequence-order and state-reset guard.
- `weighted_score.py` — streaming weighted Pearson implementation using mergeable moments.
- `gru_model.py` — configurable PyTorch GRU model skeleton.
- `architecture.md` — high-level design and validation flow.

These modules are building blocks rather than a ready-to-submit model. They require user-provided data and trained weights to produce useful predictions.

## Setup

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
```

## Privacy and scope

This is a curated public subset. It does not contain challenge datasets, model checkpoints, ONNX files, submission archives, detailed experiment results, private logs, account identifiers, or credentials. No score or performance claim is published here.
