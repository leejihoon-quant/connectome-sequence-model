# Architecture and validation notes

## Stateful inference

The predictor processes each sequence in order. Recurrent state is initialized at the first step of a new sequence, then carried forward one step at a time. `SequenceOrderGuard` checks sequence boundaries and monotonic step order; the caller decides how warm-up rows affect model state and whether a prediction is required.

## Model interface

`SequenceGRU` is a configurable PyTorch module. It accepts batches of ordered feature vectors, optionally receives an initial hidden state, and returns output vectors together with the next state. It contains no trained parameters.

## Streaming score aggregation

`WeightedMoments` maintains weighted means, variances, and cross-moments. Independent blocks can be merged without retaining the original rows or averaging per-block correlations. The caller supplies the target, prediction, and weight vectors, and applies any domain-specific masks or clipping before accumulation.

## Validation stages

A production evaluation can check input shapes and finite values, sequence resets, warm-up behavior, output shape and finiteness, deterministic replay, runtime, and model-package size. Dataset-specific rules and measured results are intentionally omitted from this portfolio version.
