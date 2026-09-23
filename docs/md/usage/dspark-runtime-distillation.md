# DSpark runtime trajectory distillation

`train/core/posttrain/train_dspark_first4.py` uses XDL `CoreModel` and `Trainer` with a structured OmegaConf configuration. `DSPARK_FIRST4_CONFIG` selects the YAML. The original defaults still initialize the released draft and frozen BF16 embedding/head.

The runtime adaptation configuration is `train/core/posttrain/train_dspark_w4_prefix.yaml`. Additional optional fields:

| Field | Meaning |
| --- | --- |
| `initial_checkpoint` | Draft directory containing `config.json` and `model.safetensors`. Continues model weights with a fresh AdamW optimizer; not an optimizer-state resume. |
| `training_head` | Optional frozen tensor loaded with `weights_only=True`, with shape equal to the shared vocabulary head. |
| `prefix_loss_weight` | Nonnegative coefficient for negative expected matched prefix length, added to weighted CE. Zero disables it. |
| `residual_rnn_head` | Adds the upstream RNN joint projection with a residual over pretrained Markov latent embeddings. Consumes recurrent state, previous token embedding and draft hidden state. Defaults to false. |
| `head_learning_rate` | Learning rate for the new joint projection when residual_rnn_head is enabled. Other parameters use learning_rate; both rates receive warmup. |
| `causal_draft` | Uses real shifted token inputs inside each draft block, with causal attention and no access to other draft blocks or future target context. Requires sequential causal deployment. |
| `draft_layers` | Optional positive prefix length of the source draft transformer layers. Copies those layers and explicitly discards later layer weights. |

The prefix loss sums cumulative products of teacher-forced correct-token probabilities across valid slots, averaged over valid anchors. Masked slots terminate the prefix. This is a differentiable surrogate, not greedy accepted length or a speed metric. Actual greedy acceptance and throughput must be measured by the deployment runtime on disjoint papers.

Cache files must supply `input_ids`, `loss_mask`, and position-aligned `target_hidden_states`, including complete prompt features. Match the target capture layer IDs and quantized target trajectory used at deployment. Do not feed the causal-MTP cache with zero-filled prompt features to DSpark. A dense BF16 training head made from native dequantized weights approximates the deployed head; it does not reproduce native BF16-rounded-product GEMV arithmetic. Runtime validation remains mandatory.

Tests: `python -m pytest tests/test_dspark_first4_loss.py -q`.

`train/core/posttrain/train_dspark_w4_rnn.yaml` selects the residual RNN experiment. The output third of a newly added joint projection is zero-initialized, so the initial vocabulary correction exactly preserves the pretrained Markov function. The new state uses a sigmoid gate and tanh candidate; output adds a tanh residual to the previous-token latent before the existing vocabulary projection. Saved config contains `markov_head_type=rnn` and `residual_rnn_head=true`. These checkpoints require a matching recurrent inference adapter; the vanilla XQT draft loader is not compatible and must not silently discard joint projection weights. `head_learning_rate` only applies to the joint projection, not the pretrained vocabulary factors. Teacher-forced training does not prove greedy rollout quality.
