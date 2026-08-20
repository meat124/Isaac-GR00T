# LARA alignment on the N1.7 AV-ALOHA stack

## Why this workspace exists

LARA (arXiv:2606.07100) adds a representation-alignment term to a VLA: a projection of one
DiT hidden token is pulled toward a latent action that a frozen (or co-trained)
latent-motion tokenizer extracts from a pair of frames. The official release is a fork of
GR00T **N1.5**.

Porting AV-ALOHA onto that fork did not work. Across every arm the pooled success rate was
1.4%, the best single task reached 14%, and — decisively — arms with alignment did not
separate from arms without it. Unfreezing the vision tower, and then the LLM as well,
recovered nothing (4.1% and 3.1% on slot_insertion). Whatever was wrong sat underneath the
method, so the method was never actually under test.

This workspace inverts the direction. It starts from the **N1.7** stack whose AV-ALOHA
anchor measures **41.0% (n=200)** and adds only the alignment, so that a difference in
success rate is attributable to LARA rather than to the base deployment.

## What is here

`avaloha-lara` branches from `e574928`, the upstream commit the anchor was measured on
(not upstream `main`: the intervening commits change video decoding and processing
defaults, which would confound the comparison). On top of it:

1. **The AV-ALOHA harness**, applied as a patch from the branch that produced the anchor —
   `examples/AVAloha/` (layout, modality config, kinematics, success metric, converter,
   train and deploy entry points), the `external_dependencies/av-aloha` submodule, and the
   LLM-LoRA core delta the anchor recipe needs (upstream has no LoRA path).
2. **The alignment**, in `gr00t/model/gr00t_n1d7/lara_align.py`, `moto_lam.py`, and
   `gr00t/data/dataset/lara_frame_dataset.py`, wired through `use_lara` and default-off.
3. **`moto/`**, copied verbatim from the LARA release. It defines what the alignment target
   *is*; a reimplementation would silently change the experiment.

## The objective

```
L = L_flow + 0.01 * L_align + 0.01 * L_LAM
L_align = mean(1 - cos(f_psi(h), z))
```

* `z` — the tokenizer's `embed` for the pair `(frame[t], frame[t+15])`: post-`vq_down`,
  **pre-quantisation**, `[B, 8, 32]` flattened to 256-D.
* `h` — `all_hidden_states[-3]`, the second-to-last DiT block's output (the paper's L-2),
  at the token of the final valid action step.
* `f_psi` — a single `Linear(1536, 256)`.
* `L_LAM` — the tokenizer's own reconstruction loss; it is co-trained by default (Eq. 7).

Inference never runs any of it. `get_action` does not touch the tokenizer, and the
tokenizer is not rebuilt when a checkpoint is loaded for deployment, so an aligned
checkpoint runs through the stock `Gr00tPolicy` and the eval harness is unmodified.

## Two things that do not carry over from the N1.5 implementation

**The final action token is a pad token.** N1.7 pads every chunk to the model's
`action_horizon` (40) while AV-ALOHA chunks are 16, so tokens 16..39 are loss-masked
padding whose DiT input is pure flow noise. The paper's "final action-chunk token" is
`[:, -1]`, which lands in that padding. `pool_dit_tokens` derives the true chunk end from
`action_mask` instead. This was measured on the earlier N1.7 work: every run before the fix
aligned to a pad token.

**The frame pair does not go through the video pipeline.** Requesting video observation
indices `[0, 15]` — how the official config obtains the pair — would also hand the VLM the
second frame, moving the policy off the baseline's input distribution and confounding
alignment with input distribution. The pair rides as extra dataset keys instead; the
collator stacks unknown keys and `prepare_input` passes the batch through whole, so the VLA
input is byte-identical to the baseline's.

## Reading the diagnostics

The released tokenizer's embedding is DC-dominated: a constant predictor scores cosine
0.997 against it, and only 15 of its 128 codes are live. The bare cosine therefore has a
cheap degenerate optimum — emit the target's batch mean and read nothing from the DiT.

Logged every `logging_steps` under `lara/`:

| metric | what it says |
|---|---|
| `latent_align_raw` | the objective as optimised (`1 - cos`) |
| `latent_align_centered` | the same after removing both batch means — **near 0 means no per-sample information is transferred**, whatever `raw` says |
| `latent_dc_fraction` | share of the target's energy in its batch mean; high = the shortcut is available |
| `lara_z_eff_rank` | participation ratio of the target's covariance; → 1 is rank collapse |

`latent_align_centered` must be read together with `lara_z_eff_rank`: a strong centred
cosine against a rank-1 target means nothing.

The `guard` arm switches on centring, a per-dim variance floor, a covariance penalty, a
3-layer projector, and mean pooling over valid action tokens. Those are off by default so
the `lara` arm reproduces the paper exactly.

## Running it

```bash
sbatch scripts/slurm/setup_venv.sbatch      # /scratch2/meat124/.venvs/lara-ws
sbatch scripts/slurm/smoke_lara.sbatch      # both arms briefly + a deployment load

scripts/slurm/run_lara_pipeline.sh base     # finetune + eval, chained with afterok
scripts/slurm/run_lara_pipeline.sh lara
```

Evaluation is pinned to RTX3090. Success rate is measured through an EGL-rendered
observation, so the GPU model is part of the measurement, and every eval behind the 41.0%
anchor ran on a 3090.

`base` reproduces the anchor recipe inside this workspace. Its slot_insertion score is the
gate: the anchor is 48.5% (n=200, seed blocks 48 and 49), so anything below ~43.5% means
the port is wrong and the alignment result would not be interpretable.
