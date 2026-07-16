# SAE-Latent Alignment Adversarial Training

Research codebase for the idea from our design conversation: standard
adversarial training (Madry-style PGD-AT or TRADES), plus a term that pulls
an adversarial example's representation toward its clean counterpart's
representation -- computed in a sparse autoencoder's (SAE) latent space
instead of raw activation space -- plus four different strategies for
keeping that SAE valid while the backbone keeps changing under training.

Nothing here is a toy/demo. Every method described below runs on real
CIFAR-10/100 through a shared training loop; the only thing that changes
between "plain PGD-AT" and "TRADES + SAE alignment with jointly-trained
drift control" is a config file.

## Is this a good idea? (one-paragraph recap)

Plausible and worth running, not a slam dunk. It's a natural refinement of
an already-successful family (feature-alignment AT: ALP, feature denoising,
AR-AT, AFA, DICAR/2026's "Robust Alignment", 2026's diffusion-representation-
alignment DRA) -- those align *raw* activations, and the open question is
whether a sparse, disentangled basis does better. The single most important
risk: a paper on SAE concept-representation robustness (arXiv 2505.16004)
found tiny perturbations can move SAE latents largely independently of the
base model's real behavior, which means an alignment loss in SAE space
could in principle be satisfied without buying real robustness (an
obfuscated-gradient-flavored failure mode). This is why
`scripts/evaluate_checkpoint.py` always runs a PGD step-count sweep and
(optionally) AutoAttack, not just PGD-20 -- see "How to tell if a result is
real" below. Treat every number this code produces as provisional until it
survives that check.

## Project layout

```
sae_advtrain/           importable package -- all the actual logic
  config.py              every knob, as nested dataclasses + YAML (de)serialization
  data.py                CIFAR-10/100 loaders (unnormalized) + an offline synthetic dataset for smoke tests
  models/                PreActResNet18/34, WideResNet28-10/34-10, input-normalization wrapper
  hooks.py               forward-hook activation extraction (gap-pooled or per-position/spatial)
  sae.py                 TopK and L1 sparse autoencoders, FVU / dead-feature diagnostics
  attacks.py             generic PGD (Linf/L2), driven by a caller-supplied differentiable loss
  losses.py               alignment losses, SAE reconstruction/residual losses, TRADES KL term
  drift_control.py        the 4 strategies for keeping the SAE valid (see below)
  trainer.py               the training loop: composes base x alignment x drift-control from config
  evaluate.py               clean / PGD / AutoAttack / step-sweep / representation diagnostics
  metrics.py                 linear CKA, accuracy
  cli.py                      shared --config / --set argument parsing
scripts/
  train.py                    single entry point for every training run (clean / madry / trades x align)
  pretrain_sae.py               phase 2: trains an SAE on a frozen backbone's clean activations
  evaluate_checkpoint.py          full evaluation suite for one checkpoint
  aggregate_results.py             collects eval_results_*.json across runs into one comparison table
configs/                            ready-made YAML configs, one per experiment (see below)
run_experiment_suite.sh                runs the whole pipeline in order
```

## Workflow

```
1. python scripts/train.py --config configs/cifar10_clean_pretrain.yaml
       -> runs/clean_pretrain/checkpoints/final.pt   (plain, non-adversarial classifier)

2. python scripts/pretrain_sae.py --config configs/cifar10_sae_pretrain.yaml
       -> runs/sae_pretrain/sae.pt                    (SAE trained on that checkpoint's clean activations)

3. python scripts/train.py --config configs/cifar10_baseline_madry.yaml
   python scripts/train.py --config configs/cifar10_baseline_trades.yaml
   python scripts/train.py --config configs/cifar10_feature_align_raw.yaml
   python scripts/train.py --config configs/cifar10_sae_align_frozen.yaml
   python scripts/train.py --config configs/cifar10_sae_align_frozen_residual.yaml
   python scripts/train.py --config configs/cifar10_sae_align_joint.yaml
   python scripts/train.py --config configs/cifar10_sae_align_periodic_refresh.yaml
       -> runs/<name>/checkpoints/{best,last,final}.pt for each

4. python scripts/evaluate_checkpoint.py --checkpoint runs/<name>/checkpoints/best.pt
       -> runs/<name>/eval_results_best.json for each

5. python scripts/aggregate_results.py runs/* --out results_summary.csv
       -> the comparison table
```

Or just run `bash run_experiment_suite.sh`, which does all of the above in
order. It will take a while (7-8 training runs); comment out lines for
methods you don't want yet.

All configs default to CIFAR-10 fine-tuned from the clean checkpoint (30
epochs, lr 0.01, eps 8/255) so the fast-iteration loop is quick. Every
config starts from the SAME clean checkpoint via `resume_from`, so the
comparison between methods is apples-to-apples. See "Two training regimes"
below before drawing conclusions.

## What each config tests

| Config | base | align.space | drift.mode | Question it answers |
|---|---|---|---|---|
| `cifar10_baseline_madry.yaml` | madry | - | - | Standard PGD-AT floor |
| `cifar10_baseline_trades.yaml` | trades | - | - | TRADES floor |
| `cifar10_feature_align_raw.yaml` | madry | raw | - | **Control**: does alignment help at all, in the well-established raw-activation form? |
| `cifar10_sae_align_frozen.yaml` | madry | sae | frozen | Naive SAE alignment, no drift correction -- watch `sae_fvu` climb in `eval_log.csv` |
| `cifar10_sae_align_frozen_residual.yaml` | madry | sae | frozen_residual | **Recommended starting point** -- SAE-FT-style residual regularization keeps the SAE valid |
| `cifar10_sae_align_joint.yaml` | madry | sae | joint | SAE co-trained every step; most flexible, least outside evidence |
| `cifar10_sae_align_periodic_refresh.yaml` | madry | sae | periodic_refresh | Hybrid: frozen between refreshes, retrained every few epochs |
| `cifar10_trades_sae_align_frozen_residual.yaml` | trades | sae | frozen_residual | Do the two axes (base recipe, alignment) compose? |

The critical comparison is **`feature_align_raw` vs. `sae_align_frozen_residual`**:
raw-activation alignment is already known to help (AR-AT etc.); the actual
research contribution is whether the SAE's disentangled basis does *better*
than that, not whether alignment helps at all.

## Every ambiguity we didn't have an answer for, and how to test it

All of these are config flips, not code changes:

- **Which drift-control strategy wins?** -- run all four `sae_align_*`
  configs, compare `pgd50_multirestart_acc` and `autoattack_acc` in
  `results_summary.csv`, and check `sae_fvu`/`cka_vs_initial` in each run's
  `eval_log.csv` to see how "valid" each strategy actually kept the SAE.
- **`residual_kind: absolute` vs. `delta`** (`drift.residual_kind`) --
  absolute keeps current clean activations reconstructable; delta (needs no
  code change, just this flag) constrains the *drift* from the initial
  backbone instead, which is more permissive. `--set drift.residual_kind=delta`.
- **Alignment distance** (`align.kind`): `l2` / `l1` / `cosine` /
  `feature_add` (the SAE-FT-inspired "only penalize newly-activated
  latents" variant -- more surgical, intended for non-negative SAE codes).
- **Symmetric vs. asymmetric alignment** (`align.stop_grad_clean`) -- AR-AT's
  finding was that pulling only the adversarial branch toward a *fixed*
  clean target beats a symmetric penalty. Flip to `false` to test that
  finding here too.
- **Which layer** (`model.feature_layer`) -- defaults to `auto`
  (`layer3` for PreActResNet, `layer2` for WideResNet); pass an explicit
  layer name (`layer1`/`layer2`/`layer4` etc, see `layer_widths` in
  `models/preact_resnet.py` / `wideresnet.py`) to sweep this.
- **Pooling** (`sae.reduce`): `gap` (pool each feature map to a vector,
  default, matches SAE-FT) vs. `spatial` (treat every spatial position as
  its own token, closer to feature-denoising-style per-pixel alignment).
- **SAE architecture** (`sae.kind`): `topk` (default, no sparsity-coefficient
  tuning) vs. `l1` (classic ReLU+L1, may behave differently under drift).
  Also sweep `sae.dict_size` and `sae.k` / `sae.l1_coef`.
- **lambda** (`align.weight`) and, for the residual term, `drift.residual_weight`
  -- both need a sweep; too low and the term does nothing, too high and it
  can hurt clean/robust accuracy or (for the residual term) over-constrain
  the backbone.
- **The obfuscated-gradient ablation** (`align.inner_max_uses_alignment` +
  `align.inner_max_align_weight`) -- off by default (adversarial examples
  are generated by plain PGD-CE, identical to the Madry baseline, so the
  attack step is directly comparable across methods). Turning it on makes
  the inner PGD maximization also reward moving away from the clean
  representation. This is here for experimentation, not because it's
  recommended -- it's a plausible way the method could go wrong, not a
  plausible way to make it better; expect it to make gradient masking, if
  it happens at all, easier to detect (or possibly to induce it).
- **Does alignment compose with a stronger base recipe?**
  `cifar10_trades_sae_align_frozen_residual.yaml` -- recent work (DRA, 2026)
  found representation alignment gives gains on top of a strong AT
  baseline, not just in place of one.

## Two training regimes -- pick before you draw conclusions

All configs above fine-tune from a clean pretrained checkpoint for 30
epochs at a low learning rate -- small weight movement, the regime closest
to what SAE-FT actually measured, and the one most favorable to a frozen
SAE staying valid. `configs/cifar10_full_scale_template.yaml` is a template
for the other regime: full adversarial training from scratch, 100+ epochs,
standard Madry/TRADES-scale LR -- much larger weight movement, and the
regime where a frozen SAE is most likely to go stale. Whichever
drift-control strategy wins on the fast configs, re-validate it in this
regime before trusting it; the two regimes could easily favor different
options (see the earlier discussion of why `frozen_residual` is the
recommended *starting point*, not a foregone conclusion).

## How to tell if a result is real

Do not conclude "the SAE-aligned model is more robust" from PGD-20 alone.
`scripts/evaluate_checkpoint.py` runs, in order of increasing scrutiny:

1. clean accuracy
2. PGD-20 (fast, standard "during development" number)
3. PGD-50 with 5 random restarts (stronger, still white-box)
4. a **PGD step-count sweep** (10/20/50/100 by default) -- a genuinely
   robust model's accuracy should mostly plateau as steps increase; if it
   keeps dropping substantially, that is the signature of obfuscated
   gradients (Athalye et al. 2018), not real robustness. This is the single
   most important check for this project specifically, given the SAE
   concept-robustness caveat above.
5. **AutoAttack** (`pip install autoattack`) -- mixes white-box and
   gradient-free/black-box attacks, the field-standard check. Falls back to
   a clear message and `None` if not installed; the training/eval code does
   not require it.
6. for `align.space: sae` checkpoints: SAE FVU and dead-feature-fraction on
   the CURRENT backbone's activations (via `compute_representation_diagnostics`)
   -- confirms the SAE was actually still valid at evaluation time, not just
   early in training.

`scripts/aggregate_results.py` pulls all of this into one CSV so you can
compare methods on the metrics that actually matter, not just PGD-20.

## Testing

`configs/smoke_test.yaml` runs the whole pipeline on synthetic random data
(`data.dataset: synthetic`, generated in-memory, no download, no internet
needed) in about a minute on CPU -- useful for confirming a code change
didn't break anything before spending real GPU time. Example matrix (all
of these were exercised during development):

```bash
# every base recipe
python scripts/train.py --config configs/smoke_test.yaml --set base=madry
python scripts/train.py --config configs/smoke_test.yaml --set base=trades
python scripts/train.py --config configs/smoke_test.yaml --set base=clean

# raw alignment
python scripts/train.py --config configs/smoke_test.yaml \
    --set align.enabled=True --set align.space=raw

# sae alignment under each drift mode (needs a pretrained SAE first --
# see scripts/pretrain_sae.py, or point sae.pretrained_path at any
# checkpoint produced by a quick synthetic-data pretrain_sae.py run)
for MODE in frozen frozen_residual joint periodic_refresh; do
  python scripts/train.py --config configs/smoke_test.yaml \
      --set align.enabled=True --set align.space=sae \
      --set sae.pretrained_path=./runs/smoke_sae_pretrain/sae.pt \
      --set drift.mode=$MODE
done
```

## Practical notes

- **Compute**: this container has no GPU, so every number above was
  produced/verified on CPU with tiny synthetic data purely to confirm the
  code runs correctly end to end -- not to produce meaningful accuracy
  numbers. Run the real configs on a GPU machine; a PreActResNet18 AT run
  on CIFAR-10 at the default 30-epoch fine-tuning schedule should take
  well under an hour on a single modern GPU.
- **Dataset download**: `data.dataset: cifar10` uses
  `torchvision.datasets.CIFAR10(download=True)`, which needs outbound
  internet access on whatever machine you run this on.
- **Checkpoints embed the config** (`cfg` field) using `weights_only=False`
  -- only load checkpoints produced by this codebase, not arbitrary
  third-party `.pt` files, since unpickling a full config object is not the
  restricted-safe `weights_only=True` mode PyTorch >= 2.6 defaults to.
- **CIFAR-100 / a different backbone**: `--set data.dataset=cifar100` (the
  training script sets `model.num_classes` automatically from the loader);
  `--set model.arch=wideresnet28_10` once you're past the fast-iteration
  phase (see `cifar10_full_scale_template.yaml`).
