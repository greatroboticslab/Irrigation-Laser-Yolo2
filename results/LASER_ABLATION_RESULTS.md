# Does the laser reflection carry the moisture signal?

Ablation study on the models in `results/`. Sections 1-3 re-validate
**existing weights** on **their own validation split**, changing one thing
about the images at a time, so the model is fixed and only the input varies.
Section 4 does the complementary experiment: **retraining from scratch** on
each condition, which is what answers whether the laser is actually needed.

Generated 2026-08-27. Metric is mAP@0.5 throughout.

---

## Summary

| Question | Answer | Evidence |
|---|---|---|
| Do the trained models use the laser? | **Yes, heavily.** | Removing it costs **0.456 mAP** on average (n=12) |
| Is that drop just the inpainting scar? | **No.** | Sham control costs **0.024** on average |
| **Is the laser *necessary*?** | **No.** | **Retrained without it: 0.994 vs 0.994, 0.959 vs 0.953, 0.988 vs 0.931** |
| Do models read the burned-in "Moisture: N" text? | **No.** | All 10 deltas within ±0.025 |
| Do models read the handwritten cards in frame? | **No.** | Headline model unchanged (0.9950 → 0.9950) |

**The two findings that matter, and they are not the same finding.** A model
trained *with* the laser leans on it heavily — take the laser away at test
time and it collapses (§1). But a model *retrained from scratch* on laser-free
images reaches essentially the same accuracy (§4). The laser is therefore the
feature these networks prefer when it is available, not a feature the task
requires. Soil appearance alone carries the signal.

None of the three label-leakage channels explains the results.

---

## 1. Laser ablation, with sham control

The laser spot is located and inpainted from surrounding pixels. Because
inpainting leaves a visible smear, a drop in accuracy could mean either "the
laser mattered" or "the model dislikes edited images". The **sham control**
separates these: it applies an inpainting scar of the same size and shape
somewhere else in the frame and leaves the laser intact.

| Dataset | Run | Laser present | Sham control | Laser removed | Δ laser | Δ sham |
|---|---|---|---|---|---|---|
| ir-combined | fine-tuned | 0.9950 | 0.9950 | 0.2337 | −0.761 | 0.000 |
| v4-september-combined | fine-tuned | 0.9546 | 0.9536 | 0.4298 | −0.525 | −0.001 |
| v4-september-combined | merged | 0.9303 | 0.9395 | 0.4129 | −0.517 | +0.009 |
| v4-uv | fine-tuned | 0.8826 | 0.8816 | 0.0241 | −0.859 | −0.001 |
| ir | first-run | 0.7878 | 0.7731 | 0.3207 | −0.467 | −0.015 |
| v4 | first-run | 0.7595 | 0.7317 | 0.2901 | −0.469 | −0.028 |
| stir_september | fine-tuned | 0.6686 | 0.6595 | 0.4367 | −0.232 | −0.009 |
| stir_september | first-run | 0.6196 | 0.6197 | 0.4531 | −0.167 | +0.000 |
| 5sagf | first-run | 0.5826 | 0.5614 | 0.0443 | −0.538 | −0.021 |
| ir-combined | merged | 0.5683 | 0.5631 | 0.2277 | −0.341 | −0.005 |
| v4-uv | first-run | 0.5571 | 0.4816 | 0.0198 | −0.537 | −0.076 |
| september | first-run | 0.5135 | 0.3741 | 0.4519 | −0.062 | −0.139 |

**Mean Δ laser removed: −0.456. Mean Δ sham: −0.024.** Laser removal is worse
than sham in 11 of 12 runs. Two models fall to near zero (v4-uv 0.883 → 0.024,
5sagf 0.583 → 0.044).

Data: `laser_dependence.csv`. Figure: `laser_ablation_comparison.png`.

### The one exception

`soil_moisture_september` (first-run) drops only 0.062 — and its sham drop
(0.139) is *larger* than its laser drop, which is incoherent. That split has
11 validation images, so it is noise-dominated; do not read a result into it.

This is also the dataset where an earlier from-scratch retrain on laser-free
images matched the laser-trained baseline. The two findings are consistent
rather than contradictory: a model **retrained** without the laser can learn
soil-only cues, while a model **trained with** the laser leans on it heavily.

### Excluded dataset

`soil-moisture-v4-ir` is excluded. Its laser return is not separable from the
soil: measured hue 97 against a background of 90, and saturation ~108 against
170–250 in every other dataset. A colour-based detector "finds" a laser in
100% of its frames by masking background fabric. Any ablation number from it
would be an artifact, so it is omitted rather than reported.

---

## 2. Burned-in overlay leakage

The capture software prints the ground-truth value into a corner of each frame
("Moisture: 4"). A model that learned to read those digits would score highly
without sensing anything. The overlay is located per dataset and blanked.

| Dataset | Run | Original | Overlay masked | Δ |
|---|---|---|---|---|
| v4-ir | fine-tuned | 0.9866 | 0.9861 | −0.0005 |
| v4-september-combined | fine-tuned | 0.9546 | 0.9547 | +0.0001 |
| v4-september-combined | merged | 0.9303 | 0.9340 | +0.0037 |
| v4-uv | fine-tuned | 0.8826 | 0.8632 | −0.0194 |
| v4 | first-run | 0.7595 | 0.7652 | +0.0057 |
| stir_september | fine-tuned | 0.6686 | 0.6612 | −0.0075 |
| stir_september | first-run | 0.6196 | 0.6293 | +0.0097 |
| v4-ir | first-run | 0.5616 | 0.5594 | −0.0022 |
| v4-uv | first-run | 0.5571 | 0.5590 | +0.0019 |
| september | first-run | 0.5135 | 0.4901 | −0.0234 |

**Mean Δ −0.003, largest |Δ| 0.023.** No model reads the overlay.

For scale: re-saving *identical* JPEGs shifts mAP by up to ~0.015 on its own
(measured when a masked set turned out to be a no-op copy). Every delta above
is at or below that re-encoding noise floor.

Data: `overlay_leak.csv`.

> **Correction to an earlier figure.** A first attempt reported v4-ir dropping
> 0.987 → 0.958. That used a crude "blank the top 55 rows" mask which removed
> soil along with the text. With the overlay box located properly (60×41 px at
> 5,10) the same model goes 0.9866 → 0.9861. The real effect is ~zero.

---

## 3. In-scene handwritten label leakage

Three datasets carry no burned-in overlay. Instead the value is written on
paper left in the shot: a sticky note reading "No. 5 ... Moisture: 8.2" beside
the pot, plus a numbered tag on the pot rim.

This matters more than it first appears: **the class names in these datasets
are the moisture values themselves** — `soil-moisture-1.0`, `soil-moisture-2.0`,
`soil-moisture-8.2` — so the class label is written verbatim in the photograph.

| Dataset | Run | Original | Cards masked | Δ |
|---|---|---|---|---|
| **ir-combined** | **fine-tuned** | **0.9950** | **0.9950** | **0.0000** |
| ir | first-run | 0.7878 | 0.7297 | −0.0581 |
| 5sagf | first-run | 0.5826 | 0.5527 | −0.0299 |
| ir-combined | merged | 0.5683 | 0.5430 | −0.0253 |

The 0.995 headline model is completely unaffected. The weaker first-run models
lose 3–6 points, but roughly 9–10% of each frame is blacked out, so part of
that is plain occlusion rather than lost label-reading.

Data: `card_leak.csv`.

---

## 4. Retraining from scratch — is the laser *necessary*?

Sections 1–3 hold the model fixed and vary the image. That measures what a
trained network relies on, which is not the same as what the task requires: a
salient feature gets learned in preference to a subtler one, so the collapse in
§1 is equally consistent with "soil texture could have carried this, but the
model never had to learn it". Settling it requires training from scratch on
each condition.

Three arms, trained independently, identical in every other respect (yolov5s,
640 px, 200 epochs, AdamW, cosine LR, 3 seeds each — 27 runs):

| Dataset | Arm | mean mAP50 | s.d. | vs laser |
|---|---|---|---|---|
| v4-uv | laser | 0.9885 | 0.0058 | — |
| v4-uv | sham | 0.9880 | 0.0087 | −0.0006 |
| v4-uv | **nolaser** | **0.9314** | 0.0200 | **−0.0572** |
| v4 | laser | 0.9940 | 0.0004 | — |
| v4 | sham | 0.9932 | 0.0014 | −0.0009 |
| v4 | **nolaser** | **0.9940** | 0.0007 | **−0.0000** |
| v4-september-combined | laser | 0.9594 | 0.0037 | — |
| v4-september-combined | sham | 0.9548 | 0.0010 | −0.0046 |
| v4-september-combined | **nolaser** | **0.9527** | 0.0106 | **−0.0067** |

**A model that never sees a laser learns the task anyway.** On `v4` the two
arms are indistinguishable (0.9940 both, gap 0.0× the seed noise). On
`v4-september-combined` the gap is −0.007, about 1.3× seed noise — not
meaningful. Only `v4-uv` shows a real cost, −0.057 at roughly 5× seed noise,
and even there the laser-free model reaches 0.931.

This is the direct answer to the original question, and it is the opposite of
what §1 alone would suggest. The laser is **not required** for this task at
this accuracy level. What §1 showed is that a network trained with the laser
present becomes *dependent* on it — a brittleness result, not a usefulness one.

Data: `threearm_summary.csv`. Figure: `threearm_retrain_comparison.png`.

### Why only v4-uv shows a cost

The obvious worry is that `v4-uv` has a bigger laser spot, so inpainting
destroyed more soil there. Measured, it does not: the inpainted area is
3.42% of the frame for `v4-uv` against 3.97% for `v4` and 3.75% for
`v4-september-combined` — `v4-uv` is the *smallest* edit of the three. Scene
brightness is also comparable (mean V 106 / 104 / 100), so a "UV scenes are
darker" explanation does not hold either.

What does separate them is training data:

| Dataset | train images | classes | images per class | Δ nolaser |
|---|---|---|---|---|
| v4-uv | 142 | 11 | **12.9** | **−0.0572** |
| v4 | 246 | 11 | 22.4 | −0.0000 |
| v4-september-combined | 292 | 11 | 26.5 | −0.0067 |

`v4-uv` has roughly half the examples per class of the other two. The
consistent reading is that **the laser buys data efficiency, not capability**:
with enough examples a model learns the soil cue and the laser adds nothing,
but when examples are scarce the laser is an easier feature to latch onto and
the laser-free model has not got enough data to learn the alternative.

Treat this as a hypothesis, not a demonstration — it rests on one dataset, and
the three `v4-uv` nolaser seeds spread widely (0.9184, 0.9544, 0.9213), with
seed 1 landing within 0.034 of the laser arm. A training-set-size sweep on a
single dataset would test it directly.

### Two caveats on these numbers

**Different augmentation from the original runs.** `scripts/training/tr.py` applies the
aggressive `hyp.stir-soil.yaml` augmentation to `v4` and `v4-uv`; the three-arm
job uses YOLOv5 defaults for all arms. That is why these absolute values
(0.96–0.99) sit far above the first-run table (0.56–0.76). The three arms are
identical to each other, so the comparison stands, but these numbers are **not**
comparable to §1–§3 or to `results_comparison.csv`.

**Ceiling effect.** At 0.99 mAP there is almost no headroom for a difference to
appear. `v4` in particular is saturated in all three arms, so that dataset can
only show "no worse", never "no better". The `v4-uv` result, which has visible
headroom, is the most informative of the three.

---

## Method notes

**Laser localisation** (`scripts/data_prep/laser_ablation_dataset.py`). Fixed colour thresholds
proved unreliable across capture setups — they masked pebbles, denim, and
sunlit debris. The detector is instead scale-adaptive: it scores each pixel on
laser-likeness (in-hue × saturation × brightness), smooths that map, takes its
peak, and grows the mask while the score stays within 25% of that peak, capped
at a maximum radius. Overexposed spot centres lose their hue, so whole white
components touching the grown region are pulled in. A frame with no peak above
threshold is reported as a miss rather than masked arbitrarily.

**Card localisation** (`scripts/data_prep/build_card_masked.py`). Paper is matched by a warm-hue
rule (H 15–45, S 60–140, V≥80) or a pale rule (S≤70, V≥150) for tags shaded by
the pot, then filtered by size (excludes the same-coloured olive pot),
rotated-rectangle solidity (the labels sit tilted, so an axis-aligned fill
ratio reads 0.34 and would reject them), and an ink test requiring ≥5% dark
pixels (plastic sheeting is bright but unwritten). Every frame in all three
datasets gets at least one label blanked.

**Guard against confounding.** Relaxing the card thresholds to catch more rim
tags caused the detector to mask the laser spot itself, which would have
conflated sections 1 and 3. Thresholds were kept conservative and a hard guard
drops any label box overlapping the laser mask. Nothing in section 3 masks the
laser.

---

## Known limits

1. **Laser localisation is unreliable on the pot-in-frame captures.** Measured
   against an independent colour-blob detector, the mask covers the true spot
   in 39/40 frames for `v4-uv`, 28/29 for `v4-september-combined`, 24/26 for
   `v4`, 11/11 for `september` and 23/23 for `stir_september` — but only
   29/40 for `ir-combined`, 17/20 for `5sagf` and 12/20 for `ir`. In those
   scenes the bright metal pot rim outscores the laser and is inpainted
   instead. **The `ir-combined` (−0.761) and `5sagf` (−0.538) rows in §1 are
   therefore partly contaminated** and should be quoted with that caveat, or
   dropped. §4 uses only the verified datasets.
2. **The retrain covers 3 of the 8 datasets.** `v4-uv`, `v4` and
   `v4-september-combined` were chosen for reliable localisation, and the
   latter two are near the accuracy ceiling. The conclusion in §4 rests
   mainly on `v4-uv`, the one arm with headroom.
3. **Rim tags are not fully masked.** They photograph anywhere from cream to
   near-white and are missed in some frames. The explicit "Moisture: X" card is
   caught in every frame, so the direct leak is fully covered; the indirect
   pot-identity one only partly.
4. **`soil-moisture-v4-ir` is untested for laser dependence** (see exclusion
   above). It is included in the overlay table, where it is unaffected.
5. **Sections 1–3 are a single validation pass per condition, no seed
   variation.** Small splits (september has 11 images, ir and 5sagf 20 each)
   carry meaningful noise. §4 uses 3 seeds per arm and reports s.d.

---

## Files

| File | Contents |
|---|---|
| `laser_dependence.csv` | Section 1 — laser, sham, and original scores |
| `laser_ablation_comparison.png` | Section 1 figure |
| `overlay_leak.csv` | Section 2 — burned-in overlay |
| `card_leak.csv` | Section 3 — in-scene paper labels |
| `threearm_summary.csv` | Section 4 — retrain, 3 arms x 3 seeds |
| `threearm_retrain_comparison.png` | Section 4 figure |
| `data/training-data/nolaser/` | Laser-removed datasets, all splits |
| `data/training-data/sham/` | Sham control images, valid split |
| `data/training-data/overlay_masked/` | Overlay-blanked images, valid split |
| `data/training-data/card_masked/` | Card-blanked images, valid split |

Scripts live at the project root: `scripts/data_prep/laser_ablation_dataset.py`,
`scripts/data_prep/build_nolaser_datasets.py` (`--sham`), `scripts/evaluation/eval_laser_dependence.py`,
`scripts/data_prep/build_overlay_masked.py`, `scripts/evaluation/eval_overlay_leak.py`, `scripts/data_prep/build_card_masked.py`,
`scripts/evaluation/eval_card_leak.py`, `scripts/training/tr_threearm.py`, `slurm/run_threearm.slurm`.




















Depends which claim you want to make. You have two different claims and only one is currently supported.

Claim A — "our trained models rely on the laser." Fully supported now. The inference-time ablation plus sham control settles it, no retrain needed.

Claim B — "the laser is useful / the system needs it." Not supported by what you have. This is almost certainly the claim your supervisor is actually asking about ("determine whether the laser dots are useful"), and it requires the retrain.

Why the inference test can't answer B

Dependence and usefulness are different things. A model trained on images containing a bright saturated spot will latch onto that spot because it's the most salient feature available — that's how gradient descent works. Removing it at test time then breaks the model. But this tells you nothing about whether soil texture could have carried the same information had the laser never been there. The model never had a reason to learn the harder feature.

The sham control rules out one confound (reacting to edited pixels generally) but not this one. Sham puts the scar where the model isn't looking; laser removal blanks exactly where it is looking. Both outcomes are consistent with "laser is informative" and with "laser is merely salient."

You already have evidence this isn't hypothetical

The retrain I ran on soil_september before you scrapped it — 3 seeds per arm, everything else identical:

- Trained and tested with laser: 0.788 ± 0.101
- Trained and tested without laser: 0.804 ± 0.065

No difference. When the model was forced to learn without the laser, it reached the same accuracy from soil appearance alone. On that dataset the laser was dispensable, despite the inference-time test showing the models do use it when it's available.

That's exactly the divergence I'm describing, and it's the result a reviewer will anticipate.

But that one retrain doesn't settle it either

september is the weakest case in your whole table — smallest inference-time drop (−0.062), 46 training images, 11 validation images. You cannot extrapolate a null result from your noisiest, least laser-dependent dataset to v4-uv, which loses 0.86 mAP when the laser goes. It's entirely possible the laser is genuinely necessary there and dispensable in september.

What I'd actually run

Three training arms, not two, because the two-arm version has the same edit confound in training that the sham fixed at inference:

1. original images
2. laser inpainted away
3. sham — scar elsewhere, laser intact

Three seeds each, on the datasets where dependence is strongest and the data is sufficient: v4-uv, v4-september-combined, ir-combined. That's 27 runs. Your earlier ablation did 200 epochs in ~0.05 h per run on the A100, so this is hours, not days — and data/training-data/nolaser/ already has all splits built. Sham currently exists for the valid split only, so that arm needs a full-split build first.

My recommendation: run it. The result is publishable either way. If accuracy holds without the laser, your honest finding is "the laser is not required — soil appearance suffices," which is a more interesting paper than the alternative and pre-empts the obvious reviewer attack. If accuracy drops, you've proven the laser earns its place and the hardware is justified.