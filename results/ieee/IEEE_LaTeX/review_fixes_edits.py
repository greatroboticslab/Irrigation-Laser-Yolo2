"""Fixes from the 2026-10-21 review pass (external read of the paper, checked against the repo).

- Section reference for the confusion matrices was hard-coded "Section~IV"; they are in Results.
- Table II nc: IR-Soil-Only has nc 7 and UV-Strawberry-Pot nc 9 in their data.yaml, not 11.
- Tables V and VI truncated some values and rounded others; all are now rounded from
  results/runs/*/results.csv (best mAP@0.5; Table V's mAP@0.5:0.95 is the maximum over epochs).
- Table VI: "0.00" for datasets that belong to no Stage 2 pool becomes a dash, and the footnote
  says that pool members carry their pool's Stage 2 and Stage 3 results.
- Fig. 3 caption: COCO (yolov5s.pt), not ImageNet; class counts; Stage 2 from COCO; Stage 3 from
  IR-Pot-Background. The figure (scripts/figures/generate_pipeline_figure.py) is drawn for a
  full-column include.

Usage: python review_fixes_edits.py IN.tex OUT.tex [edits.json]
"""

import json
import sys

EDITS = [
    ("consistent with the clean confusion-matrix diagonals of Section~IV.",
     "consistent with the clean confusion-matrix diagonals of Section~\\ref{sec:confusion}."),
    ("\\subsection{Confusion Matrix Analysis}\n",
     "\\subsection{Confusion Matrix Analysis}\n\\label{sec:confusion}\n"),
    # Table II
    ("IR-Soil-Only & 11 & IR & Fine-tuning target \\\\",
     "IR-Soil-Only & 7 & IR & Fine-tuning target; 7 of the 11 levels \\\\"),
    ("UV-Strawberry-Pot & 11 & UV & Outdoor field; merge partner \\\\",
     "UV-Strawberry-Pot & 9 & UV & Outdoor field; merge partner; 9 of the 11 levels \\\\"),
    # Table V
    ("UV-Soil-Only-Baseline & 0.76 & 0.49 \\\\", "UV-Soil-Only-Baseline & 0.76 & 0.50 \\\\"),
    ("UV-Strawberry-Pot & 0.71 & 0.36 \\\\", "UV-Strawberry-Pot & 0.72 & 0.37 \\\\"),
    ("UV-Soil-Only-Target & 0.61 & 0.38 \\\\", "UV-Soil-Only-Target & 0.61 & 0.39 \\\\"),
    ("UV-Pot-Background & 0.58 & 0.38 \\\\", "UV-Pot-Background & 0.59 & 0.38 \\\\"),
    ("IR-Soil-Only & 0.54 & 0.19 \\\\", "IR-Soil-Only & 0.55 & 0.19 \\\\"),
    # Table VI
    ("UV-Soil-Only-Baseline & 0.75 & 0.94 & 0.95 & 0.96 \\\\",
     "UV-Soil-Only-Baseline & 0.76 & 0.94 & 0.96 & 0.96 \\\\"),
    ("IR-Soil-Only & 0.54 & 0.00 & 0.99 & 0.99 \\\\",
     "IR-Soil-Only & 0.55 & --- & 0.99 & 0.99 \\\\"),
    ("UV-Soil-Only-Target & 0.61 & 0.00 & 0.91 & 0.91 \\\\",
     "UV-Soil-Only-Target & 0.61 & --- & 0.91 & 0.91 \\\\"),
    ("UV-Pot-Background & 0.58 & 0.59 & 1.00 & 1.00 \\\\",
     "UV-Pot-Background & 0.59 & 0.59 & 1.00 & 1.00 \\\\"),
    ("UV-Strawberry-Pot & 0.71 & 0.94 & 0.96 & 0.96 \\\\",
     "UV-Strawberry-Pot & 0.72 & 0.94 & 0.96 & 0.96 \\\\"),
    ("Mixed Color and Background & 0.62$^{a}$ & 0.00 & 0.64$^{a}$ & 0.64 \\\\",
     "Mixed Color and Background & 0.62$^{a}$ & --- & 0.64$^{a}$ & 0.64 \\\\"),
    ("on the held-out test split the two runs score 0.673 and 0.741.}",
     "on the held-out test split the two runs score 0.673 and 0.741. Run~1 to~3 are Stages~1 to~3. "
     "A dash marks a dataset in neither Stage~2 pool. UV-Soil-Only-Baseline and UV-Strawberry-Pot "
     "belong to UV-Combined, and IR-Pot-Background and UV-Pot-Background to IR-Combined; their Run~2 "
     "and Run~3 entries are their pool's results, not separate runs.}"),
    # Fig. 3
    ("\\includegraphics[width=0.85\\columnwidth]{figs/fig3_three_stage_training.png}",
     "\\includegraphics[width=\\columnwidth]{figs/fig3_three_stage_training.png}"),
    ("\\caption{The three-stage protocol, initialized from ImageNet/COCO weights. Stage I trains seven "
     "datasets independently: the 11-class UV-Soil-Only-Baseline and its infrared IR-Soil-Only and "
     "ultraviolet UV-Soil-Only-Target counterpart, the 5-class infrared IR-Pot-Background (transfer "
     "source), the ultraviolet UV-Pot-Background, the 11-class outdoor UV-Strawberry-Pot, and the primary "
     "Mixed Color and Background (56 originals, 1157 frames after expansion). Stage II adds the pooled "
     "UV-Combined (UV-Soil-Only-Baseline + UV-Strawberry-Pot) and IR-Combined (UV-Pot-Background + "
     "IR-Pot-Background). Stage III fine-tunes the five targets from the best soil-moisture checkpoint to "
     "produce the final detector.",
     "\\caption{The three-stage protocol. Stage I trains seven datasets independently from COCO weights "
     "(\\texttt{yolov5s.pt}): the 11-class UV-Soil-Only-Baseline, its 7-class infrared counterpart "
     "IR-Soil-Only and 11-class ultraviolet counterpart UV-Soil-Only-Target, the 5-class infrared "
     "IR-Pot-Background (transfer source), the 5-class ultraviolet UV-Pot-Background, the 9-class outdoor "
     "UV-Strawberry-Pot, and the primary Mixed Color and Background (56 originals, 1157 frames after "
     "expansion). Stage II trains the pooled UV-Combined (UV-Soil-Only-Baseline + UV-Strawberry-Pot) and "
     "IR-Combined (IR-Pot-Background + UV-Pot-Background), also from COCO weights. Stage III fine-tunes "
     "the five targets from the Stage I IR-Pot-Background checkpoint to produce the final detector."),
]

# Hue cap. augment_stir_september.py sets hue_shift_limit=8; on uint8 images Albumentations
# adds that in OpenCV hue units (0-179), so the shift is ±16°, verified by running
# HueSaturationValue(hue_shift_limit=(8, 8)) on images of known hue (every output +16.0°).
# No crossover measurement exists in the repo, so that claim is replaced by what is known.
HUE_EDITS = [
    ("caps hue jitter at $\\pm 8^{\\circ}$, an index",
     "caps hue jitter at $\\pm 16^{\\circ}$, an index"),
    ("caps hue jitter at $\\pm 8^{\\circ}$ (the empirical interclass crossover bound) and combines",
     "caps hue jitter at $\\pm 16^{\\circ}$ and combines"),
    ("and a $\\pm 20^{\\circ}$ perturbation applied to a level 3 (pink magenta) frame produces a glow that is "
     "visually indistinguishable from level 7 (blue violet), which injects silently mislabelled samples directly "
     "into training. We measured the interclass crossover threshold empirically across the five LED states and "
     "adopted a hue shift cap of $\\pm 8^{\\circ}$ as the largest perturbation that preserves dominant wavelength "
     "identity for every class.",
     "and a large enough hue perturbation can move a level 3 (pink magenta) frame toward level 7 (blue violet), "
     "which injects silently mislabelled samples directly into training. We therefore cap the hue shift at 8 "
     "OpenCV hue units. OpenCV stores 8-bit hue on a 0 to 179 scale, so one unit is $2^{\\circ}$ and the cap is "
     "$\\pm 16^{\\circ}$; applying the transform to images of known hue shifts them by exactly that amount. The cap "
     "was chosen to be small rather than derived from a measured interclass crossover threshold, and whether "
     "$\\pm 16^{\\circ}$ keeps every level clear of its neighbours has not been tested."),
    ("constrained HSV hue shift capped at $\\pm 8^{\\circ}$ ($p=0.5$)",
     "constrained HSV hue shift capped at $\\pm 16^{\\circ}$ ($p=0.5$)"),
    ("a hue perturbation of $\\pm 7.2^{\\circ}$ at Stage~1 and $\\pm 5.4^{\\circ}$ at Stage~3, both inside the "
     "$\\pm 8^{\\circ}$ crossover cap established in Section~\\ref{sec:method},",
     "a hue perturbation of at most $\\pm 7.2^{\\circ}$ at Stage~1 and $\\pm 5.4^{\\circ}$ at Stage~3 (YOLOv5 "
     "scales hue by a random gain of up to 2\\% and 1.5\\%, \\texttt{hsv\\_h} 0.02 and 0.015), both below the "
     "$\\pm 16^{\\circ}$ offline cap of Section~\\ref{sec:method},"),
    ("a Class Preserving $\\pm 8^{\\circ}$ hue cap augmentation policy",
     "a Class Preserving $\\pm 16^{\\circ}$ hue cap augmentation policy"),
]
EDITS += HUE_EDITS

# IR-Soil-Only gain: 0.9869 - 0.5497 = 0.437 (results.csv), not 0.439.
GAIN_EDITS = [
    ("the IR-Soil-Only variant improves by 43.9 points", "the IR-Soil-Only variant improves by 43.7 points"),
    ("(a $+43.9$ point gain", "(a $+43.7$ point gain"),
]
EDITS += GAIN_EDITS

# Table IV is wider than the column, so its last header was clipped ("mAP fina"); headers go on two lines.
TABLE4_EDITS = [
    ("\\textbf{Run} & \\textbf{Min.\\ ep.} & \\textbf{Final/min.} & \\textbf{mAP peak} & \\textbf{mAP final} \\\\",
     "\\textbf{Run} & \\textbf{Min.} & \\textbf{Final/} & \\textbf{mAP} & \\textbf{mAP} \\\\\n"
     " & \\textbf{ep.} & \\textbf{min.} & \\textbf{peak} & \\textbf{final} \\\\"),
]
EDITS += TABLE4_EDITS

# "Laser" named two different things: the specular return at the probe tip (ablation) and the
# detected bright spot (feature maps). Where the return comes from is not documented, so both
# are named by what is observed. Figs. 7 and 8 are regenerated with the same wording.
LASER_EDITS = [
    ("\\subsection{Laser Reflection Ablation}", "\\subsection{Specular Return Ablation}"),
    ("The \\emph{laser} arm is the unmodified data. The \\emph{no-laser} arm inpaints the detected region "
     "from surrounding pixels. Inpainting leaves a visible smear, so a drop in the no-laser arm alone",
     "The \\emph{return} arm is the unmodified data. The \\emph{no-return} arm inpaints the detected region "
     "from surrounding pixels. Inpainting leaves a visible smear, so a drop in the no-return arm alone"),
    ("the no-laser arm matches the laser arm exactly (0.9940 versus 0.9940)",
     "the no-return arm matches the return arm exactly (0.9940 versus 0.9940)"),
    ("The sham arm is statistically indistinguishable from the laser arm on all three datasets",
     "The sham arm is statistically indistinguishable from the return arm on all three datasets"),
    ("\\textbf{Laser} & \\textbf{Sham} & \\textbf{No laser} \\\\",
     "\\textbf{Return} & \\textbf{Sham} & \\textbf{No return} \\\\"),
    ("because a no-laser arm in which roughly two frames in five still contain the return",
     "because a no-return arm in which roughly two frames in five still contain the return"),
    ("strongest mean activation inside the detected laser region. The progression",
     "strongest mean activation inside the detected LED region. The progression"),
    ("activations reorganize around components of the laser spot, namely its bright core",
     "activations reorganize around components of the bright spot, namely its core"),
    ("respond almost exclusively at the laser location while background",
     "respond almost exclusively at the spot location while background"),
    ("in favor of the laser signature indicates", "in favor of the bright spot indicates"),
    ("inside the detected laser region at one backbone stage", "inside the detected LED region at one backbone stage"),
    ("concentrate activation on the laser spot core, halo, and rim", "concentrate activation on the bright spot's core, halo, and rim"),
    ("The red outline marks the most laser-selective channel at each stage.",
     "The red outline marks the most spot-selective channel at each stage."),
]
EDITS += LASER_EDITS


def apply(text):
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"expected 1 match, found {n}: {old[:60]!r}")
    for old, new in EDITS:
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        out = apply(f.read())
    with open(dst, "w") as f:
        f.write(out)
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            json.dump(EDITS, f)
