"""Fixes from the 2026-10-23 weakness review (items 1, 3 and 6).

1. Count only real Stage 3 runs: five targets were fine-tuned, four exceed 0.91. The other
   datasets' Stage 3 entries in Table VI are their pool's result. UV-Soil-Only-Baseline's
   "Best" in Table IX is likewise the UV-Combined pool. "Deployment grade" is dropped; the
   stir probe, the primary task, reaches 0.64 validation and 0.74 test mAP@0.5.
3. The augmentation is not hue-preserving in full. RGBShift (+-15 per channel) moves hue by
   at most about 7 deg at full chroma, more than the 16 deg cap below chroma ~120, ~34 deg at
   chroma 60 (worst case over the shift corners, computed with colorsys). HueSaturationValue
   also shifts saturation (+-30) and value (+-25); Downscale (p=0.2) was missing from the list.
   Scale transforms change the spot diameter that Table I lists as a cue.
   Source: scripts/data_prep/augment_stir_september.py.
6. Cut the unevaluated diffusion subsection; "one of five moisture classes" holds only for the
   stir probe; UV-Combined is ultraviolet, not visible light.

Usage: python review2_edits.py IN.tex OUT.tex [edits.json]
"""

import json
import sys

EDITS = [
    # ---- 1: recount and tone down ----
    ("After fine-tuning, six of seven datasets exceed $\\text{mAP@0.5}=0.91$, the IR-combined merge reaches 0.995,",
     "After fine-tuning, four of the five fine-tuned targets exceed $\\text{mAP@0.5}=0.91$ on their validation splits (which, for the soil-only collections, share capture bursts with training), the IR-combined merge reaches 0.995,"),
    ("The framework demonstrates that a commodity sensor and a single fixed camera can support continuous, non contact moisture monitoring at deployment grade accuracy, providing a scalable alternative to per plant transducer arrays.",
     "The framework indicates that a commodity sensor and a single fixed camera can support non contact moisture monitoring, although the primary stir-probe task reaches only 0.64 validation and 0.74 test mAP@0.5 and accuracy under deployment conditions has not been measured."),
    ("showing that all three capture sets reach $\\text{mAP@0.5} \\geq 0.91$ once domain specific weights are in use,",
     "showing that the two capture sets fine-tuned directly reach $\\text{mAP@0.5}$ of 0.91 and 0.99, and the UV-Combined pool that contains the third reaches 0.96, once domain specific weights are in use,"),
    ("lifted every target into the high accuracy regime (Table~\\ref{tab:best}). Six of the seven datasets, all but the stir probe primary, exceeded $\\text{mAP@0.5}=0.91$ under the fine-tuned configuration.",
     "lifted every target (Table~\\ref{tab:best}). Four of the five fine-tuned targets, all but the stir probe primary, exceeded $\\text{mAP@0.5}=0.91$; the four datasets that entered Stage~3 only through a pool carry that pool's result in Table~\\ref{tab:best} and are not separate runs."),
    ("infrared reaches the highest $\\text{mAP@0.5}$ at 0.99, and the two ultraviolet captures reach 0.96 and 0.91, all three clearing the 0.91 deployment-grade threshold.",
     "infrared reaches the highest $\\text{mAP@0.5}$ at 0.99, \\texttt{UV-Soil-Only-Target} reaches 0.91, and the UV-Combined pool that contains \\texttt{UV-Soil-Only-Baseline} reaches 0.96. All three validation splits share capture bursts with training (Section~\\ref{sec:dynamics}), so these figures are upper bounds."),
    ("IR-Soil-Only & 104 & 7 & 0.55 & 0.99 & 0.59 \\\\\n\\bottomrule\n\\end{tabular}\n",
     "IR-Soil-Only & 104 & 7 & 0.55 & 0.99 & 0.59 \\\\\n\\bottomrule\n\\end{tabular}\n\\\\[2pt]\n\\parbox{\\columnwidth}{\\footnotesize UV-Soil-Only-Baseline was not fine-tuned on its own; its Best columns are those of the UV-Combined pool it belongs to.}\n"),
    ("every Stage~3 target enters the $\\geq 0.91$ regime,",
     "every Stage~3 target except the stir probe enters the $\\geq 0.91$ regime,"),
    ("can support continuous, non-contact, deployment grade moisture monitoring of an unmodified consumer device.",
     "can read an unmodified consumer device without contact, as a basis for continuous moisture monitoring."),
    ("together lift six of seven datasets to $\\text{mAP@0.5} \\geq 0.91$ and the IR-combined merge to 0.995.",
     "together lift four of the five fine-tuned targets to $\\text{mAP@0.5} \\geq 0.91$ and the IR-combined merge to 0.995; the primary stir-probe task reaches 0.64 on validation and 0.74 on its test split, and the soil-only validation splits share capture bursts with training, so testing on independent scenes and under field lighting is the next step."),

    # ---- 3: what the augmentation does to the class signal ----
    ("combines it with non hue photometric and full geometric diversity, expanding the 56 image primary dataset to 1{,}157 frames ($20.7\\times$) without injecting hue induced label noise.",
     "combines it with other photometric and full geometric diversity, expanding the 56 image primary dataset to 1{,}157 frames ($20.7\\times$). The cap bounds the explicit hue transform only; Section~\\ref{sec:aug} describes the other transforms that move hue or the spot size."),
    ("\\subsection{Class Preserving Data Augmentation}\n",
     "\\subsection{Class Preserving Data Augmentation}\n\\label{sec:aug}\n"),
    ("Within that constraint, we apply non-hue photometric transforms and full geometric diversity",
     "Alongside it, we apply other photometric transforms and full geometric diversity"),
    ("constrained HSV hue shift capped at $\\pm 16^{\\circ}$ ($p=0.5$);",
     "constrained HSV shift with hue capped at $\\pm 16^{\\circ}$, saturation $\\pm 30$ and value $\\pm 25$ ($p=0.5$);"),
    ("and random shadow ($p=0.3$) and random fog ($p=0.1$).",
     "random shadow ($p=0.3$) and random fog ($p=0.1$); and downscaling to 60 to 90\\% resolution ($p=0.2$)."),
    ("and a combined flip, rotation and brightness change. No transform in this pass alters hue.",
     "and a combined flip, rotation and brightness change. No transform in this pass shifts hue systematically.\n\nThe policy preserves the class signal only approximately. The RGB channel shift also moves hue, by an amount that grows as the spread between a pixel's largest and smallest channel falls: at the worst combination of $\\pm 15$ shifts it is at most about $7^{\\circ}$ for a spread of 255, exceeds the $16^{\\circ}$ cap below a spread of about 120, and reaches about $34^{\\circ}$ at 60, the range of a dim or weakly saturated glow such as level~1. The saturation and value shifts, the brightness and contrast change, and fog alter the saturation and brightness that separate levels 7, 8 and 10 (Table~\\ref{tab:led}), and the crop, scale and perspective transforms alter the spot diameter that Table~\\ref{tab:led} lists as a cue between the same levels. How often these transforms move a frame across a class boundary has not been measured."),

    # ---- 6: cut the diffusion subsection; wording ----
    ("\\subsection{Synthetic Data Label Noise}\nNone of the training sets behind the results reported here contains diffusion-generated frames; the Mixed Color and Background build consists of the 56 originals and their classical augmentations. The diffusion pipeline was built but its output was not evaluated, and one limitation of its design should be stated before it is. Its bounding boxes are sampled from per class spatial priors rather than computed from a true LED detection in the generated frame, so they are labels by assumption. The size of this noise depends on how often the generative model places the LED at the statistical center assumed by the prior. A natural refinement, deferred to future work, is a post hoc detector, a small auxiliary network trained on real frames only, applied to each generated frame to produce a true LED glow box that replaces the spatial prior fallback. For downstream practitioners, the broader lesson is that programmatic labelling of generative augmentation is an estimate, not a measurement, and its variance must be modeled before assuming a synthetic data win.\n\n",
     ""),
    ("Section~\\ref{sec:discussion} discusses implications, label noise limitations, and deployment outlook.",
     "Section~\\ref{sec:discussion} discusses the feature maps, what drives accuracy, and the limits of the illumination comparison."),
    ("assigns one of five moisture classes, and we address",
     "assigns a moisture class (five levels for the stir probe, up to 11 in the supporting datasets), and we address"),
    ("additive for ambient visible light and sub-additive for IR,",
     "additive for the ultraviolet pool and sub-additive for the IR pool,"),
]


def apply(text):
    # In order: a later edit may target text an earlier one wrote.
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"expected 1 match, found {n}: {old[:60]!r}")
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
