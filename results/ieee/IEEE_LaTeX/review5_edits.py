"""Corrections from the 2026-10-26 second external review. Applies on top of review4_edits.py.

1. Split method: the paper said the soil-only export split frames "at random"; that is not
   recorded anywhere. What is measured is that the splits were not grouped by burst.
2. Cross-burst drops (0.565, 0.224) are COCO-initialized runs; the fine-tuned headline numbers
   were not cross-validated, so how much of them survives is unmeasured.
3. High-confidence errors were measured on the unregularized run only. Regularized Stage 3 CV
   arm (results/stir_cv_oof_reg.csv, epoch 59): 26 errors in 168 held-out readings of originals
   (56 x 3 seeds), none above 0.80 confidence; mean confidence 0.72 correct, 0.53 wrong; 2 of
   965 errors above 0.9 with augmented copies included.
4. Augmentation crossing rate, measured (scripts/evaluation/eval_stir_hue_margin.py): glow hue
   clusters near 250 or 350 deg at every level, level means 2.6-4.3 deg apart against a
   within-level s.d. of 45-59 deg, 41 of 56 originals nearer another level's mean; copies move
   by median 1.4 deg (p90 16.8), 14 of 307 copies cross.
5. Abstract: "the class signal lives in hue" contradicts 4.

Usage: python review5_edits.py IN.tex OUT.tex [changes.json]
"""

import json
import sys

EDITS = [
    # ---- 1 ----
    ("and their export assigned frames to training and validation at random, so most validation frames",
     "and their export did not group frames by capture burst when assigning them to training and validation, so most validation frames"),
    # ---- 2 ----
    ("\\texttt{IR-Soil-Only} and \\texttt{UV-Strawberry-Pot} were not tested this way, and neither were the fine-tuned models.",
     "\\texttt{IR-Soil-Only} and \\texttt{UV-Strawberry-Pot} were not tested this way, and neither were the fine-tuned models, so the drops above are for COCO-initialized training and how much of the fine-tuned soil-only accuracy would survive a held-out burst is unmeasured."),
    # ---- 3 ----
    ("The regularized Stage~1 detector has not been put through this test.",
     "The regularized Stage~1 detector has not been put through this test. The regularized Stage~3 recipe has, on the cross-validation folds of Section~\\ref{sec:stir-cv}: at its final epoch none of its 26 errors in 168 held-out readings of originals (56 originals, three seeds) carries a confidence above 0.80, its mean confidence is 0.72 on correct readings and 0.53 on wrong ones, and with augmented copies included 2 of its 965 errors exceed 0.9."),
    # ---- 4 ----
    ("How often these transforms move a frame across a class boundary has not been measured.",
     "We measured the hue part of this on the stir-probe build, taking the circular mean hue of the brightest tenth of each annotated box. Unlike the progression of Table~\\ref{tab:led}, that hue clusters near $250^{\\circ}$ or near $350^{\\circ}$ at every level: the five level means lie $2.6^{\\circ}$ to $4.3^{\\circ}$ apart against a within-level circular s.d.\\ of $45^{\\circ}$ to $59^{\\circ}$, and 41 of the 56 originals already lie closer to another level's mean than to their own, consistent with the failure of the color baseline in Section~\\ref{sec:stir-cv}. In this dataset spot hue therefore does not separate the levels, and the hue cap protects a cue the levels barely differ in. Across the 1{,}101 augmented copies the spot's hue moves by a median of $1.4^{\\circ}$ (90th percentile $16.8^{\\circ}$), and 14 of the 307 copies whose original lies nearest its own level's mean are moved nearer another's. How often the saturation, brightness and size transforms move a frame across a class boundary has not been measured."),
    # ---- 5 ----
    ("First, the class signal lives in hue, so standard photometric augmentation destroys labels.",
     "First, the class is meant to be carried by the color of the LED, so standard photometric augmentation can corrupt labels."),
]
def apply(text):
    # In order: a later edit may target text an earlier one wrote.
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            sys.exit(f"expected 1 match, found {n}: {old[:60]!r}")
        text = text.replace(old, new)
    # The paper is in American spelling; earlier edits brought in "colour" and "neighbour".
    return text.replace("olour", "olor").replace("eighbour", "eighbor")


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        base = f.read()
    out = apply(base)
    with open(dst, "w") as f:
        f.write(out)
    if len(sys.argv) > 3:   # line-level changes as [from, to, insert] offsets into IN
        import difflib
        a, b = base.splitlines(keepends=True), out.splitlines(keepends=True)
        offs = [0]
        for line in a:
            offs.append(offs[-1] + len(line))
        changes = [[offs[i1], offs[i2], "".join(b[j1:j2])]
                   for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
                   if op != "equal"]
        with open(sys.argv[3], "w") as f:
            json.dump(changes, f)
