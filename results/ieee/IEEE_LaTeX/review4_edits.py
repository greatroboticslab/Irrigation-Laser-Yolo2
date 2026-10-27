"""Fixes from the 2026-10-26 external review that do not depend on what the stir-probe boxes
depict (that question is open: see scripts/evaluation/eval_stir_hue_margin.py). Applies on top
of review3_edits.py.

1. Dry read as wet: in the regularized arm's out-of-fold predictions (results/stir_cv_oof_reg.csv,
   epoch 59, 8 level-1 originals x 3 seeds), level 1 is read as 8 six times and as 10 once.
2. Surrounding soil named as an untested alternative cue for the detector.
3. The fixed-model inpainting test feeds out-of-distribution frames; the drop is not a measure
   of reliance.
4. The card-masking test covered one model, at inference only.
5. Occlusion by a growing canopy is untested.

Usage: python review4_edits.py IN.tex OUT.tex [changes.json]
"""

import json
import sys

EDITS = [
    # ---- 1 ----
    ("level~1 read as 8, level~8 read as 1, and level~3 read as 10.",
     "level~1 read as 8, level~8 read as 1, and level~3 read as 10. The most frequent of these is the costliest for irrigation: over the three seeds, 7 of the 24 readings of level~1, the driest level, come out as level~8 or~10, so a controller acting on them would withhold water from the driest pots."),
    # ---- 2 ----
    ("though which cues it uses is not established here.",
     "though which cues it uses is not established here. One candidate is the soil around the box: wetter soil is darker, the detector's receptive field covers that soil, and a model can learn the level from it rather than from the spot. Re-scoring the fold models with everything outside the box replaced by a uniform fill would separate the two, and has not been done."),
    # ---- 3 ----
    ("The sham result establishes that this is a response to the missing return rather than to the editing artifact.",
     "The sham result establishes that this is a response to the missing return rather than to the editing artifact. It does not measure how much a model needs the return: the inpainted frames lie outside the distribution the fixed models were trained on, so part of the drop reflects that shift, and the retraining arm below is the test of reliance."),
    # ---- 4 ----
    ("masking the handwritten moisture cards visible in the lab collections leaves the headline model unchanged at 0.995.",
     "masking the handwritten moisture cards visible in the lab collections leaves the headline model unchanged at 0.995. The card test covered only that model, and only at inference; the other models were not scored with the cards masked, and no model was retrained without them."),
    # ---- 5 ----
    ("and reading it as a property of the sensing modality will invert the moment domain-specific weights are introduced.\n",
     "and reading it as a property of the sensing modality will invert the moment domain-specific weights are introduced.\n\nA deployment limit follows from the scenes themselves. Every frame was taken with a clear line of sight from the overhead camera to the soil at the annotated spot, with foliage only at the edge of the pot. As a canopy grows, leaves will hide that spot from an overhead view, and no frame in these collections tests reading under occlusion.\n"),
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
