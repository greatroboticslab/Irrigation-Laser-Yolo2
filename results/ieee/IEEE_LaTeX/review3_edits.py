"""Fixes from the 2026-10-23 weakness review (items 2, 4 and 5). Applies on top of review2_edits.py.

2. Soil-only results are in-session. New Section "Cross-Burst Validation of the Soil-Only
   Collections" with the matched cross-burst / within-burst runs of 2026-09-19 (tr_cv.py:
   COCO init, default hyp, 200 epochs, 2 folds x 3 seeds, best-epoch val mAP@0.5):
   UV-Soil-Only-Baseline 0.9937 +- 0.0033 vs 0.4291 +- 0.0389 (10 levels), UV-Soil-Only-Target
   0.9484 +- 0.0814 vs 0.7240 +- 0.0714 (6 levels). Burst counts and validation overlap from
   runs/cv/*_groups.csv. Burst-overlapped rows of Tables V, VI and IX get a dagger.
4. Colour baseline on the stir CV folds (scripts/evaluation/eval_stir_color_baseline.py,
   results/stir_color_baseline.csv): oracle box, kNN 0.339, nearest centroid 0.018, against
   0.357 for always answering level 10; kNN scores 0.982 on training originals. A second
   feature set (weighted hue histogram) gave 0.304 / 0.071.
5. Reading error for the stir detector (scripts/evaluation/eval_stir_ordinal.py,
   results/stir_cv_ordinal.csv), and the regularization sweep is stated to have used the
   test split.

Usage: python review3_edits.py IN.tex OUT.tex [edits.json]
"""

import json
import sys

D = "$^{\\dagger}$"
DAGGER_NOTE = "$^{\\dagger}$Validation frames share capture bursts with training (Section~\\ref{sec:burst-cv})."

EDITS = [
    # ---- 2: burst overlap ----
    ("on their validation splits (which, for the soil-only collections, share capture bursts with training),",
     "on their validation splits (which, for the soil-only collections, share capture bursts with training; validating across bursts costs those collections 0.22 to 0.57 mAP@0.5),"),
    ("and a cross-burst experiment rather than the curve is what speaks to them.",
     "and a cross-burst experiment rather than the curve is what speaks to them (Section~\\ref{sec:burst-cv})."),
    # Table V
    ("UV-Soil-Only-Baseline & 0.76 & 0.50 \\\\\nUV-Strawberry-Pot & 0.72 & 0.37 \\\\",
     f"UV-Soil-Only-Baseline{D} & 0.76 & 0.50 \\\\\nUV-Strawberry-Pot{D} & 0.72 & 0.37 \\\\"),
    ("UV-Soil-Only-Target & 0.61 & 0.39 \\\\\nUV-Pot-Background & 0.59 & 0.38 \\\\\nIR-Soil-Only & 0.55 & 0.19 \\\\",
     f"UV-Soil-Only-Target{D} & 0.61 & 0.39 \\\\\nUV-Pot-Background & 0.59 & 0.38 \\\\\nIR-Soil-Only{D} & 0.55 & 0.19 \\\\"),
    ("$^{a}$Mean over three seeds of each seed's best epoch (standard deviation 0.04), recipe of Section~\\ref{sec:stir-reg}.}",
     f"$^{{a}}$Mean over three seeds of each seed's best epoch (standard deviation 0.04), recipe of Section~\\ref{{sec:stir-reg}}. {DAGGER_NOTE}}}"),
    # Table VI
    ("UV-Soil-Only-Baseline & 0.76 & 0.94 & 0.96 & 0.96 \\\\\nIR-Soil-Only & 0.55 & --- & 0.99 & 0.99 \\\\\nUV-Soil-Only-Target & 0.61 & --- & 0.91 & 0.91 \\\\",
     f"UV-Soil-Only-Baseline{D} & 0.76 & 0.94 & 0.96 & 0.96 \\\\\nIR-Soil-Only{D} & 0.55 & --- & 0.99 & 0.99 \\\\\nUV-Soil-Only-Target{D} & 0.61 & --- & 0.91 & 0.91 \\\\"),
    ("UV-Strawberry-Pot & 0.72 & 0.94 & 0.96 & 0.96 \\\\",
     f"UV-Strawberry-Pot{D} & 0.72 & 0.94 & 0.96 & 0.96 \\\\"),
    ("recipes of Section~\\ref{sec:stir-reg}; on the held-out test split the two runs score 0.673 and 0.741.",
     f"recipes of Section~\\ref{{sec:stir-reg}}; on the held-out test split, which was used in choosing those recipes, the two runs score 0.673 and 0.741. {DAGGER_NOTE}"),
    # Table IX
    ("UV-Soil-Only-Baseline & 246 & 11 & 0.76 & 0.96 & 0.70 \\\\\nUV-Soil-Only-Target & 142 & 11 & 0.61 & 0.91 & 0.69 \\\\\nIR-Soil-Only & 104 & 7 & 0.55 & 0.99 & 0.59 \\\\",
     f"UV-Soil-Only-Baseline{D} & 246 & 11 & 0.76 & 0.96 & 0.70 \\\\\nUV-Soil-Only-Target{D} & 142 & 11 & 0.61 & 0.91 & 0.69 \\\\\nIR-Soil-Only{D} & 104 & 7 & 0.55 & 0.99 & 0.59 \\\\"),
    ("its Best columns are those of the UV-Combined pool it belongs to.}",
     f"its Best columns are those of the UV-Combined pool it belongs to. {DAGGER_NOTE}}}"),
    ("(Section~\\ref{sec:dynamics}), so these figures are upper bounds.",
     "(Section~\\ref{sec:burst-cv}), so these figures are in-session upper bounds; validating across bursts costs the two ultraviolet captures 0.565 and 0.224 mAP@0.5 from COCO weights."),
    ("and the only systematic difference between them is 22.4 against 12.9 training images per level. Whatever",
     "and the only systematic difference between them is 22.4 against 12.9 training images per level, or 3.9 against 2.0 capture bursts per level when only independent captures are counted. Whatever"),
    # new subsection, before the illumination study
    ("costs no measurable accuracy.\n\n\\subsection{Illumination and Training Volume}\n",
     "costs no measurable accuracy.\n\n"
     "@@READING@@"
     "\\subsection{Cross-Burst Validation of the Soil-Only Collections}\n\\label{sec:burst-cv}\n"
     "The soil-only collections were photographed in long runs at a fixed station, and their export assigned frames to training and validation at random, so most validation frames have a near-duplicate in training (Section~\\ref{sec:dynamics}). Grouping every frame into capture bursts by pixel similarity (mean squared error at $128 \\times 128$~px, with the cut placed where the burst count is most stable across thresholds) gives 43 bursts for the 351 frames of \\texttt{UV-Soil-Only-Baseline}, 21 for the 202 of \\texttt{UV-Soil-Only-Target}, 20 for the 149 of \\texttt{IR-Soil-Only} and 31 for the 68 of \\texttt{UV-Strawberry-Pot}, against 99 for each 100-frame pot-background collection. In the shipped splits, 69 of 70, 40 of 40, 30 of 30 and 8 of 11 validation frames of the first four share a burst with training, and none of the pot-background collections' do.\n\n"
     "To measure what that overlap is worth, we trained two matched arms from COCO weights with default YOLOv5 hyperparameters for 200 epochs, three seeds per fold. The cross-burst arm holds out half of every level's bursts, so that no validation frame has a burst sibling in training; the within-burst arm holds out the same number of frames from the same levels at random, which reproduces the shipped condition. Several levels were captured in only two bursts, so two folds is the most that keeps every level in validation, and levels captured in a single burst cannot be held out and are dropped, leaving 10 of 11 levels for \\texttt{UV-Soil-Only-Baseline} and 6 of 11 for \\texttt{UV-Soil-Only-Target}. Absolute values are therefore not comparable with the 11-level tables; the contrast between the arms is.\n\n"
     "\\begin{table}[!t]\n\\renewcommand{\\arraystretch}{1.3}\n\\caption{Cross-Burst Versus Within-Burst Validation, mAP@0.5}\n\\label{tab:burstcv}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{l c c c}\n\\toprule\n\\textbf{Dataset (levels)} & \\textbf{Within-burst} & \\textbf{Cross-burst} & \\textbf{$\\Delta$} \\\\\n\\midrule\nUV-Soil-Only-Baseline (10) & $0.994 \\pm 0.003$ & $0.429 \\pm 0.039$ & $-0.565$ \\\\\nUV-Soil-Only-Target (6) & $0.948 \\pm 0.081$ & $0.724 \\pm 0.071$ & $-0.224$ \\\\\n\\bottomrule\n\\end{tabular}\n\\\\[2pt]\n\\parbox{\\columnwidth}{\\footnotesize Mean $\\pm$ s.d.\\ over 2 folds $\\times$ 3 seeds of the best validation epoch, COCO initialization. Within-burst draws validation frames at random, as the shipped splits do; cross-burst holds out whole bursts.}\n\\end{table}\n\n"
     "Burst overlap accounts for most of the accuracy these collections report (Table~\\ref{tab:burstcv}). Holding out whole bursts costs \\texttt{UV-Soil-Only-Baseline} 0.565 mAP@0.5 (paired $t(5) = 36.7$, 95\\% CI 0.525 to 0.604) and \\texttt{UV-Soil-Only-Target} 0.224 ($t(5) = 5.8$, 95\\% CI 0.125 to 0.323), all twelve paired comparisons run in the same direction, and the two folds of each dataset agree to within 0.03. \\texttt{IR-Soil-Only} and \\texttt{UV-Strawberry-Pot} were not tested this way, and neither were the fine-tuned models. The soil-only figures in Tables~\\ref{tab:baseline}, \\ref{tab:best} and~\\ref{tab:modality}, marked $^{\\dagger}$, are therefore in-session accuracies: they show that the detector reads the probe within a capture session it has seen, not that it transfers to a new one. The pot-background collections, and with them the IR-Combined result of 0.995, have no burst overlap to remove, and the stir-probe set is cross-validated over bursts in Section~\\ref{sec:stir-cv}.\n\n"
     "\\subsection{Illumination and Training Volume}\n"),

    # ---- 4 and 5: reading error, colour baseline, test-split use ----
    ("@@READING@@",
     "\\emph{Reading error and a colour baseline.} Top-1 accuracy treats every wrong reading alike, but for irrigation a dry pot read as wet is worse than a neighbouring level. Table~\\ref{tab:stirread} therefore adds, from the same out-of-fold predictions, the mean absolute error in moisture levels and the share of originals read three or more steps away on the five-level scale $\\{1, 3, 7, 8, 10\\}$. The regularized detector errs by 0.84 levels on average (s.d.\\ 0.37 over seeds), and 9.5\\% of originals are read three or more steps away: level~1 read as 8, level~8 read as 1, and level~3 read as 10.\n\n"
     "To test whether the detector adds anything over the colour of the LED, we classified each original from the colour statistics of its annotated box alone: the circular mean hue, mean saturation, value and RGB of the brightest tenth of the box's pixels, the spot's extent within the box, and the box size, with $k$-nearest-neighbour ($k = 5$) and nearest-centroid classifiers trained on each fold's training frames, augmented copies included. The baseline is given the true box and so needs no localization, whereas the detector's top-1 accuracy also requires IoU~$\\geq 0.5$. It nonetheless reaches only 0.34 and 0.02, against 0.36 for always answering level~10, the most frequent level, while the same $k$-nearest-neighbour classifier scores 0.98 on the training originals. A second feature set, a saturation- and value-weighted hue histogram of the whole box, gave 0.30 and 0.07. Colour within the box varies more between capture bursts than between moisture levels, so a colour rule fitted to some bursts does not transfer to others. The detector's 0.85 on the same held-out bursts shows that it uses more than the box's colour statistics, though which cues it uses is not established here.\n\n"
     "\\begin{table}[!t]\n\\renewcommand{\\arraystretch}{1.3}\n\\caption{Stir-Probe Reading Error Under Grouped Cross-Validation}\n\\label{tab:stirread}\n\\centering\n\\footnotesize\n\\setlength{\\tabcolsep}{3.5pt}\n\\begin{tabular}{l c c c}\n\\toprule\n\\textbf{Method} & \\textbf{Top-1} & \\textbf{Level MAE} & \\textbf{$\\geq 3$ steps} \\\\\n\\midrule\nDetector, full fine-tune & 0.83 & 0.83 & 0.11 \\\\\nDetector, regularized & 0.85 & 0.84 & 0.10 \\\\\nDetector, frozen backbone & 0.31 & 3.31 & 0.34 \\\\\nColour, $k$-NN (true box) & 0.34 & 3.70 & 0.34 \\\\\nColour, nearest centroid (true box) & 0.02 & 4.36 & 0.25 \\\\\nAlways level 10 & 0.36 & 3.50 & 0.36 \\\\\n\\bottomrule\n\\end{tabular}\n\\\\[2pt]\n\\parbox{\\columnwidth}{\\footnotesize Out-of-fold over the 56 originals, five burst-grouped folds; detector rows are means of three seeds at the final epoch. Level MAE is the mean absolute difference between read and true level; $\\geq 3$ steps is the share of originals read three or more steps away on the scale $\\{1, 3, 7, 8, 10\\}$. Detector top-1 requires IoU~$\\geq 0.5$; its error columns use the class of the top detection.}\n\\end{table}\n\n"),
    ("Its held-out test $\\text{mAP@0.5}$ is $0.741 \\pm 0.011$ against 0.540.",
     "Its held-out test $\\text{mAP@0.5}$ is $0.741 \\pm 0.011$ against 0.540. Test $\\text{mAP@0.5}$ was computed for every recipe and compared alongside the loss ratio when the recipe was chosen, so the test split is not untouched and this figure is optimistic; the grouped cross-validation of Section~\\ref{sec:stir-cv} is the estimate that does not reuse it."),
    ("Rows other than the originals are means of three seeds",
     "Recipes were compared on this column as well as on the ratio, so the bold test figures are optimistic. Rows other than the originals are means of three seeds"),
    ("the primary stir-probe task reaches 0.64 on validation and 0.74 on its test split, and the soil-only validation splits share capture bursts with training, so testing",
     "the primary stir-probe task reaches 0.64 on validation and 0.74 on a test split that was used in choosing its recipe, and under grouped cross-validation it reads 85\\% of independent scenes correctly with a mean error of 0.84 levels. The soil-only validation splits share capture bursts with training, which accounts for 0.22 to 0.57 of their mAP@0.5, so testing"),
]


def apply(text):
    # In order: a later edit may target text an earlier one wrote (@@READING@@).
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
