"""Paper edits: report the regularized Mixed Color runs as the Mixed Color runs.

Applied after stir_s1_retrain_edits.py. Outside the sweep section
(sec:stir-reg, Table VII) the paper no longer shows the unregularized
("original") Mixed Color and Background runs next to the new ones or labels the
new ones "retrained": Fig. 4 drops the dashed originals, Table IV keeps one
Mixed Color row per stage, and the overfitting narrative of the unregularized
runs (and the calibration check made on the unregularized Stage 1 detector)
moves into the sweep section.

Each entry is (old, new); every old string must occur exactly once.
"""

EDITS = [
    # Methods.
    (r"""Under this recipe the \texttt{Mixed Color and Background} baseline overfits (Section~\ref{sec:dynamics}); the Stage~1 model reported for it is therefore retrained with label smoothing of 0.1, AdamW weight decay of 0.05, an initial learning rate of $3 \times 10^{-3}$ rather than $10^{-2}$ and a 100-epoch cosine schedule, chosen by the sweep of Section~\ref{sec:stir-reg}.""",
     r"""For \texttt{Mixed Color and Background}, on which this recipe overfits, the Stage~1 run adds label smoothing of 0.1 and AdamW weight decay of 0.05 and uses an initial learning rate of $3 \times 10^{-3}$ rather than $10^{-2}$ on a 100-epoch cosine schedule, chosen by the sweep of Section~\ref{sec:stir-reg}."""),
    (r"""Under this recipe the \texttt{Mixed Color and Background} fine-tune overfits (Section~\ref{sec:dynamics}); the Stage~3 model reported for it is therefore retrained with label smoothing of 0.1, AdamW weight decay of 0.05, an initial learning rate of $3 \times 10^{-4}$ and a 60-epoch cosine schedule, chosen by the sweep of Section~\ref{sec:stir-reg}.""",
     r"""For \texttt{Mixed Color and Background}, on which this recipe overfits, the Stage~3 run adds label smoothing of 0.1 and AdamW weight decay of 0.05 and uses an initial learning rate of $3 \times 10^{-4}$ on a 60-epoch cosine schedule, chosen by the sweep of Section~\ref{sec:stir-reg}."""),

    # Overfitting section: all twelve runs are clean.
    (r"""Ten of the twelve runs show no overfitting (Fig.~\ref{fig:overfit}, Table~\ref{tab:overfit}). Nine of them reach""",
     r"""None of the twelve runs shows overfitting (Fig.~\ref{fig:overfit}, Table~\ref{tab:overfit}). Ten of them cover 200 epochs, and nine of those reach"""),
    (r"""\texttt{IR-Soil-Only} at Stage~1 is the loosest of the ten, ending 28\% above a minimum at epoch 159, though its trend over the post-warm-up window is still downward and its validation split shares capture bursts with training in any case.""",
     r"""\texttt{IR-Soil-Only} at Stage~1 is the loosest of the ten, ending 28\% above a minimum at epoch 159, though its trend over the post-warm-up window is still downward and its validation split shares capture bursts with training in any case. The two \texttt{Mixed Color and Background} runs, on budgets of 100 and 60 epochs, end 1.10 and 1.08 times their minima: validation loss falls throughout the Stage~1 budget, with validation $\text{mAP@0.5}$ still rising at its last epoch, and falls and then holds flat at Stage~3."""),
    (r"""mean the network does not see the same frame twice, and training classification loss for these runs settles between $8.5 \times 10^{-3}$ and $3.2 \times 10^{-2}$ rather than collapsing toward zero. Augmentation is doing the regularizing it was adopted to do.""",
     r"""mean the network does not see the same frame twice, and training classification loss for all twelve runs settles between $8.5 \times 10^{-3}$ and $3.2 \times 10^{-2}$ rather than collapsing toward zero. Augmentation is doing the regularizing it was adopted to do, except on \texttt{Mixed Color and Background}, where it is not enough on its own: without the label smoothing, weight decay and shorter, gentler schedule of Section~\ref{sec:stir-reg}, both of its runs overfit."""),
    (r"""The stir-probe runs are the exception, and they are the exception in both stages. Validation classification loss bottoms at epoch 54 in Stage~1 and epoch 57 in Stage~3, then rises over the remainder of the budget to end $1.61\times$ and $1.69\times$ above those minima. Training classification loss over the same span falls to $4.3 \times 10^{-4}$ at Stage~1, a factor of 20 below the lowest value reached by any other run in the study, and to $3.5 \times 10^{-3}$ at Stage~3, the lowest of that stage. Validation mAP follows: it peaks at epoch 101 in Stage~1 and epoch 32 in Stage~3 and decays thereafter, ending at 0.440 and 0.480 against peaks of 0.683 and 0.711. Three independent signatures, a rising validation loss, a collapsing training loss and a decaying validation mAP, agree, which is the classical picture of a network fitting its training frames. Section~\ref{sec:stir-reg} retrains both runs under explicit regularization, a lower learning rate and a shorter schedule, and the turn disappears in both.
""", ""),
    (r"""The collections with genuinely independent validation are the two pot-background captures, which show no overfitting, and the stir-probe capture, which does.""",
     r"""The collections with genuinely independent validation are the two pot-background captures and the stir-probe capture, and on the stir-probe capture the curve does detect overfitting when it is present (Section~\ref{sec:stir-reg})."""),
    # Calibration paragraph moves into the sweep section (see below).
    (r"""

As an independent post-training check that requires no retraining, we examined the calibration of each Stage~1 detector,""",
     r"""

_CALIB_START_ As an independent post-training check that requires no retraining, we examined the calibration of each Stage~1 detector,"""),

    # Fig. 4 caption.
    (r"""Markers give each run's lowest validation loss after epoch 50. Ten runs descend or flatten. The original stir-probe runs of both stages (dashed) turn upward while their training loss falls well below every other run. The retrained runs (solid; mean of three seeds, band spanning them) do not: their validation loss, rescored every five epochs without label smoothing so that it shares the scale of the other runs, falls throughout the 100-epoch Stage~1 budget and falls and then holds flat at Stage~3. Label smoothing also places a floor under their training loss, about 0.013 at Stage~1 and 0.025 at Stage~3.}""",
     r"""Markers give the lowest validation loss after epoch 50 of each 200-epoch run. All twelve runs descend or flatten. Mixed Color and Background (heavy line) is the mean of three seeds, with a band spanning them, trained on the recipes of Section~\ref{sec:stir-reg}; its validation loss is rescored every five epochs without label smoothing so that it shares the scale of the other runs, and label smoothing places a floor under its training loss, about 0.013 at Stage~1 and 0.025 at Stage~3.}"""),

    # Table IV: one Mixed Color row per stage.
    (r"""Mixed Color and Background & 54 & $\mathbf{1.61}$ & 0.683 & 0.440 \\
\quad retrained (3 seeds) & 75 & $1.10$ & 0.620 & 0.556 \\""",
     r"""Mixed Color and Background & 75 & $1.10$ & 0.620 & 0.556 \\"""),
    (r"""Mixed Color and Background & 57 & $\mathbf{1.69}$ & 0.711 & 0.480 \\
\quad retrained (3 seeds) & 34 & $1.08$ & 0.635 & 0.566 \\""",
     r"""Mixed Color and Background & 34 & $1.08$ & 0.635 & 0.566 \\"""),
    (r"""All runs cover 200 epochs except the original Stage~3 stir-probe run, which terminated at epoch 171, and the two retrained runs (Section~\ref{sec:stir-reg}), with budgets of 100 epochs at Stage~1 and 60 at Stage~3. For those the values are means over three seeds (final/min.\ 1.08 to 1.12 at Stage~1, 1.04 to 1.16 at Stage~3), the Stage~3 minimum is taken over all epochs, and validation loss is logged against label-smoothed targets, so only their ratios, not their levels, are comparable with the other rows.}""",
     r"""All runs cover 200 epochs except Mixed Color and Background (Section~\ref{sec:stir-reg}), with budgets of 100 epochs at Stage~1 and 60 at Stage~3. For it the values are means over three seeds (final/min.\ 1.08 to 1.12 at Stage~1, 1.04 to 1.16 at Stage~3), the Stage~3 minimum is taken over all epochs, and validation loss is logged against label-smoothed targets, so only its ratio, not its level, is comparable with the other rows.}"""),

    # Baseline Performance text, Table V, Fig. 5.
    (r"""The original stir probe baseline of 0.683 exceeded the 0.5 detection threshold despite only 39 original training frames (858 after expansion), but it is a pre-collapse peak on a sparse validation split (Section~\ref{sec:dynamics}). The retrained baseline of Section~\ref{sec:stir-reg}, which Table~\ref{tab:baseline} reports, reaches $0.620 \pm 0.042$ and does not collapse.""",
     r"""The stir probe baseline of $0.620 \pm 0.042$ exceeded the 0.5 detection threshold despite only 39 original training frames (858 after expansion), and it was still rising at the end of its 100-epoch budget; with 12 validation originals it is a noisy estimate."""),
    (r"""$^{a}$Retrained Stage~1 run of Section~\ref{sec:stir-reg}, mean over three seeds of each seed's best epoch (standard deviation 0.04). It replaces the original run's pre-collapse peak of 0.68 (mAP@0.5:0.95 0.24), which ended at 0.44.}""",
     r"""$^{a}$Mean over three seeds of each seed's best epoch (standard deviation 0.04), recipe of Section~\ref{sec:stir-reg}.}"""),
    (r"""The Mixed Color and Background bar is the retrained run of Section~\ref{sec:stir-reg}, the mean of its three seeds' peaks with their standard deviation; the original run's pre-collapse peak was 0.683.}""",
     r"""The Mixed Color and Background bar is the mean of its three seeds' peaks with their standard deviation (Section~\ref{sec:stir-reg}).}"""),

    # Table VI and Fig. 6.
    (r"""$^{a}$Retrained Stage~1 and Stage~3 runs of Section~\ref{sec:stir-reg}, means of the three seeds' peaks (standard deviation 0.04 and 0.03). They replace the original runs' pre-collapse peaks of 0.68 and 0.71, which ended at 0.44 and 0.48; on the held-out test split the retrained runs score 0.673 and 0.741 against 0.577 and 0.540.}""",
     r"""$^{a}$Means of the three seeds' peaks (standard deviation 0.04 and 0.03), recipes of Section~\ref{sec:stir-reg}; on the held-out test split the two runs score 0.673 and 0.741.}"""),
    (r"""The Mixed Color and Background bar is the retrained run of Section~\ref{sec:stir-reg}, the mean of its three seeds' peaks with their standard deviation; the original run's pre-collapse peak was 0.711.}""",
     r"""The Mixed Color and Background bar is the mean of its three seeds' peaks with their standard deviation (Section~\ref{sec:stir-reg}).}"""),

    # Augmentation subsections: these describe the unregularized runs.
    (r"""The original Stage~1 and Stage~3 results of 0.683 and 0.711 for Mixed Color and Background were trained on the expanded set of Table~\ref{tab:aug}, 858 training and 204 validation frames from 39 and 12 originals. A further run (stir5) on the same expanded set reached $\text{mAP@0.5}=0.625$. It differs from the 0.711 run in its starting point,""",
     r"""Every Mixed Color and Background run in this paper was trained on the expanded set of Table~\ref{tab:aug}, 858 training and 204 validation frames from 39 and 12 originals. A further run (stir5) on the same expanded set reached $\text{mAP@0.5}=0.625$. It differs from the unregularized Stage~3 run of Section~\ref{sec:stir-reg} (0.711) in its starting point,"""),
    (r"""That the one overfitting collection is also the most heavily augmented one is not a contradiction""",
     r"""That the one collection on which the unregularized recipes overfit (Section~\ref{sec:stir-reg}) is also the most heavily augmented one is not a contradiction"""),
    (r"""What the curves establish is the fact of memorization and its timing; which incidental property is being memorized is not settled by them, and the held-out 95-frame test split is too small to settle it. Scored at $\text{mAP@0.5}$, it gives 0.552 for the Stage~3 best-fitness checkpoint and 0.540 for the final one,""",
     r"""What the unregularized curves establish is the fact of memorization and its timing; which incidental property is being memorized is not settled by them, and the held-out 95-frame test split is too small to settle it. Scored at $\text{mAP@0.5}$, it gives 0.552 for the unregularized Stage~3 run's best-fitness checkpoint and 0.540 for its final one,"""),
    (r"""First, the stir-probe figure of 0.683 in Table~\ref{tab:baseline} and the original Stage~3 peak of 0.711 are peak-epoch values read from the validation log rather than converged accuracies, and should be read together with the 0.440 and 0.480 the same runs end at. No saved checkpoint corresponds to either peak, because YOLOv5 retains the checkpoint of best combined fitness (epochs 133 and 70) rather than of best $\text{mAP@0.5}$. The retrained runs of Section~\ref{sec:stir-reg}, and for Stage~3 the cross-validated estimate of Section~\ref{sec:stir-cv}, supersede both. Second, the remaining eleven runs""",
     r"""First, the unregularized runs' peaks of 0.683 and 0.711 are peak-epoch values read from the validation log rather than converged accuracies; the same runs end at 0.440 and 0.480, and no saved checkpoint corresponds to either peak, because YOLOv5 retains the checkpoint of best combined fitness (epochs 133 and 70) rather than of best $\text{mAP@0.5}$. The tables therefore report the regularized runs of Section~\ref{sec:stir-reg}, and for Stage~3 the cross-validated estimate of Section~\ref{sec:stir-cv}. Second, the other ten runs"""),

    # Sweep section.
    (r"""\subsection{Regularized Retraining of the Stir-Probe Runs}""",
     r"""\subsection{Regularized Recipes for the Stir-Probe Runs}"""),
    (r"""The fine-tuning recipe of the original Stage~3 run carries almost no explicit regularization: AdamW with a decoupled weight decay of $5 \times 10^{-4}$, no label smoothing, and a 200-epoch cosine schedule. We retrained the run from the same""",
     r"""Under the unregularized recipes of Section~\ref{sec:method}, the original Mixed Color and Background runs overfit in both stages. Validation classification loss bottoms at epoch 54 in Stage~1 and epoch 57 in Stage~3, then rises over the remainder of the budget to end $1.61\times$ and $1.69\times$ above those minima. Training classification loss over the same span falls to $4.3 \times 10^{-4}$ at Stage~1, a factor of 20 below the lowest value reached by any other run in the study, and to $3.5 \times 10^{-3}$ at Stage~3, the lowest of that stage. Validation mAP follows: it peaks at epoch 101 in Stage~1 and epoch 32 in Stage~3 and decays thereafter, ending at 0.440 and 0.480 against peaks of 0.683 and 0.711. Three independent signatures, a rising validation loss, a collapsing training loss and a decaying validation mAP, agree, which is the classical picture of a network fitting its training frames.

The fine-tuning recipe of the original Stage~3 run carries almost no explicit regularization: AdamW with a decoupled weight decay of $5 \times 10^{-4}$, no label smoothing, and a 200-epoch cosine schedule. We trained the run again from the same"""),
    (r"""This recipe is the Stage~1 model reported in Tables~\ref{tab:baseline}, \ref{tab:best} and~\ref{tab:overfit} and Fig.~\ref{fig:baseline}; it has not been cross-validated.""",
     r"""This recipe is the Stage~1 model reported in Tables~\ref{tab:baseline}, \ref{tab:best} and~\ref{tab:overfit} and Fig.~\ref{fig:baseline}; it has not been cross-validated.

_CALIB_HERE_"""),

    # Cross-validation section.
    (r"""Two arms were trained on every fold with three seeds each: the Stage~3 recipe of Table~\ref{tab:best} unchanged, initialized from \texttt{IR-Pot-Background} with the same hyperparameters, the same recipe with the ten backbone layers frozen, and the retrained recipe of Section~\ref{sec:stir-reg} on its own 60-epoch schedule.""",
     r"""Three arms were trained on every fold with three seeds each: the unregularized Stage~3 recipe, initialized from \texttt{IR-Pot-Background} with the fine-tuning hyperparameters, the same recipe with the ten backbone layers frozen, and the regularized recipe of Section~\ref{sec:stir-reg} on its own 60-epoch schedule."""),
    (r"""by the top-1 protocol of Section~\ref{sec:dynamics}, giving""",
     r"""by the top-1 protocol of Section~\ref{sec:stir-reg}, giving"""),
    (r"""Retrained & 0.35 & 0.72 & 0.83 & 0.85 & 0.760 & 0.081 \\""",
     r"""Regularized & 0.35 & 0.72 & 0.83 & 0.85 & 0.760 & 0.081 \\"""),
    (r"""Final is epoch 99, or epoch 59 for the retrained arm.""",
     r"""Final is epoch 99, or epoch 59 for the regularized arm."""),
    (r"""The cross-validated recipe therefore does not reproduce the collapse of the shipped run,""",
     r"""The cross-validated unregularized recipe therefore does not reproduce the collapse of the shipped unregularized run (Section~\ref{sec:stir-reg}),"""),
    (r"""Two weaknesses of the unregularized recipe remain, and the retrained recipe narrows both.""",
     r"""Two weaknesses of the unregularized recipe remain, and the regularized recipe narrows both."""),
    (r"""The retrained recipe matches the unregularized one on independent scenes:""",
     r"""The regularized recipe matches the unregularized one on independent scenes:"""),

    # Confusion matrix paragraph: describes the unregularized Stage 1 detector.
    (r"""\emph{Baseline Mixed Color and Background (original run).} Off diagonal""",
     r"""\emph{Baseline Mixed Color and Background, unregularized recipe (Section~\ref{sec:stir-reg}).} Off diagonal"""),
    (r"""level 1 is instead the weakest level at 0.50 and is most often read as level 8, so this ordering""",
     r"""level 1 is instead the weakest level at 0.50 (0.71 under the regularized recipe) and is most often read as level 8, so this ordering"""),
]

CALIB_EDITS = [
    # Applied to the moved calibration paragraph.
    (r"""we examined the calibration of each Stage~1 detector,""",
     r"""we examined the calibration of each Stage~1 detector as first trained, the Mixed Color and Background one under the unregularized recipe,"""),
    (r"""The original stir-probe detector is the only one to commit high-confidence errors""",
     r"""The unregularized stir-probe detector is the only one to commit high-confidence errors"""),
    (r"""tracking the post-epoch-54 collapse in Table~\ref{tab:overfit},""",
     r"""tracking its post-epoch-54 collapse,"""),
    (r"""singling out the same collection and the same checkpoint, rather than adding evidence for the remaining runs.""",
     r"""singling out the same collection and the same checkpoint, rather than adding evidence for the remaining runs. The regularized Stage~1 detector has not been put through this test."""),
]


def apply(text):
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"{n} matches for: {old[:70]!r}")
        text = text.replace(old, new)
    # Move the calibration paragraph (it ends at the next blank line).
    start = text.index("_CALIB_START_ ")
    end = text.index("\n\n", start)
    para = text[start + len("_CALIB_START_ "):end]
    text = text[:start].rstrip("\n") + text[end:]
    for old, new in CALIB_EDITS:
        if para.count(old) != 1:
            raise ValueError(f"calibration: {old[:60]!r}")
        para = para.replace(old, new)
    assert text.count("_CALIB_HERE_") == 1
    text = text.replace("_CALIB_HERE_", para)
    assert "_CALIB_" not in text
    return text


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "IEEE_paper_edited_s1_20261014.tex"
    out = apply(open(src).read())
    print(f"all {len(EDITS)} + {len(CALIB_EDITS)} edits match")
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out)
