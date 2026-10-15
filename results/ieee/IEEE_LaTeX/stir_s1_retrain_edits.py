"""Paper edits for the retrained Stage 1 Mixed Color and Background run.

Applied after stir_retrain_edits.py (Stage 3). Each entry is (old, new); every
old string must occur exactly once in the live IEEE_paper.tex. apply() checks
that and returns the edited text. The same pairs are applied to the Overleaf
editor through the browser.

Numbers come from:
  results/stir_reg_round3                              (sweep, shipped split)
  results/stir_reg_round3/s1_ls_wd_lr3_e100/seed*_results.csv  (logged curves)
  results/runs/train/downloaded/soil_moisture_stir_september/results.csv (original)
"""

EDITS = [
    # 1. Methods, Stage 1.
    (r"""other datasets use default YOLOv5 hyperparameters.""",
     r"""other datasets use default YOLOv5 hyperparameters. Under this recipe the \texttt{Mixed Color and Background} baseline overfits (Section~\ref{sec:dynamics}); the Stage~1 model reported for it is therefore retrained with label smoothing of 0.1, AdamW weight decay of 0.05, an initial learning rate of $3 \times 10^{-3}$ rather than $10^{-2}$ and a 100-epoch cosine schedule, chosen by the sweep of Section~\ref{sec:stir-reg}."""),

    # 2. Fig. 4 caption.
    (r"""The Stage~1 stir-probe run turns upward while its training loss falls well below every other run, and so does the original Stage~3 stir-probe run (dashed). The retrained Stage~3 run (solid; mean of three seeds, band spanning them) does not: its validation loss, rescored every five epochs without label smoothing so that it shares the scale of the other runs, falls and stays flat. Label smoothing also places a floor of about 0.025 under its training loss.}""",
     r"""The original stir-probe runs of both stages (dashed) turn upward while their training loss falls well below every other run. The retrained runs (solid; mean of three seeds, band spanning them) do not: their validation loss, rescored every five epochs without label smoothing so that it shares the scale of the other runs, falls throughout the 100-epoch Stage~1 budget and falls and then holds flat at Stage~3. Label smoothing also places a floor under their training loss, about 0.013 at Stage~1 and 0.025 at Stage~3.}"""),

    # 3. Table IV: add the Stage 1 retrained row; rewrite the footnote for two.
    (r"""Mixed Color and Background & 54 & $\mathbf{1.61}$ & 0.683 & 0.440 \\""",
     r"""Mixed Color and Background & 54 & $\mathbf{1.61}$ & 0.683 & 0.440 \\
\quad retrained (3 seeds) & 75 & $1.10$ & 0.620 & 0.556 \\"""),
    (r"""and its retrained counterpart (Section~\ref{sec:stir-reg}), which has a 60-epoch budget; for it the minimum is taken over all epochs, the values are means over three seeds (final/min.\ 1.04 to 1.16), and validation loss is logged against label-smoothed targets, so only its ratio, not its level, is comparable with the other rows.}""",
     r"""and the two retrained runs (Section~\ref{sec:stir-reg}), with budgets of 100 epochs at Stage~1 and 60 at Stage~3. For those the values are means over three seeds (final/min.\ 1.08 to 1.12 at Stage~1, 1.04 to 1.16 at Stage~3), the Stage~3 minimum is taken over all epochs, and validation loss is logged against label-smoothed targets, so only their ratios, not their levels, are comparable with the other rows.}"""),

    # 4. End of the overfitting paragraph.
    (r"""Section~\ref{sec:stir-reg} retrains the Stage~3 run under explicit regularization and a shorter schedule, and the turn disappears.""",
     r"""Section~\ref{sec:stir-reg} retrains both runs under explicit regularization, a lower learning rate and a shorter schedule, and the turn disappears in both."""),

    # 5. Calibration paragraph describes the original run.
    (r"""The stir-probe detector is the only one to commit high-confidence errors""",
     r"""The original stir-probe detector is the only one to commit high-confidence errors"""),

    # 6. Baseline Performance text.
    (r"""The stir probe baseline of 0.683 exceeded the 0.5 detection threshold despite only 39 original training frames (858 after expansion), but it represents a noisy estimate from sparse validation, consistent with the volatility described above.""",
     r"""The original stir probe baseline of 0.683 exceeded the 0.5 detection threshold despite only 39 original training frames (858 after expansion), but it is a pre-collapse peak on a sparse validation split (Section~\ref{sec:dynamics}). The retrained baseline of Section~\ref{sec:stir-reg}, which Table~\ref{tab:baseline} reports, reaches $0.620 \pm 0.042$ and does not collapse."""),

    # 7. Table tab:baseline row + footnote.
    (r"""Mixed Color and Background & 0.68 & 0.24 \\""",
     r"""Mixed Color and Background & 0.62$^{a}$ & 0.28 \\"""),
    (r"""IR-Soil-Only & 0.54 & 0.19 \\
\bottomrule
\end{tabular}""",
     r"""IR-Soil-Only & 0.54 & 0.19 \\
\bottomrule
\end{tabular}
\\[2pt]
\parbox{\columnwidth}{\footnotesize $^{a}$Retrained Stage~1 run of Section~\ref{sec:stir-reg}, mean over three seeds of each seed's best epoch (standard deviation 0.04). It replaces the original run's pre-collapse peak of 0.68 (mAP@0.5:0.95 0.24), which ended at 0.44.}"""),

    # 8. Fig. 5 caption.
    (r"""Bars are the best epoch, the ``mAP peak'' column of Table~\ref{tab:overfit}.}
\label{fig:baseline}""",
     r"""Bars are the best epoch, the ``mAP peak'' column of Table~\ref{tab:overfit}. The Mixed Color and Background bar is the retrained run of Section~\ref{sec:stir-reg}, the mean of its three seeds' peaks with their standard deviation; the original run's pre-collapse peak was 0.683.}
\label{fig:baseline}"""),

    # 9. Table tab:best, Run 1 and Best columns for Mixed Color; footnote.
    (r"""Mixed Color and Background & 0.68 & 0.00 & 0.64$^{a}$ & 0.68 \\""",
     r"""Mixed Color and Background & 0.62$^{a}$ & 0.00 & 0.64$^{a}$ & 0.64 \\"""),
    (r"""$^{a}$Retrained Stage~3 run of Section~\ref{sec:stir-reg}, mean of the three seeds' peaks (standard deviation 0.03). It replaces the original run's pre-collapse peak of 0.71, which ended at 0.48; on the held-out test split the retrained run scores 0.741 against 0.540.}""",
     r"""$^{a}$Retrained Stage~1 and Stage~3 runs of Section~\ref{sec:stir-reg}, means of the three seeds' peaks (standard deviation 0.04 and 0.03). They replace the original runs' pre-collapse peaks of 0.68 and 0.71, which ended at 0.44 and 0.48; on the held-out test split the retrained runs score 0.673 and 0.741 against 0.577 and 0.540.}"""),

    # 10. Section title, sweep table, Stage 1 paragraph.
    (r"""\subsection{Regularized Retraining of the Stage~3 Stir-Probe Run}""",
     r"""\subsection{Regularized Retraining of the Stir-Probe Runs}"""),
    (r"""\caption{Regularization Sweep for the Stage~3 Stir-Probe Run}""",
     r"""\caption{Regularization Sweep for the Stir-Probe Runs}"""),
    (r"""\textbf{Recipe} & \textbf{Epochs} & \textbf{Final/min.} & \textbf{Test mAP@0.5} \\
\midrule
Original & 200 & 2.47 & 0.540 \\""",
     r"""\textbf{Recipe} & \textbf{Epochs} & \textbf{Final/min.} & \textbf{Test mAP@0.5} \\
\midrule
\multicolumn{4}{l}{\emph{Stage 3 (lr0 $10^{-3}$ unless stated)}} \\
Original & 200 & 2.47 & 0.540 \\"""),
    (r"""\quad same & 60 & $\mathbf{1.05}$ & $\mathbf{0.741}$ \\
\bottomrule""",
     r"""\quad same & 60 & $\mathbf{1.05}$ & $\mathbf{0.741}$ \\
\midrule
\multicolumn{4}{l}{\emph{Stage 1 (lr0 $10^{-2}$ unless stated)}} \\
Original & 200 & 1.66 & 0.577 \\
LS + weight decay & 200 & 1.36 & 0.595 \\
LS + weight decay + lr0 $10^{-3}$ & 200 & 1.28 & 0.631 \\
\quad same & 100 & 1.09 & 0.615 \\
\quad same & 60 & 1.01 & 0.514 \\
LS + weight decay + lr0 $3 \times 10^{-3}$ & 100 & $\mathbf{1.03}$ & $\mathbf{0.673}$ \\
\bottomrule"""),
    (r"""Rows other than the original are means of three seeds (seed standard deviation 0.02 to 0.23 for the ratio, 0.01 to 0.08 for test mAP@0.5). The original run's ratio is from its logged curve, which ended at epoch 171.}""",
     r"""Rows other than the originals are means of three seeds (seed standard deviation 0.02 to 0.23 for the ratio, 0.01 to 0.12 for test mAP@0.5). At Stage~1 the minimum skips checkpoints before validation mAP@0.5 first reaches 0.1, while the COCO-initialized detector still predicts almost nothing. The original runs' ratios are from their logged curves; the Stage~3 one ended at epoch 171, and the Stage~1 minimum is taken over epochs 50 to 200 as in Table~\ref{tab:overfit}.}"""),
    (r"""This recipe is the Stage~3 model reported in Tables~\ref{tab:best} and~\ref{tab:overfit} and Fig.~\ref{fig:best}, and Section~\ref{sec:stir-cv} tests it on independent scenes.""",
     r"""This recipe is the Stage~3 model reported in Tables~\ref{tab:best} and~\ref{tab:overfit} and Fig.~\ref{fig:best}, and Section~\ref{sec:stir-cv} tests it on independent scenes.

The Stage~1 run differs in its starting point, COCO weights, and in its recipe, \texttt{hyp.stir-soil.yaml}, whose AdamW initial learning rate of $10^{-2}$ is ten times the fine-tuning value. Its validation loss after epoch 50 does not so much turn as swing, between 0.035 and 0.087 from one epoch to the next, and adding label smoothing and weight decay at that learning rate leaves the ratio at 1.36 (Table~\ref{tab:stirreg}). Lowering the learning rate is what removes it. At $3 \times 10^{-3}$ on a 100-epoch schedule the rescored validation classification loss falls from about 0.033 to 0.024 on all three seeds and ends 1.03 times its minimum, and training loss settles at $1.4 \times 10^{-2}$ rather than $4.3 \times 10^{-4}$. Validation $\text{mAP@0.5}$ is still rising at the last epoch; its best epoch is $0.620 \pm 0.042$ against the original run's single best epoch of 0.683, and its held-out test $\text{mAP@0.5}$ is $0.673 \pm 0.044$ against 0.577 for the original final checkpoint. At $10^{-3}$ a 200-epoch schedule gives the highest validation peak of the sweep (0.784) but turns upward again, and a 60-epoch schedule is flat only because it stops before the network has learned the task (test 0.514). As for Stage~3, single-checkpoint validation $\text{mAP@0.5}$ of the chosen recipe ends 0.036 below its best, inside the checkpoint-to-checkpoint swing of about 0.05 on 12 validation originals. This recipe is the Stage~1 model reported in Tables~\ref{tab:baseline}, \ref{tab:best} and~\ref{tab:overfit} and Fig.~\ref{fig:baseline}; it has not been cross-validated."""),

    # 11. Practical consequences paragraph.
    (r"""The retrained run of Section~\ref{sec:stir-reg} and the cross-validated estimate of Section~\ref{sec:stir-cv} supersede both.""",
     r"""The retrained runs of Section~\ref{sec:stir-reg}, and for Stage~3 the cross-validated estimate of Section~\ref{sec:stir-cv}, supersede both."""),

    # 12. Augmentation section names the original peaks.
    (r"""The Stage~1 result of 0.683 and the original Stage~3 result of 0.711 for Mixed Color and Background were trained""",
     r"""The original Stage~1 and Stage~3 results of 0.683 and 0.711 for Mixed Color and Background were trained"""),

    # 13. Confusion matrix paragraph describes the original run.
    (r"""\emph{Baseline Mixed Color and Background.} Off diagonal""",
     r"""\emph{Baseline Mixed Color and Background (original run).} Off diagonal"""),
]


def apply(text):
    for old, new in EDITS:
        n = text.count(old)
        if n != 1:
            raise ValueError(f"{n} matches for: {old[:70]!r}")
        text = text.replace(old, new)
    return text


if __name__ == "__main__":
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else "IEEE_paper_edited_20261014.tex"
    out = apply(open(src).read())
    print(f"all {len(EDITS)} edits match")
    if len(sys.argv) > 2:
        open(sys.argv[2], "w").write(out)
