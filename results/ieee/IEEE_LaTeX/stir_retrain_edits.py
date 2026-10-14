"""Paper edits for the retrained Stage 3 Mixed Color and Background run.

Each entry is (old, new); every old string must occur exactly once in the live
IEEE_paper.tex. apply() checks that and returns the edited text. The same pairs
are applied to the Overleaf editor through the browser.

Numbers come from:
  results/stir_reg_round1, results/stir_reg_round2  (sweep, shipped split)
  results/stir_reg_round2/ls_wd_lr_e60/seed*_results.csv  (logged curves)
  results/stir_cv_summary_reg.csv                    (grouped CV, reg arm)
"""

EDITS = [
    # 1. Methods, Stage 3: say this row uses a different recipe.
    (r"""Fig.~\ref{fig:threestage} depicts the protocol.""",
     r"""Fig.~\ref{fig:threestage} depicts the protocol. Under this recipe the \texttt{Mixed Color and Background} fine-tune overfits (Section~\ref{sec:dynamics}); the Stage~3 model reported for it is therefore retrained with label smoothing of 0.1, AdamW weight decay of 0.05, an initial learning rate of $3 \times 10^{-4}$ and a 60-epoch cosine schedule, chosen by the sweep of Section~\ref{sec:stir-reg}."""),

    # 2. Fig. 4 caption.
    (r"""Ten runs descend or flatten; the stir-probe runs alone turn upward while their training loss falls well below every other run.}""",
     r"""Ten runs descend or flatten. The Stage~1 stir-probe run turns upward while its training loss falls well below every other run, and so does the original Stage~3 stir-probe run (dashed). The retrained Stage~3 run (solid; mean of three seeds, band spanning them) does not: its validation loss, rescored every five epochs without label smoothing so that it shares the scale of the other runs, falls and stays flat. Label smoothing also places a floor of about 0.025 under its training loss.}"""),

    # 3. Table IV: keep the original row, add the retrained one.
    (r"""Mixed Color and Background & 57 & $\mathbf{1.69}$ & 0.711 & 0.480 \\""",
     r"""Mixed Color and Background & 57 & $\mathbf{1.69}$ & 0.711 & 0.480 \\
\quad retrained (3 seeds) & 34 & $1.08$ & 0.635 & 0.566 \\"""),
    (r"""All runs cover 200 epochs except the Stage~3 stir-probe run, which terminated at epoch 171.}""",
     r"""All runs cover 200 epochs except the original Stage~3 stir-probe run, which terminated at epoch 171, and its retrained counterpart (Section~\ref{sec:stir-reg}), which has a 60-epoch budget; for it the minimum is taken over all epochs, the values are means over three seeds (final/min.\ 1.04 to 1.16), and validation loss is logged against label-smoothed targets, so only its ratio, not its level, is comparable with the other rows.}"""),

    # 4. End of the overfitting paragraph: point to the fix.
    (r"""agree, which is the classical picture of a network fitting its training frames.""",
     r"""agree, which is the classical picture of a network fitting its training frames. Section~\ref{sec:stir-reg} retrains the Stage~3 run under explicit regularization and a shorter schedule, and the turn disappears."""),

    # 5. Table tab:best, Stage 3 column for Mixed Color.
    (r"""Mixed Color and Background & 0.68 & 0.00 & 0.71 & 0.71 \\
\bottomrule
\end{tabular}""",
     r"""Mixed Color and Background & 0.68 & 0.00 & 0.64$^{a}$ & 0.68 \\
\bottomrule
\end{tabular}
\\[2pt]
\parbox{\columnwidth}{\footnotesize $^{a}$Retrained Stage~3 run of Section~\ref{sec:stir-reg}, mean of the three seeds' peaks (standard deviation 0.03). It replaces the original run's pre-collapse peak of 0.71, which ended at 0.48; on the held-out test split the retrained run scores 0.741 against 0.540.}"""),

    # 6. Fig. 6 caption.
    (r"""The Mixed Color and Background bar is a pre-collapse peak (epoch~32): its last-ten-epoch mean is 0.48, and grouped five-fold cross-validation gives mAP@0.5 of 0.78 (Section~\ref{sec:stir-cv}).}""",
     r"""The Mixed Color and Background bar is the retrained run of Section~\ref{sec:stir-reg}, the mean of its three seeds' peaks with their standard deviation; the original run's pre-collapse peak was 0.711.}"""),

    # 7. Augmentation subsection opener.
    (r"""Both Mixed Color and Background results in Table~\ref{tab:best}, 0.683 at Stage~1 and 0.711 at Stage~3, were trained on""",
     r"""The Stage~1 result of 0.683 and the original Stage~3 result of 0.711 for Mixed Color and Background were trained on"""),

    # 8. Practical consequences.
    (r"""First, the stir-probe figures of 0.683 and 0.711 in Tables~\ref{tab:baseline} and~\ref{tab:best} are peak-epoch values read from the validation log rather than converged accuracies, and should be read together with the 0.440 and 0.480 the same runs end at.""",
     r"""First, the stir-probe figure of 0.683 in Table~\ref{tab:baseline} and the original Stage~3 peak of 0.711 are peak-epoch values read from the validation log rather than converged accuracies, and should be read together with the 0.440 and 0.480 the same runs end at."""),
    (r"""The cross-validated estimate of Section~\ref{sec:stir-cv} supersedes both.""",
     r"""The retrained run of Section~\ref{sec:stir-reg} and the cross-validated estimate of Section~\ref{sec:stir-cv} supersede both."""),

    # 9. New subsection, before the grouped CV.
    (r"""\subsection{Grouped Cross-Validation of the Stir-Probe Recipe}""",
     r"""\subsection{Regularized Retraining of the Stage~3 Stir-Probe Run}
\label{sec:stir-reg}
The fine-tuning recipe of the original Stage~3 run carries almost no explicit regularization: AdamW with a decoupled weight decay of $5 \times 10^{-4}$, no label smoothing, and a 200-epoch cosine schedule. We retrained the run from the same \texttt{IR-Pot-Background} initialization on the same shipped split, changing one factor family at a time: label smoothing of 0.1, weight decay raised to 0.05, the initial learning rate lowered from $10^{-3}$ to $3 \times 10^{-4}$, and the schedule shortened to 100, 60 or 50 epochs. Each recipe was trained with three seeds and early stopping disabled, so every curve covers its full budget. Because YOLOv5 computes validation loss against the same smoothed targets it trains on, a checkpoint saved every five epochs was rescored on the validation split with smoothing off, which puts every recipe on the scale of the original run.

\begin{table}[!t]
\renewcommand{\arraystretch}{1.3}
\caption{Regularization Sweep for the Stage~3 Stir-Probe Run}
\label{tab:stirreg}
\centering
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{l c c c}
\toprule
\textbf{Recipe} & \textbf{Epochs} & \textbf{Final/min.} & \textbf{Test mAP@0.5} \\
\midrule
Original & 200 & 2.47 & 0.540 \\
Weight decay 0.05 & 200 & 2.26 & 0.606 \\
Label smoothing 0.1 (LS) & 200 & 1.55 & 0.598 \\
LS + weight decay & 200 & 1.69 & 0.598 \\
LS + weight decay & 50 & 1.16 & 0.617 \\
LS + weight decay + lr0 $3 \times 10^{-4}$ & 200 & 1.39 & 0.719 \\
\quad same & 100 & 1.21 & 0.687 \\
\quad same, LS 0.2 & 100 & 1.11 & 0.730 \\
\quad same & 60 & $\mathbf{1.05}$ & $\mathbf{0.741}$ \\
\bottomrule
\end{tabular}
\\[2pt]
\parbox{\columnwidth}{\footnotesize Final/min.\ is the unsmoothed validation classification loss at the last checkpoint over its lowest value, both over all epochs; Test mAP@0.5 is the last checkpoint on the 95-frame test split. Rows other than the original are means of three seeds (seed standard deviation 0.02 to 0.23 for the ratio, 0.01 to 0.08 for test mAP@0.5). The original run's ratio is from its logged curve, which ended at epoch 171.}
\end{table}

Weight decay alone leaves the turn in place (Table~\ref{tab:stirreg}), and label smoothing halves the final loss but does not remove the rise. With all three changes the validation loss stays flat until about epoch 90 and then climbs, so the schedule, not the absence of regularization alone, drives the late rise. The 60-epoch version of that recipe removes it: validation classification loss falls from 0.058 to about 0.043 and holds there on all three seeds, ending 1.05 times its minimum against 2.47 for the original run, and the original run had already risen 1.69-fold over the same first 60 epochs. Its held-out test $\text{mAP@0.5}$ is $0.741 \pm 0.011$ against 0.540. On the 12-original validation split the picture is noisier: its best-epoch $\text{mAP@0.5}$ is $0.635 \pm 0.030$ and its last-ten-epoch mean $0.566 \pm 0.046$, lower than the original run's single best epoch of 0.711 but without the original's fall to 0.480. The third seed's logged validation loss ends 15\% above its minimum at epoch 41, so the residual drift is small but not zero. This recipe is the Stage~3 model reported in Tables~\ref{tab:best} and~\ref{tab:overfit} and Fig.~\ref{fig:best}, and Section~\ref{sec:stir-cv} tests it on independent scenes.

\subsection{Grouped Cross-Validation of the Stir-Probe Recipe}"""),

    # 10. CV design: add the third arm.
    (r"""and the same recipe with the ten backbone layers frozen. Each run trained for a fixed 100 epochs""",
     r"""the same recipe with the ten backbone layers frozen, and the retrained recipe of Section~\ref{sec:stir-reg} on its own 60-epoch schedule. Each of the first two arms trained for a fixed 100 epochs"""),

    # 11. CV table: add the retrained arm. Final column is epoch 99, or 59 for it.
    (r"""\textbf{Arm} & \textbf{Ep.~10} & \textbf{Ep.~30} & \textbf{Ep.~50} & \textbf{Ep.~99} & \textbf{mAP@0.5} & \textbf{ECE} \\""",
     r"""\textbf{Arm} & \textbf{Ep.~10} & \textbf{Ep.~30} & \textbf{Ep.~50} & \textbf{Final} & \textbf{mAP@0.5} & \textbf{ECE} \\"""),
    (r"""Frozen backbone & 0.28 & 0.33 & 0.36 & 0.31 & 0.258 & 0.351 \\""",
     r"""Frozen backbone & 0.28 & 0.33 & 0.36 & 0.31 & 0.258 & 0.351 \\
Retrained & 0.35 & 0.72 & 0.83 & 0.85 & 0.760 & @ECE_REG@ \\"""),
    (r"""mAP@0.5 is the mean held-out value at epoch 99 across folds and seeds, augmented copies included; ECE is computed over all held-out frames at epoch 99.}""",
     r"""Final is epoch 99, or epoch 59 for the retrained arm. mAP@0.5 is the mean held-out value at the final epoch across folds and seeds, augmented copies included; ECE is computed over all held-out frames at the final epoch.}"""),

    # 12. CV result for the retrained arm.
    (r"""The cross-validated recipe therefore does not reproduce the collapse of the shipped run,""",
     r"""The retrained recipe matches it on independent scenes: out-of-fold accuracy $0.845 \pm 0.057$ over seeds at epoch 59, still rising at the end of its schedule, @CI_REG@ and a mean held-out $\text{mAP@0.5}$ of 0.760. The difference from the unregularized recipe, 0.012, is less than one original in 56, so the regularization that removes the loss turn costs no measurable accuracy.

The cross-validated recipe therefore does not reproduce the collapse of the shipped run,"""),
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
    src = sys.argv[1] if len(sys.argv) > 1 else "IEEE_paper_live_20261014.tex"
    out = apply(open(src).read())
    left = [p for p in ("@ECE_REG@", "@CI_REG@") if p in out]
    print("all edits match;", "placeholders left: " + ", ".join(left) if left else "no placeholders")
