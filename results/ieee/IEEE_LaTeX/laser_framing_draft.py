"""DRAFT, not applied to Overleaf: reframe the paper around what the primary images show.

Evidence (2026-10-26, scripts/evaluation/eval_stir_hue_margin.py, results/stir_hue_check_montage.jpg,
and figs/fig1_led_levels.png itself): every annotated box in Mixed Color and Background encloses
either a small red dot with a clipped white core and red scatter over the soil, or a larger
blue-violet patch. Both occur at every level (red/blue originals per level: 1: 2/6, 3: 3/8,
7: 2/6, 8: 2/6, 10: 6/13; two originals in neither hue band). Level mean hues lie 2.6-4.3 deg
apart against a within-level s.d. of 45-59 deg; box size (sqrt area, px) has medians 68-84 with
overlapping ranges. Fig. 1 shows level 1 blue and levels 3, 7, 10 red, against Table I's
red-orange (1) to indigo (10). The repo and the supervisor's own questions call the spot a laser.

Hardware facts we do not have are left as [CONFIRM: ...] markers, which must be resolved by
whoever captured the data before this goes to Overleaf.

Usage: python laser_framing_draft.py IN.tex OUT.tex [changes.json]
"""

import json
import sys

C = lambda s: "\\textbf{[CONFIRM: " + s + "]}"

EDITS = [
    # ---- running head ----
    ("{Author \\MakeLowercase{\\textit{et al.}}: Reading a Commercial Soil Stir-Probe by Camera}",
     "{Author \\MakeLowercase{\\textit{et al.}}: Estimating Soil Moisture from an Imaged Laser Spot}"),

    # ---- abstract ----
    ("Real time soil moisture sensing in irrigated horticulture is dominated by contact instrumentation that is intrusive, energy hungry, and costly to scale on a per plant basis. We show that a single overhead camera can autonomously read a low cost commercial stir probe whose tip mounted RGB LED encodes moisture as a continuous color shift from red orange (level~1, dry) to blue violet (level~10, saturated), turning passive imagery of an unmodified consumer device into a quantitative moisture index. We pose the problem as multiclass object detection, in which a YOLOv5s detector simultaneously localizes the LED tip and assigns a moisture class (five levels for the stir probe, up to 11 in the supporting datasets), and we address three challenges that arise. First, the class is meant to be carried by the color of the LED, so standard photometric augmentation can corrupt labels.",
     "Soil moisture sensing in irrigated horticulture is dominated by contact instrumentation that is intrusive, energy hungry, and costly to scale on a per plant basis. We study whether a single overhead camera can estimate the moisture level of potted soil from images of a laser spot on the soil surface, each training frame labeled with the reading of a low cost commercial stir probe. We pose the problem as multiclass object detection, in which a YOLOv5s detector simultaneously localizes the spot and assigns a moisture class (five levels in the primary dataset, up to 11 in the supporting datasets), and we address three challenges that arise. First, the cue that separates the levels is not known in advance; in the primary dataset red and blue spots occur at every level and the spot's hue does not separate them, so the detector must learn it from the spot's shape, scatter and brightness or from the surrounding soil, and augmentation must not destroy whichever it is."),
    ("The framework indicates that a commodity sensor and a single fixed camera can support non contact moisture monitoring,",
     "The framework indicates that a laser and a single fixed camera can support non contact moisture estimation,"),

    # ---- introduction ----
    ("To our knowledge, no published study reads a low cost commercial stir probe whose LED color encodes a continuous moisture index, despite the wide commercial availability of such devices.",
     "These approaches read either the plant or an instrument. We instead image the soil surface under a laser spot and use a commercial probe only to label the training frames."),
    ("The stir probe targeted in this study costs under USD~15, is sold as a consumer plant care accessory, and reports moisture on a 1 to 10 ordinal scale through a tip mounted RGB LED that shifts monotonically from red orange (dry) through pink magenta and blue violet to saturated indigo (wet). Reading the device by camera converts a per plant transducer into a per pot visual cue that one fixed overhead camera can monitor for an arbitrary number of pots simultaneously.",
     "The labeling probe costs under USD~15, is sold as a consumer plant care accessory, and reports moisture on a 1 to 10 ordinal scale " + C("how the probe's reading was taken and recorded for each frame") + ". In each frame a laser " + C("laser type, wavelength(s), power, and how it is aimed") + " places a spot on the soil, and the camera records the spot and the soil around it. If that appearance varies with moisture in a learnable way, one fixed overhead camera could estimate moisture across many pots without a probe in each " + C("whether the probe is needed at deployment or only for labeling") + "."),
    ("Three properties make the task nontrivial. First, the discriminative signal is hue, not shape: the LED occupies less than 2\\% of the frame, and adjacent classes in the blue violet range (levels 7, 8, and 10) differ primarily in spot diameter and saturation. Second, the LED and soil coupling reverses contrast across the moisture scale, so a faint red orange glow over bright dry soil at one extreme shifts to a bright blue violet bloom over dark wet soil at the other. Third,",
     "Three properties make the task nontrivial. First, the target is small: the spot occupies less than 2\\% of the frame. Second, the cue is not known in advance: in the primary dataset red and blue spots occur at every level and their hue does not separate the levels (Section~\\ref{sec:sensor}), so whatever the detector uses, whether the spot's shape, scatter and brightness or the darkening of wet soil around it, must be learned. Third,"),
    ("\\item \\textbf{Vision based reading framework.} A framework for an unmodified consumer soil probe, formulated as a single multiclass object detection task that produces a localized, ordinal moisture estimate suitable for downstream irrigation logic.",
     "\\item \\textbf{Vision based estimation framework.} A framework that formulates moisture estimation from an imaged laser spot as a single multiclass object detection task, producing a localized, ordinal moisture estimate suitable for downstream irrigation logic."),
    ("The cap bounds the explicit hue transform only; Section~\\ref{sec:aug} describes the other transforms that move hue or the spot size.",
     "The cap bounds the explicit hue transform only; Section~\\ref{sec:aug} describes the other transforms that move hue or the spot size, and shows that hue does not separate the levels of the primary dataset, so the cap is a precaution."),

    # ---- sensor section, Table I, Fig. 1 ----
    ("\\subsection{Sensor and Physical Color Signal}\nThe target device is a cylindrical consumer stir probe inserted vertically into potting soil. Embedded circuitry reads substrate capacitance and drives a tip mounted RGB LED whose dominant wavelength shifts monotonically with moisture (Table~\\ref{tab:led}, Fig.~\\ref{fig:led}). Total luminous flux at the soil surface increases with moisture, producing two coupled effects that are relevant for detection. At low moisture the soil is light and dry, which depresses contrast against a faint red orange glow. At high moisture the soil is dark and wet, which amplifies contrast but also drives camera side core clipping that can distort apparent color. A practically useful detector must remain robust across both regimes.",
     "\\subsection{Imaging Setup and Moisture Labels}\n\\label{sec:sensor}\nEach frame shows potted soil photographed from above with a laser spot on its surface " + C("capture rig: laser(s), camera, distance, lighting, whether the probe is in the pot during capture") + ". A frame's label is the stir probe's moisture reading, and its annotated box encloses the spot. In the primary dataset the spot takes one of two forms (Table~\\ref{tab:led}, Fig.~\\ref{fig:led}): a small red dot with a clipped white core and red scatter across the surrounding soil, or a larger blue-violet patch. Both forms occur at every level in about the same proportion, and box size does not change systematically with level. Whatever separates the levels is therefore not the spot's color. Candidates are the spot's shape and scatter, which depend on the wetness of the soil surface, and the appearance of the soil itself."),
    ("\\caption{Moisture Level LED Color Mappings and Observed Signal Characteristics}\n\\label{tab:led}\n\\centering\n\\begin{tabular}{c p{3.6cm} c}\n\\toprule\n\\textbf{Level} & \\textbf{LED Color Description} & \\textbf{Spot Diam.} \\\\\n\\midrule\n1  & Faint red orange glow; dim, concentrated point & 15--30 px \\\\\n3  & Pink magenta to pale purple; moderate brightness & 25--55 px \\\\\n7  & Clear blue violet (indigo); visible halo on soil & 35--70 px \\\\\n8  & Strong blue violet bloom; diffuse into wet soil & 60--100 px \\\\\n10 & Intensely bright blue violet/indigo; saturated core & 80--140 px \\\\\n\\bottomrule\n\\end{tabular}\n",
     "\\caption{Spot Appearance by Moisture Level in the Primary Dataset}\n\\label{tab:led}\n\\centering\n\\begin{tabular}{c c c c c}\n\\toprule\n\\textbf{Level} & \\textbf{Originals} & \\textbf{Red} & \\textbf{Blue} & \\textbf{Box size (px)} \\\\\n\\midrule\n1  & 8  & 2 & 6  & 70 (17--130) \\\\\n3  & 12 & 3 & 8  & 74 (29--129) \\\\\n7  & 8  & 2 & 6  & 75 (30--129) \\\\\n8  & 8  & 2 & 6  & 84 (34--139) \\\\\n10 & 20 & 6 & 13 & 68 (25--125) \\\\\n\\bottomrule\n\\end{tabular}\n\\\\[2pt]\n\\parbox{\\columnwidth}{\\footnotesize Spot color from the circular mean hue of the brightest tenth of the box: red above $300^{\\circ}$ or below $40^{\\circ}$, blue $200^{\\circ}$ to $300^{\\circ}$; one original each at levels 3 and 10 is in neither band. Box size is the square root of box area, median (range).}\n"),
    ("\\caption{LED appearance at all five moisture levels.}",
     "\\caption{One original at each moisture level of the primary dataset. Red and blue spots occur at every level (Table~\\ref{tab:led}); the colors shown do not encode the level.}"),

    # ---- datasets, augmentation ----
    ("each carrying a single YOLO bounding box around the LED tip and a label",
     "each carrying a single YOLO bounding box around the laser spot and a label"),
    ("Augmentation of LED color classification data violates an assumption that underlies most detection pipelines. Standard photometric augmentation perturbs hue by $\\pm 20$ to $30^{\\circ}$ because object identity is hue invariant in conventional detection. Here the class label \\emph{is} the hue, and a large enough hue perturbation can move a level 3 (pink magenta) frame toward level 7 (blue violet), which injects silently mislabelled samples directly into training.",
     "Standard photometric augmentation perturbs hue by $\\pm 20$ to $30^{\\circ}$ because object identity is hue invariant in conventional detection. We designed the policy for the case in which the class is carried by the spot's color, where a large enough hue perturbation would move a frame toward another level and inject silently mislabeled samples into training."),
    ("The cap was chosen to be small rather than derived from a measured interclass crossover threshold, and whether $\\pm 16^{\\circ}$ keeps every level clear of its neighbors has not been tested.",
     "The cap was chosen to be small rather than derived from a measured interclass threshold; the measurement below shows that hue does not separate the levels of the primary dataset, so the cap is a precaution rather than a requirement."),
    ("crops in which the LED region falls below", "crops in which the spot falls below"),
    ("the range of a dim or weakly saturated glow such as level~1.", "the range of a dim or weakly saturated spot."),
    ("and fog alter the saturation and brightness that separate levels 7, 8 and 10 (Table~\\ref{tab:led}), and the crop, scale and perspective transforms alter the spot diameter that Table~\\ref{tab:led} lists as a cue between the same levels.",
     "and fog alter the spot's saturation and brightness, and the crop, scale and perspective transforms alter its size, any of which the detector may use."),
    ("Unlike the progression of Table~\\ref{tab:led}, that hue clusters", "As Table~\\ref{tab:led} shows, that hue clusters"),

    # ---- ablation motivation ----
    ("The overhead captures contain a bright specular return at the probe tip alongside the LED glow that carries the class label. Because the return co-occurs with the LED, a detector could in principle key on the return rather than on the color that encodes moisture, which would not transfer to a deployment in which the return is absent or displaced.",
     "In the supporting collections the annotated spot contains a bright specular return of the laser " + C("that the return is the laser's reflection, and what else is in the box") + ". A detector could key on that return, or on the soil around it, rather than on the scattered light that might carry the moisture signal, and either would transfer poorly to a deployment in which the return is absent or displaced."),

    # ---- results and discussion ----
    ("onto LED hue categories", "onto the moisture categories"),
    ("spatial priors for LED tip placement", "spatial priors for spot placement"),
    ("before discriminating LED hues", "before discriminating moisture levels"),
    ("overlay rather than the LED, is excluded", "overlay rather than the spot, is excluded"),
    ("level~1 is the dimmest LED state and its eight originals come from few bursts",
     "level~1's eight originals come from few bursts"),
    ("anything over the color of the LED,", "anything over the color of the spot,"),
    ("Off diagonal mass concentrates among adjacent levels in the blue violet range (7, 8, 10), reflecting their physical visual similarity. On the shipped validation split every level 1 (red orange) frame is classified correctly, as its dominant wavelength being maximally distinct from the other classes would predict, but those frames derive from only two originals;",
     "Off diagonal mass concentrates among levels 7, 8 and 10. On the shipped validation split every level 1 frame is classified correctly, but those frames derive from only two originals;"),
    ("Level 3 shows bidirectional confusion with both level 1 and level 7, occupying the intermediate hue position that the baseline model is insufficiently trained to assign consistently.",
     "Level 3 shows bidirectional confusion with both level 1 and level 7."),
    (", consistent with the reduced visual contrast of LED differences over near saturated wet soil.", "."),
    ("inside the detected LED region. The progression", "inside the detected spot. The progression"),
    ("indicates that the network localizes moisture evidence in the LED glow itself rather than in correlated background cues,",
     "indicates that the network's strongest activations lie in the spot rather than in the surrounding scene, though it does not show that the class decision rests on them,"),
    ("strongest mean activation inside the detected LED region at one backbone stage", "strongest mean activation inside the detected spot at one backbone stage"),

    ("and left the annotated glow intact.", "and left the annotated spot intact."),

    # ---- conclusion ----
    ("We have shown that a single fixed overhead camera, a low cost commercial stir-probe, and a YOLOv5s detector trained under a three stage protocol can read an unmodified consumer device without contact, as a basis for continuous moisture monitoring.",
     "We have shown that a single fixed overhead camera and a YOLOv5s detector trained under a three stage protocol can estimate the moisture level of potted soil from an imaged laser spot, with a low cost commercial stir probe supplying the training labels."),
    ("Taken together, the work establishes that vision based reading of a designed signal, a probe LED whose color encodes a quantitative index, is a tractable and cheap path to scaling moisture sensing across the spatial resolution that precision irrigation requires.",
     "The spot's color does not carry the level, and which cue the detector uses, the spot's scatter or the soil around it, is the open question that decides whether the approach transfers beyond these pots."),
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
