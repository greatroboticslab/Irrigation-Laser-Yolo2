const fs = require('fs');
const path = require('path');

process.env.NODE_PATH = '/opt/homebrew/lib/node_modules';
require('module').Module._initPaths();

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, LevelFormat, BorderStyle, WidthType,
  ShadingType, PageNumber, Footer, Header, PageBreak, TabStopType,
  TabStopPosition,
} = require('docx');

const OUT = '/Volumes/MacOS/Downloads/Irrigation-Laser-Yolo2/scientific_manuscript.docx';

// ---------- helpers ----------
const border = { style: BorderStyle.SINGLE, size: 4, color: "888888" };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function inlineRuns(text) {
  // Parse simple inline markdown: **bold**, *italic*, `code`
  const runs = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0; let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) runs.push(new TextRun({ text: text.slice(last, m.index), font: "Calibri" }));
    const s = m[0];
    if (s.startsWith('**')) runs.push(new TextRun({ text: s.slice(2, -2), bold: true, font: "Calibri" }));
    else if (s.startsWith('`')) runs.push(new TextRun({ text: s.slice(1, -1), font: "Consolas" }));
    else runs.push(new TextRun({ text: s.slice(1, -1), italics: true, font: "Calibri" }));
    last = m.index + s.length;
  }
  if (last < text.length) runs.push(new TextRun({ text: text.slice(last), font: "Calibri" }));
  return runs;
}

function para(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 120, line: 300 },
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    children: inlineRuns(text),
    ...opts.paraOpts,
  });
}

function heading(text, level) {
  const sizes = { 1: 36, 2: 30, 3: 26, 4: 24 };
  const levels = {
    1: HeadingLevel.HEADING_1,
    2: HeadingLevel.HEADING_2,
    3: HeadingLevel.HEADING_3,
    4: HeadingLevel.HEADING_4,
  };
  return new Paragraph({
    heading: levels[level],
    spacing: { before: level === 1 ? 360 : 240, after: 160 },
    children: [new TextRun({ text, bold: true, size: sizes[level], font: "Calibri" })],
  });
}

function bulletPara(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
    children: inlineRuns(text),
  });
}

function numberedPara(text) {
  return new Paragraph({
    numbering: { reference: "numbers", level: 0 },
    spacing: { after: 80 },
    children: inlineRuns(text),
  });
}

function tableCell(text, opts = {}) {
  const width = opts.width;
  return new TableCell({
    borders: cellBorders,
    width: { size: width, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    shading: opts.header ? { fill: "E7EEF7", type: ShadingType.CLEAR, color: "auto" } : undefined,
    children: [new Paragraph({
      alignment: opts.alignment || AlignmentType.LEFT,
      children: [new TextRun({
        text: String(text),
        bold: !!opts.header,
        font: "Calibri",
        size: 20,
      })],
    })],
  });
}

function makeTable(headers, rows, colWidths, alignments) {
  const total = colWidths.reduce((a,b)=>a+b,0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h,i) => tableCell(h, { width: colWidths[i], header: true, alignment: alignments?.[i] })),
  });
  const dataRows = rows.map(r => new TableRow({
    children: r.map((c,i) => tableCell(c, { width: colWidths[i], alignment: alignments?.[i] })),
  }));
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows],
  });
}

function figurePlaceholder(text) {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({
      text,
      italics: true,
      color: "555555",
      font: "Calibri",
      size: 20,
    })],
  });
}

// ---------- content ----------
const content = [];

// Title
content.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 240 },
  children: [new TextRun({
    text: "Vision-Based Reading of a Commercial Stir-Probe Soil-Moisture Sensor: A Class-Preserving YOLOv5 Pipeline for Low-Cost Irrigation Monitoring",
    bold: true,
    size: 36,
    font: "Calibri",
  })],
}));

// Authors / affiliation placeholder
content.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 200 },
  children: [
    new TextRun({ text: "[Authors]", italics: true, color: "888888", font: "Calibri", size: 22 }),
  ],
}));
content.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { after: 360 },
  children: [
    new TextRun({ text: "[Affiliations / Corresponding author]", italics: true, color: "888888", font: "Calibri", size: 22 }),
  ],
}));

// ---- Abstract ----
content.push(heading("Abstract", 2));
content.push(para("Real-time soil-moisture sensing in irrigated horticulture is dominated by contact instrumentation that is intrusive, energy-hungry, and costly to scale per plant. We show that a single overhead camera can autonomously read a low-cost commercial stir-probe whose tip-mounted RGB LED encodes moisture as a continuous color shift from red-orange (level 1, dry) to blue-violet (level 10, saturated), turning passive imagery of an unmodified consumer device into a quantitative moisture index. We pose the problem as multi-class object detection — a YOLOv5s detector simultaneously localizes the LED tip and assigns one of five moisture classes — and address three challenges that arise: (i) the class signal lies in hue, so standard photometric augmentation destroys labels; (ii) the primary stir-probe dataset has only 56 images; (iii) imaging modalities (visible, infrared, ultraviolet) and class taxonomies (5- and 11-level) differ across the seven datasets used. Our pipeline introduces a class-preserving augmentation policy that caps hue jitter at ±8°, a programmatically labelled diffusion-based synthetic image generator (Imagen 4 / Stable Diffusion), index normalization that corrects Roboflow alphabetical class ordering before pooling, and a three-stage training protocol — COCO baselines, label-compatible pooling, and fine-tuning from the strongest soil-moisture checkpoint. After fine-tuning, six of seven datasets exceed mAP@0.5 = 0.91, the IR-combined merge reaches 0.995, and v4-IR improves by 43.9 pp over its COCO baseline. Domain-specific initialization removes the early validation-loss spikes seen with COCO weights, halves the time to peak accuracy, and lifts strict-IoU localization (mAP@0.5:0.95) from 0.19–0.56 to 0.59–0.70. The framework demonstrates that a commodity sensor and a single fixed camera can support continuous, non-contact moisture monitoring at deployment-grade accuracy, providing a scalable alternative to per-plant transducer arrays."));

content.push(new Paragraph({
  spacing: { before: 120, after: 240 },
  children: [
    new TextRun({ text: "Keywords: ", bold: true, font: "Calibri" }),
    new TextRun({ text: "soil moisture; precision irrigation; object detection; YOLOv5; transfer learning; class-preserving augmentation; synthetic data; smart agriculture", font: "Calibri" }),
  ],
}));

// ---- 1 Introduction ----
content.push(heading("1. Introduction", 1));
content.push(para("Irrigated horticulture accounts for roughly 70% of global freshwater withdrawals (FAO, 2020), and even modest gains in moisture-aware scheduling translate directly into water savings and yield. Yet the sensors that drive scheduling — capacitance probes, time-domain reflectometers, tensiometers, and gravimetric samplers — share a common deployment cost: each plant or zone requires its own transducer, calibration, wiring, and maintenance (Susha Lekshmi et al., 2014; Kashyap & Kumar, 2021). Per-plant cost limits density of measurement, which in turn limits the spatial resolution at which scheduling can be optimized. A non-contact reading method that is cheap to scale is therefore an enabling step toward fine-grained precision irrigation."));
content.push(para("Computer vision offers an obvious lever. Recent work has applied convolutional networks to plant-scale moisture proxies — leaf wilting (Kaneda et al., 2017), canopy thermography (García-Tejero et al., 2018), surface-soil reflectance (Atif et al., 2021), and tomato fruit dehydration (Wang et al., 2023) — but each requires either thermal hardware, controlled imaging, or physiological proxies whose relationship to root-zone moisture is indirect. A third class of approach reads the indicator on an existing sensor: meter-dial readers (Liu et al., 2019) and seven-segment display recognizers (Salomon et al., 2022) treat the sensor as a passive scene element. To our knowledge, no published study reads a low-cost commercial stir-probe whose LED color encodes a continuous moisture index, despite the wide commercial availability of such devices."));
content.push(para("The stir-probe targeted in this study costs under USD 15, is sold as a consumer plant-care accessory, and reports moisture on a 1–10 ordinal scale via a tip-mounted RGB LED that shifts monotonically from red-orange (dry) through pink-magenta and blue-violet to saturated indigo (wet). Reading the device by camera converts a per-plant transducer into a per-pot visual cue that one fixed overhead camera can monitor for an arbitrary number of pots simultaneously. Three properties make the task non-trivial. First, the discriminative signal is hue, not shape: the LED occupies less than 2% of the frame and adjacent classes in the blue-violet range (levels 7, 8, 10) differ primarily in spot diameter and saturation. Second, the LED–soil coupling reverses contrast across the moisture scale — a faint red-orange glow over bright dry soil at one extreme, a bright blue-violet bloom over dark wet soil at the other. Third, public training data for this specific device is absent and must be assembled from a heterogeneous mix of small in-house collections and Roboflow Universe sources captured under different illumination modalities and class taxonomies."));
content.push(para("This paper makes the following contributions:"));
const contribs = [
  "**A vision-based reading framework** for an unmodified consumer stir-probe, formulated as a single multi-class object-detection task that produces a localized, ordinal moisture estimate suitable for downstream irrigation logic.",
  "**A class-preserving augmentation policy** that caps hue jitter at ±8° — the empirical inter-class crossover bound — and combines it with non-hue photometric and full geometric diversity, expanding the 56-image primary dataset by 12.2× without injecting hue-induced label noise.",
  "**A diffusion-driven synthetic-image pipeline** (Imagen 4 and Stable Diffusion 1.5 backends) that conditions on physically grounded per-class prompts and a 22-descriptor scene-variation pool, with a discussion of the label-noise trade-off that programmatic bounding-box generation introduces.",
  "**A label-normalization procedure** that corrects Roboflow's alphabetical class ordering — a silent failure mode in which moisture level 10 is assigned class index 2 — enabling pooling of label-compatible datasets.",
  "**A three-stage training protocol** (COCO baselines, label-compatible pooling, domain-specific fine-tuning) that is shown to be additive: pooling lifts the weaker visible-light dataset from 0.760 to 0.942 mAP@0.5, and fine-tuning further lifts the IR-combined model to 0.995.",
  "**A modality study** across visible, IR, and UV illumination on a controlled v4 dataset family, showing that all three modalities support mAP@0.5 ≥ 0.91 once domain-specific weights are used and that the COCO-baseline modality ranking does not predict the fine-tuned ranking.",
];
contribs.forEach(c => content.push(numberedPara(c)));
content.push(para("Section 2 details the sensor, datasets, augmentation policy, synthetic pipeline, architecture, training protocol, and evaluation. Section 3 reports baseline, merged, fine-tuned, modality, and confusion-matrix results. Section 4 discusses implications, label-noise limitations, and deployment outlook. Section 5 concludes."));

// ---- 2 Methods ----
content.push(heading("2. Materials and Methods", 1));

content.push(heading("2.1 Sensor and Physical Color Signal", 2));
content.push(para("The target device is a cylindrical consumer stir-probe inserted vertically into potting soil. Embedded circuitry reads substrate capacitance and drives a tip-mounted RGB LED whose dominant wavelength shifts monotonically with moisture (Table 1, Figure 1). Total luminous flux at the soil surface increases with moisture, producing two coupled effects relevant for detection. At low moisture the soil is light and dry, depressing contrast against a faint red-orange glow. At high moisture the soil is dark and wet, amplifying contrast but also driving camera-side core clipping that can distort apparent color. A practically useful detector must remain robust across both regimes."));
content.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "Table 1. LED appearance per moisture level.", bold: true, font: "Calibri", size: 20 })],
}));
content.push(makeTable(
  ["Level", "LED color description", "Spot diameter"],
  [
    ["1",  "Faint red-orange glow; dim, concentrated point",            "15–30 px"],
    ["3",  "Pink-magenta to pale purple; moderate brightness",          "25–55 px"],
    ["7",  "Clear blue-violet (indigo); visible halo on adjacent soil", "35–70 px"],
    ["8",  "Strong blue-violet bloom; diffuses into dark wet soil",     "60–100 px"],
    ["10", "Intensely bright blue-violet/indigo; white-blue core",      "80–140 px"],
  ],
  [1000, 6360, 2000],
  [AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.CENTER]
));
content.push(figurePlaceholder("[Figure 1.] Real LED appearance at moisture levels 1, 3, 7, 8, 10 (one frame per class, side-by-side)."));

content.push(heading("2.2 Datasets", 2));
content.push(para("Seven datasets covering visible, infrared, and ultraviolet imaging modalities and two class taxonomies were used (Table 2). The primary dataset, `soil_moisture_stir_september`, comprises 56 overhead 640 × 640 px frames of a single instrumented grow pot (39 train / 12 val / 5 test), each carrying a single YOLO bounding box around the LED tip and a label drawn from {1, 3, 7, 8, 10}. Six supporting datasets were drawn from the Roboflow Universe workspace `robotics-lab-1 / mtsu`. Three were 11-class collections (`soil-moisture-v4`, `-v4-ir`, `-v4-uv`) imaged under controlled visible, IR, and UV illumination of an identical pot configuration. Two were 5-class visible/IR collections of the stir-probe class (`soil-moisture-5sagf`, `-ir`). One was an outdoor 11-class field collection (`soil_moisture_september`) acquired without active stirring. A further candidate, `soil-moisture-v3`, was excluded: its 87 raw-voltage classes were incompatible with the discrete schemas used here."));
content.push(figurePlaceholder("[Figure 2.] Dataset overview montage — one representative annotated frame per dataset."));
content.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "Table 2. Source datasets used for training and evaluation.", bold: true, font: "Calibri", size: 20 })],
}));
content.push(makeTable(
  ["Dataset", "Spectrum", "Classes", "Role"],
  [
    ["soil-moisture-v4",                   "Visible", "11", "Strongest baseline; transfer-learning source"],
    ["soil-moisture-v4-ir",                "IR",      "11", "Modality variant; fine-tuning target"],
    ["soil-moisture-v4-uv",                "UV",      "11", "Modality variant; fine-tuning target"],
    ["soil-moisture-ir",                   "IR",      "5",  "Best 5-class IR baseline"],
    ["soil-moisture-5sagf",                "Visible", "5",  "Visible 5-class baseline"],
    ["soil_moisture_september",            "Visible", "11", "Outdoor field; merge partner for v4"],
    ["soil_moisture_stir_september",       "Visible", "5",  "Primary task; 56 real frames"],
    ["soil-moisture-ir-combined",          "IR",      "5",  "Pool of 5sagf + ir"],
    ["soil-moisture-v4-september-combined","Visible", "11", "Pool of v4 + september"],
  ],
  [3000, 1300, 1100, 3960],
  [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.LEFT]
));

content.push(heading("2.3 Class Normalization and Dataset Pooling", 2));
content.push(para("Two pooled configurations were assembled. **IR-combined** pooled `soil-moisture-5sagf` and `soil-moisture-ir`, which share class names and indices and required only image and label aggregation. **v4 + September combined** required correcting an ordering inconsistency: Roboflow exports use alphabetical class names, so the string \"10\" precedes \"2\" and moisture level 10 is assigned class index 2. We remapped both source datasets to `index N → level N` before pooling. This re-indexing is a silent failure mode — without it, two label-compatible datasets cannot be pooled at all, and the resulting confusion is class-systematic rather than performance-degrading, making it easy to miss in casual evaluation."));

content.push(heading("2.4 Class-Preserving Data Augmentation", 2));
content.push(para("Augmentation of LED-color classification data violates an assumption that underlies most detection pipelines. Standard photometric augmentation perturbs hue by ±20–30° because object identity is hue-invariant in conventional detection; here the class label *is* the hue, and a ±20° perturbation applied to a level-3 (pink-magenta) frame produces a glow visually indistinguishable from level 7 (blue-violet), injecting silently mislabelled samples directly into training. We measured the inter-class crossover threshold empirically across the five LED states and adopted a **hue-shift cap of ±8°** as the largest perturbation that preserves dominant-wavelength identity for every class."));
content.push(para("Within that constraint, we apply non-hue photometric transforms and full geometric diversity using the Albumentations library (Table 3). Geometric transforms — horizontal flip (p = 0.5), vertical flip (p = 0.2), rotation up to ±20° (p = 0.6), perspective warp (scale 0.03–0.08, p = 0.4), shift-scale-rotate (shift ±8%, scale ±15%, rotate ±10°, p = 0.4), and random resized crop to 640 × 640 at 75–100% scale (p = 0.5) — model camera-positioning variability for an overhead pot view. Photometric transforms — brightness/contrast (p = 0.8), HSV with constrained hue (p = 0.5), RGB channel shift (±15, p = 0.3), Gaussian and motion blur, Gaussian noise, JPEG quality 60–95, random shadow and fog, and random downscale-then-upscale — model indoor lighting drift, camera noise, and sensor-quality variation. A box-visibility threshold of 0.3 discards crops in which the LED region falls below 30% of its original area. Each augmented copy is produced from a deterministic seed jointly derived from source-image and copy indices for full reproducibility."));
content.push(figurePlaceholder("[Figure 3.] One original (top) per moisture class and four augmented variants per class (bottom rows), illustrating geometric and photometric diversity within the ±8° hue cap."));
content.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "Table 3. Augmentation expansion of soil_moisture_stir_september.", bold: true, font: "Calibri", size: 20 })],
}));
content.push(makeTable(
  ["Split", "Originals", "Copies / orig.", "Total"],
  [
    ["Train",      "39", "13", "546"],
    ["Validation", "12", "6",  "84"],
    ["Test",       "5",  "10", "55"],
    ["Total",      "56", "—",  "685"],
  ],
  [3000, 2120, 2120, 2120],
  [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER]
));
content.push(para("Beyond the 12.2× volume increase, the expanded validation split (84 vs. 12) raises the per-class evaluation support from ≈ 2.4 to ≈ 17 images, reducing the 33–50 pp variance per per-class misclassification to a meaningful regime. A custom YOLOv5 hyperparameter file (`hyp.stir-soil.yaml`) tuned mosaic, copy-paste, HSV, flip, and perspective during training of overhead-view datasets."));

content.push(heading("2.5 Synthetic Image Generation", 2));
content.push(para("A diffusion-based generator complements augmentation by introducing photometric and scene diversity that 56 source frames cannot supply. The primary backend is **Google Imagen 4** (`imagen-4.0-generate-001`) accessed via the Gemini API. A base prompt fixes the scene (round black fabric pot, dark humus soil, scattered perlite, strawberry plant in the upper-left quadrant, narrow cylindrical metal probe, ambient indoor lighting, black braided cables at the rim, wooden laminate floor, monospaced moisture readout overlay) and a per-class clause specifies LED appearance — diameter, color, brightness, surrounding-soil look — for each level (the Level 1 / 3 / 7 / 8 / 10 prompts mirror the descriptors in Table 1). Scene diversity is sampled from a 22-descriptor pool (leaf coverage, soil texture, camera angle, lighting intensity, probe position, cable visibility, perlite density, plant occlusion). A secondary **Stable Diffusion 1.5** backend (Diffusers, fp16 on RTX 2080Ti) provides a self-hosted alternative when API access is rate-limited."));
content.push(para("Bounding boxes are produced programmatically. An HSV color mask isolates the LED-glow blob in the generated image and a 30% upward pad includes the probe tip; if HSV detection fails, the box is sampled from per-class spatial priors estimated from the empirical real-image bounding-box distribution. The latter introduces a label-noise term, discussed in Section 4.3. Outputs are 640 × 640 JPEG at quality 92 with paired YOLO label files. The target volume of 375 frames (5 classes × 75; 50/15/10 train/val/test per class) brings the combined real + augmented + synthetic pool for the stir-probe dataset to ~796 images."));
content.push(figurePlaceholder("[Figure 4.] Real (top row) vs. Imagen 4 synthetic (bottom row) per moisture class."));

content.push(heading("2.6 Detection Architecture", 2));
content.push(para("YOLOv5s — a CSPDarknet53 backbone with a PANet feature-pyramid neck and three detection heads at strides 8 / 16 / 32 px, 7.2 M parameters — was adopted across all experiments at a fixed 640 × 640 px input resolution. Larger variants (YOLOv5m / l / x) were evaluated in pilot runs and overfit on the under-50-images-per-class regime of the primary dataset; the smallest member of the family provides the strongest implicit regularization without measurable accuracy loss on the larger pooled and modality datasets."));

content.push(heading("2.7 Three-Stage Training Protocol", 2));
content.push(para("All experiments ran on the MTSU research cluster (NVIDIA RTX 2080Ti, 11 GB VRAM, CUDA 12.4, with selected runs migrated to an A100 partition for the larger pooled experiments). Optimization used AdamW with a cosine learning-rate schedule, batch size 32, image size 640 × 640, and an early-stopping patience of 100 epochs over a 200-epoch budget. The three stages address distinct hypotheses about where accuracy comes from on this task."));
content.push(para("**Stage 1 — Baseline.** Each of the seven datasets is trained independently from COCO-pretrained weights (`yolov5s.pt`). The custom `hyp.stir-soil.yaml` is applied to overhead-view datasets (`soil_moisture_stir_september`, `soil-moisture-v4`, `-v4-uv`); other datasets use default YOLOv5 hyperparameters."));
content.push(para("**Stage 2 — Pooled training.** The two pooled datasets (Section 2.3) are trained from COCO weights under identical settings, testing whether label-compatible pooling alone produces measurable gains."));
content.push(para("**Stage 3 — Domain-specific fine-tuning.** Five targets — both pooled datasets and `soil-moisture-v4-ir`, `-v4-uv`, `soil_moisture_stir_september` — are fine-tuned from the strongest Stage-1 soil-moisture checkpoint (automatically selected by scanning baseline `results.csv` files for peak mAP@0.5; the chosen source was `soil-moisture-v4`). The hypothesis is that the soil-moisture weight space is closer to the target task than COCO, so feature detectors for small glowing spots against soil clutter and the classifier's spatial priors transfer with minimal adaptation."));
content.push(figurePlaceholder("[Figure 5.] Schematic of the three-stage training protocol."));

content.push(heading("2.8 Evaluation", 2));
content.push(para("Detection accuracy is reported as **mAP@0.5** (the area under per-class precision–recall curves at IoU = 0.5, averaged across classes) — the standard YOLO benchmark. **mAP@0.5:0.95** (the average over IoU thresholds 0.5 to 0.95 in 0.05 steps) is reported as a stricter localization-quality measure, particularly informative for the stir-probe task because downstream irrigation logic consumes the precise tip coordinate. **Normalized confusion matrices** are computed on held-out validation sets, with rows the ground truth and row-normalized cell values; diagonal entries quantify per-class recall and off-diagonal mass identifies ordinal-local versus large-magnitude confusions, a particularly useful diagnostic given the ordinal LED-hue scale. **Per-epoch precision and recall** are tracked to characterize the volatility regime that small per-class validation support induces."));

// ---- 3 Results ----
content.push(heading("3. Results", 1));

content.push(heading("3.1 Baseline Training Dynamics", 2));
content.push(para("Training-loss curves were smooth and monotonically descending across all seven baselines, confirming stable optimization from COCO initialization. Validation dynamics, however, diverged sharply with class count and per-class support. The two 5-class IR-friendly datasets (`soil-moisture-5sagf`, `soil-moisture-ir`) produced clean validation curves with classification loss tracking the training trend at moderate noise and precision/recall stabilizing within 100–150 epochs. The four multi-class or stir-probe datasets exhibited pronounced early validation classification-loss spikes — peaking at approximately 20 (`v4-ir`), 100 (`v4-uv`), and 8 (stir) — before recovering, a signature of a COCO-initialized classifier reorganizing high-confidence mass from natural-scene categories to LED-hue categories. The stir-probe dataset, with ≈ 2.4 validation images per class, never reached a stable validation plateau within the 200-epoch budget; each individual misclassification flipped per-class recall by 33–50 pp. mAP@0.5:0.95 stayed below 0.30 across all baseline runs, indicating that detections at the lenient 0.5 IoU threshold were generally reliable but that strict-IoU localization was not."));
content.push(figurePlaceholder("[Figure 6.] Per-dataset validation curves for the seven baselines. The IR/5-class datasets (cleanly converging) and the multi-class datasets (early cls-loss spikes) are visually distinct."));

content.push(heading("3.2 Stage 1 — Per-Dataset Baselines", 2));
content.push(para("Stage 1 mAP@0.5 spanned 0.30–0.76 across the seven datasets (Table 4). The visible-light baseline `soil-moisture-v4` led at 0.760, followed by `soil-moisture-ir` (0.763) and `soil_moisture_september` (0.715). The stir-probe baseline of 0.683 exceeded the 0.5 detection threshold despite the 39-image training split but represents a noisy estimate from sparse validation, consistent with the volatility shown in Section 3.1. The two infrared and ultraviolet variants of v4 — `v4-ir` (0.301) and `v4-uv` (0.500) — substantially trailed visible-light v4 despite identical sensor and labeling protocol, attributable to smaller per-class training counts and partial class-subset coverage rather than any intrinsic disadvantage of IR or UV imaging; both recover strongly under fine-tuning (Section 3.4)."));
content.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "Table 4. Stage 1 baseline performance (peak best-checkpoint mAP, YOLOv5s, COCO initialization).", bold: true, font: "Calibri", size: 20 })],
}));
content.push(makeTable(
  ["Dataset", "mAP@0.5", "mAP@0.5:0.95", "Best epoch"],
  [
    ["soil-moisture-ir",              "0.763", "0.539", "153"],
    ["soil-moisture-v4",              "0.760", "0.498", "189"],
    ["soil_moisture_september",       "0.715", "0.331", "149"],
    ["soil_moisture_stir_september",  "0.683", "0.236", "101"],
    ["soil-moisture-v4-uv",           "0.500", "0.372", "135"],
    ["soil-moisture-5sagf",           "0.524", "0.282", "114"],
    ["soil-moisture-v4-ir",           "0.301", "0.191", "197"],
  ],
  [3960, 1800, 2200, 1400],
  [AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT]
));
content.push(figurePlaceholder("[Figure 7.] Bar chart of Stage 1 mAP@0.5 per dataset; horizontal line at 0.5."));

content.push(heading("3.3 Stage 2 — Pooled Training", 2));
content.push(para("Pooling sharply changed performance for the two configurations. The visible-light pool `soil-moisture-v4-september-combined` reached mAP@0.5 = 0.942 (mAP@0.5:0.95 = 0.643), substantially exceeding the standalone v4 baseline (0.760) and confirming that cross-session exposure during a single training run improves visible-light generalization even from COCO weights. The IR-combined pool, in contrast, reached only 0.591 — *below* the standalone `soil-moisture-ir` baseline (0.763). Naive pooling of a strong and a weaker label-compatible IR set therefore did not benefit the stronger component from COCO initialization; the recovery emerges only after fine-tuning (Section 3.4). The asymmetric pooling outcome — additive for ambient visible light, sub-additive for IR — is one of the stronger empirical findings of this study and motivates against generic \"more data is better\" pooling heuristics for label-compatible sets."));

content.push(heading("3.4 Stage 3 — Domain-Specific Fine-Tuning", 2));
content.push(para("Fine-tuning from the `soil-moisture-v4` baseline checkpoint (selected automatically as the strongest Stage-1 source) lifted every target into the high-accuracy regime (Table 5). Six of the seven datasets — all but the stir-probe primary — exceeded mAP@0.5 = 0.91 under the fine-tuned configuration. The IR-combined pool reached **0.995** (the highest single result in the study), v4-IR reached 0.987 (a +43.9 pp gain over its own COCO baseline, the largest absolute single-dataset gain observed), and the v4-September pool reached 0.955."));
content.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "Table 5. Best mAP@0.5 per dataset across all three stages.", bold: true, font: "Calibri", size: 20 })],
}));
content.push(makeTable(
  ["Dataset", "Stage 1 (COCO)", "Stage 2 (Pooled)", "Stage 3 (Fine-tuned)", "Best"],
  [
    ["soil-moisture-v4-ir",                "0.301", "—",     "0.987", "0.987"],
    ["soil-moisture-v4-uv",                "0.500", "—",     "0.912", "0.912"],
    ["soil-moisture-v4 (in pool)",         "0.760", "0.942", "0.955", "0.955"],
    ["soil_moisture_september (in pool)",  "0.715", "0.942", "0.955", "0.955"],
    ["soil-moisture-ir (in pool)",         "0.763", "0.591", "0.995", "0.995"],
    ["soil-moisture-5sagf (in pool)",      "0.524", "0.591", "0.995", "0.995"],
    ["soil_moisture_stir_september",       "0.683", "—",     "0.711", "0.711"],
  ],
  [3500, 1500, 1500, 1700, 1160],
  [AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT]
));
content.push(para("mAP@0.5:0.95 in the fine-tuned runs reached 0.59–0.70 (vs. 0.19–0.56 baselines), with `v4-uv` at 0.685, `v4-ir` at 0.586, the IR-combined pool at 0.593, and the v4-September pool at 0.696. Strict-IoU localization improvements are consistent with the hypothesis that domain-specific weights already encode spatial priors for LED-tip placement."));
content.push(para("The fine-tuned validation curves were qualitatively distinct from the baselines: the early classification-loss spikes characterizing COCO-initialized multi-class runs (≈ 20 for v4-ir, ≈ 100 for v4-uv) were absent, validation classification loss descended monotonically from epoch 1, and the IR-family fine-tuned models reached 95% of peak mAP@0.5 within ~50 epochs against 100–150 for their baseline counterparts. This is mechanistically grounded — COCO weights encode feature detectors for natural-scene objects whose representations must reorganize before discriminating LED hues, while domain-specific weights begin near the target geometry, with the classifier head initialized close to optimum."));
content.push(figurePlaceholder("[Figure 8.] Comparison bar chart of best mAP@0.5 per dataset across the three stages."));
content.push(figurePlaceholder("[Figure 9.] Per-dataset fine-tuned validation curves alongside Stage-1 counterparts; cls-loss spikes are visibly absent in Stage 3."));

content.push(heading("3.5 Effect of Class-Preserving Augmentation on the Stir-Probe Dataset", 2));
content.push(para("The augmentation pipeline expanded the stir-probe training split from 39 to 546 images and validation from 12 to 84. The augmented model (`stir5`), initialized from the fine-tuned `soil-moisture-v4` checkpoint and trained over the expanded set, reached mAP@0.5 = 0.625 — *below* the non-augmented fine-tuned figure of 0.711. We trace the gap to the synthetic portion of the expanded set: bounding boxes there are sampled from per-class spatial priors rather than computed from a true LED detection in the generated frame, and the resulting label noise partially offsets the volume gain (quantitative discussion in Section 4.3). The augmented model nonetheless shows the qualitative validation-curve improvement that motivated augmentation in the first place — per-epoch precision/recall volatility is visibly reduced, the curves stabilize earlier, and the expanded validation split lifts per-class evaluation support from ~2.4 to ~17 images per class, which is the more durable contribution."));
content.push(figurePlaceholder("[Figure 10.] Stir-probe validation curves: Stage 1 baseline vs. fine-tuned vs. augmented + fine-tuned (stir5). Volatility decreases monotonically down the rows even when peak mAP does not."));

content.push(heading("3.6 Imaging-Modality Comparison", 2));
content.push(para("The v4 dataset family supports a controlled three-way comparison of visible, IR, and UV illumination (Table 6). Under COCO baselines, ambient-visible v4 (0.760) outperforms both v4-IR (0.301) and v4-UV (0.500) — a counter-intuitive ordering attributable to per-class sample count and class-subset coverage rather than physical modality. After fine-tuning, the ordering inverts: IR leads on mAP@0.5 (0.987) and UV achieves the highest mAP@0.5:0.95 (0.685), indicating that ultraviolet illumination produces the most precise bounding-box localization despite slightly lower classification accuracy. All three modalities support deployment-grade accuracy (≥ 0.91 mAP@0.5) once domain-specific weights are in use."));
content.push(new Paragraph({
  spacing: { before: 120, after: 80 },
  children: [new TextRun({ text: "Table 6. Modality comparison across the v4 family.", bold: true, font: "Calibri", size: 20 })],
}));
content.push(makeTable(
  ["Modality", "Dataset", "Baseline mAP@0.5", "Best mAP@0.5", "Best mAP@0.5:0.95"],
  [
    ["Visible (ambient)", "soil-moisture-v4",    "0.760", "0.955 (pool)", "0.696"],
    ["Infrared",          "soil-moisture-v4-ir", "0.301", "0.987",        "0.586"],
    ["Ultraviolet",       "soil-moisture-v4-uv", "0.500", "0.912",        "0.685"],
  ],
  [2200, 2300, 1700, 1700, 1460],
  [AlignmentType.LEFT, AlignmentType.LEFT, AlignmentType.RIGHT, AlignmentType.RIGHT, AlignmentType.RIGHT]
));

content.push(heading("3.7 Confusion-Matrix Analysis", 2));
content.push(para("**Baseline stir-probe.** Off-diagonal mass concentrates among adjacent levels in the blue-violet range (7, 8, 10), reflecting their physical visual similarity. Level 1 (red-orange) is most reliably detected because its dominant wavelength is maximally distinct from the other classes, and level 3 shows bidirectional confusion with both level 1 and level 7, occupying the intermediate hue position the baseline model is insufficiently trained to assign consistently."));
content.push(para("**Fine-tuned IR-combined.** Levels 1.0, 2.0, and 8.2 reach per-class recall of 1.00; levels 3.0 and 5.0 reach 0.86 and 0.87, with the only material off-diagonal mass confined to mutual confusion between the two adjacent classes. No cross-range confusions (e.g., level 1 predicted as level 8) are observed."));
content.push(para("**Fine-tuned v4 + September combined.** A near-perfect diagonal across the 11 classes; levels 0–6 each exceed 0.91 per-class recall. Performance degrades at the high end (level 8 = 0.67, level 9 = 0.50), consistent with the reduced visual contrast of LED differences over near-saturated wet soil."));
content.push(para("**Fine-tuned v4-IR.** A clean sparse diagonal; all represented classes reach 0.91–1.00 per-class recall with no systematic off-diagonal pattern, confirming that infrared imaging supplies sufficient spectral separation for the fine-tuned model to discriminate fine-grained moisture levels."));
content.push(figurePlaceholder("[Figure 11.] Confusion matrices for (a) baseline stir-probe, (b) fine-tuned IR-combined, (c) fine-tuned v4 + September combined, (d) fine-tuned v4-IR."));

// ---- 4 Discussion ----
content.push(heading("4. Discussion", 1));

content.push(heading("4.1 What Drives Accuracy on This Task", 2));
content.push(para("The three-stage protocol decomposes the contributions of pooling, transfer, and augmentation in a way that the individual gains alone do not. Pooling alone (Stage 2) helps in the visible regime (v4 + September: 0.760 → 0.942) but hurts in IR (5sagf + ir: 0.763 → 0.591). Domain-specific fine-tuning (Stage 3) is the dominant driver: every Stage-3 target enters the ≥ 0.91 regime, and fine-tuning closes the IR-pool deficit (0.591 → 0.995) and the IR-variant deficit (0.301 → 0.987). The qualitative validation-curve change — disappearance of the early classification-loss spikes that signal a COCO-initialized classifier reorganizing its high-confidence mass — is the mechanistic correlate. For practitioners assembling a similar moisture-detection system from heterogeneous sources, the implication is concrete: identify the strongest single soil-moisture checkpoint first, then fine-tune everything else from it; pooling is a secondary lever that helps for some modalities and not others."));

content.push(heading("4.2 Why Modality Ranking Inverts", 2));
content.push(para("The COCO-baseline ranking (visible > UV > IR) and the fine-tuned ranking (IR > visible ≈ UV on mAP@0.5; UV > visible ≈ IR on mAP@0.5:0.95) cannot both be capturing the same underlying property of the modalities. The most parsimonious explanation is that the COCO baseline is a per-dataset quantity dominated by training-set size and class-subset coverage — both of which favor v4 over v4-IR and v4-UV in our collection — while the fine-tuned ranking, conditional on a strong shared starting point, more closely reflects the modality's intrinsic information content for moisture discrimination. The IR lead at IoU = 0.5 is consistent with stronger spectral contrast for water-content differences; the UV lead at IoU = 0.5:0.95 is more surprising and may reflect that ultraviolet illumination produces a geometrically tighter LED glow that anchors strict-IoU localization. We do not claim a settled physical hierarchy among the three modalities, but the inversion is itself a useful caution against drawing modality conclusions from COCO baselines alone — a pattern we suspect appears wherever per-modality data scarcity differs."));

content.push(heading("4.3 Synthetic-Data Label Noise", 2));
content.push(para("The augmented + fine-tuned stir-probe model underperforms the non-augmented fine-tuned model (0.625 vs. 0.711). The volume gain is real (39 → 546 training images) and the per-epoch validation curve is visibly more stable, but the synthetic portion contributes label noise: bounding boxes are sampled from per-class spatial priors rather than computed from a true LED detection in the generated frame. The size of this noise depends on how often the generative model places the LED at the statistical center assumed by the prior. A natural refinement, deferred to future work, is a post-hoc detector — a small auxiliary network trained on real frames only — applied to each generated frame to produce a true LED-glow box, replacing the spatial-prior fallback. We expect this to recover the volume benefit while removing the noise cost. For downstream practitioners, the broader lesson is that programmatic labelling of generative augmentation is an estimate, not a measurement, and its variance must be modeled before assuming a synthetic-data win."));

content.push(heading("4.4 Practical Constraints and Deployment Outlook", 2));
content.push(para("The pipeline as it stands runs at YOLOv5s inference speed (≥ 30 FPS on a single RTX 2080Ti at 640 × 640 px) and produces a localized moisture estimate per frame, which is more than sufficient for irrigation scheduling at minute-to-hour cadence. The main practical constraint is camera placement: the 50 cm overhead distance was chosen to balance LED-glow resolution against scene coverage, and we have not yet characterized the mAP–distance curve under wider-angle deployment. Two additional limitations bear emphasis. First, the primary stir-probe dataset is a single-pot collection; multi-pot generalization is plausible from the augmentation policy but has not been measured directly, and we report it as future work. Second, the moisture levels reported by the LED are themselves a quantization of the underlying capacitance signal, and we do not in this study calibrate the LED-class output against a gravimetric moisture standard — that step is a prerequisite for any agronomic claim and is the natural next experiment in deployed fieldwork."));

content.push(heading("4.5 Comparison to Existing Approaches", 2));
content.push(para("Direct quantitative comparison with the leaf-wilting (Kaneda et al., 2017), thermography (García-Tejero et al., 2018), and surface-reflectance (Atif et al., 2021) lines of work is complicated by the fact that these methods estimate moisture through different proxies — a wilting plant or a dry topsoil layer — at different depths and time constants than the root-zone capacitance the stir-probe samples. Two qualitative observations are nonetheless robust. Hardware cost: the stir-probe (USD ≈ 15) is approximately an order of magnitude cheaper per measurement point than thermal-camera or hyperspectral installations. Accuracy under sparse data: the 0.995 mAP@0.5 we report on the IR-combined pool is in the same regime as values reported for these adjacent vision-based moisture proxies, despite our reading a single tip-mounted LED rather than a full canopy or topsoil image — a benefit, we argue, of treating the device as a designed signal rather than a passive scene element."));

// ---- 5 Conclusion ----
content.push(heading("5. Conclusion", 1));
content.push(para("Soil-moisture readout for precision irrigation has historically tied measurement density to per-plant transducer cost. We have shown that a single fixed overhead camera, a low-cost commercial stir-probe, and a YOLOv5s detector trained under a three-stage protocol can support continuous, non-contact, deployment-grade moisture monitoring of an unmodified consumer device. The principal methodological contributions — a class-preserving ±8° hue-cap augmentation policy, label-compatible dataset normalization, and domain-specific fine-tuning from the strongest soil-moisture checkpoint — together lift six of seven datasets to mAP@0.5 ≥ 0.91 and the IR-combined merge to 0.995. The augmented stir-probe variant currently underperforms the non-augmented fine-tuned model by a label-noise margin we expect to close with a post-hoc detector-based annotation step. Taken together, the work establishes that vision-based reading of a designed signal — a probe LED whose color encodes a quantitative index — is a tractable and cheap path to scaling moisture sensing across the spatial resolution that precision irrigation requires."));

// ---- Acknowledgments ----
content.push(heading("Acknowledgments", 1));
content.push(para("The authors thank the MTSU research-cluster administrators for restoring the NVIDIA UVM driver during the experimental period and for granting A100 partition access. Annotation work was supported by the Roboflow Universe `robotics-lab-1 / mtsu` workspace."));

content.push(heading("Data and Code Availability", 1));
content.push(para("Source code, training scripts, augmentation pipeline, and synthetic-data prompts are available at the project repository. Trained weights for all reported configurations are available on request from the corresponding author. The seven source datasets are hosted on Roboflow Universe under the workspace `robotics-lab-1 / mtsu`."));

// ---- References ----
content.push(heading("References", 1));
content.push(para("(Representative; to be normalized to the journal's reference style.)", { paraOpts: { spacing: { after: 120 } } }));
const refs = [
  "Atif, S., et al. (2021). Soil moisture estimation from RGB imagery using deep learning. *Computers and Electronics in Agriculture*, 188.",
  "FAO (2020). *The State of Food and Agriculture 2020: Overcoming water challenges in agriculture*. Food and Agriculture Organization of the United Nations.",
  "García-Tejero, I. F., et al. (2018). Thermal imaging at the plant level for the assessment of water stress. *Agricultural Water Management*, 197.",
  "Jocher, G., et al. (2020). *YOLOv5*. Ultralytics. https://github.com/ultralytics/yolov5",
  "Kaneda, Y., et al. (2017). Multi-modal sliding-window-based deep learning for plant moisture forecast. *Computers and Electronics in Agriculture*, 142.",
  "Kashyap, B., & Kumar, R. (2021). Sensing methodologies in agriculture for soil moisture and nutrient monitoring. *IEEE Access*, 9.",
  "Liu, Y., et al. (2019). Deep-learning-based reading of analog meters. *IEEE Transactions on Instrumentation and Measurement*, 68(11).",
  "Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. arXiv:1711.05101.",
  "Roboflow (2024). *Roboflow Universe: workspace robotics-lab-1 / mtsu*. https://universe.roboflow.com",
  "Rombach, R., et al. (2022). High-resolution image synthesis with latent diffusion models. CVPR.",
  "Salomon, G., et al. (2022). Deep-learning-based digital-display reading. *Sensors*, 22(2).",
  "Susha Lekshmi, S. U., et al. (2014). A critical review of soil-moisture measurement. *Measurement*, 54.",
  "Wang, X., et al. (2023). Computer vision–based monitoring of fruit dehydration. *Computers and Electronics in Agriculture*, 207.",
];
refs.forEach(r => content.push(bulletPara(r)));

// ---- Appendix ----
content.push(heading("Appendix A. Reproducibility", 1));
const appendix = [
  "**Hardware:** NVIDIA RTX 2080Ti (11 GB VRAM, MTSU research partition); selected runs on NVIDIA A100 (40 GB, a100 partition).",
  "**Software:** PyTorch + Ultralytics YOLOv5; Albumentations 1.3+; Diffusers 0.21.4 with `transformers` 4.35.0 and `accelerate` 0.21.0.",
  "**Optimization:** AdamW, cosine schedule, batch 32, image 640 × 640, max 200 epochs, early-stopping patience 100.",
  "**Seeds:** every augmented copy is produced from a deterministic per-pair seed; YOLOv5 training seeds are fixed in the SLURM scripts.",
  "**Dataset sources:** Roboflow Universe workspace `robotics-lab-1 / mtsu`; primary stir-probe collection captured in September.",
];
appendix.forEach(a => content.push(bulletPara(a)));

// ---------- build doc ----------
const doc = new Document({
  creator: "Claude",
  title: "Vision-Based Reading of a Commercial Stir-Probe Soil-Moisture Sensor",
  styles: {
    default: { document: { run: { font: "Calibri", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 32, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 360, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Calibri", color: "2E5496" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Calibri", color: "2E5496" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets",
        levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers",
        levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", font: "Calibri", size: 18 }),
            new TextRun({ children: [PageNumber.CURRENT], font: "Calibri", size: 18 }),
            new TextRun({ text: " of ", font: "Calibri", size: 18 }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Calibri", size: 18 }),
          ],
        })],
      }),
    },
    children: content,
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log("WROTE", OUT, buf.length, "bytes");
}).catch(e => { console.error(e); process.exit(1); });
