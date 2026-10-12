"""
Merge compatible soil moisture datasets to increase training data.

Merge 1: soil-moisture-5sagf + soil-moisture-ir
  → data/training-data/downloaded/soil-moisture-ir-combined
  Both share identical classes: soil-moisture-1.0 through soil-moisture-8.2
  No class remapping needed.

Merge 2: soil_moisture_september + soil-moisture-v4
  → data/training-data/downloaded/soil-moisture-v4-september-combined
  Both datasets are remapped to natural numeric order: index 0='0', 1='1', ..., 10='10'.
  v4's alphabetical Roboflow ordering (where '10' is index 2) is corrected.

Usage:
    python merge_datasets.py
"""

import os
import shutil
import yaml

BASE = 'data/training-data/downloaded'
SPLITS = ['train', 'valid', 'test']


def copy_images(src_dir, dst_dir):
    """Copy all images from src to dst, skipping duplicates by renaming."""
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    if not os.path.isdir(src_dir):
        return count
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            # Avoid collision: prepend dataset prefix
            name, ext = os.path.splitext(fname)
            dst = os.path.join(dst_dir, f'{name}_dup{count}{ext}')
        shutil.copy2(src, dst)
        count += 1
    return count


def copy_labels(src_dir, dst_dir, class_remap=None):
    """
    Copy YOLO label files from src to dst.
    If class_remap is provided (dict: old_idx -> new_idx), rewrite class indices.
    """
    os.makedirs(dst_dir, exist_ok=True)
    count = 0
    if not os.path.isdir(src_dir):
        return count
    for fname in os.listdir(src_dir):
        if not fname.endswith('.txt'):
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.exists(dst):
            name, ext = os.path.splitext(fname)
            dst = os.path.join(dst_dir, f'{name}_dup{count}{ext}')

        if class_remap is None:
            shutil.copy2(src, dst)
        else:
            lines_out = []
            with open(src) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    old_idx = int(parts[0])
                    new_idx = class_remap.get(old_idx, old_idx)
                    lines_out.append(f'{new_idx} ' + ' '.join(parts[1:]))
            with open(dst, 'w') as f:
                f.write('\n'.join(lines_out) + '\n')
        count += 1
    return count


def write_yaml(dst_dataset, classes, nc):
    yaml_path = os.path.join(dst_dataset, 'data.yaml')
    # Use paths relative to project root, consistent with other datasets
    rel = dst_dataset.replace('\\', '/')
    data = {
        'train': f'./{rel}/train/images',
        'val':   f'./{rel}/valid/images',
        'test':  f'./{rel}/test/images',
        'nc':    nc,
        'names': classes,
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    print(f'  Wrote {yaml_path}')


# ── Merge 1: soil-moisture-5sagf + soil-moisture-ir ──────────────────────────
# Both datasets have identical class names and indices — just pool the images.

def merge_ir_combined():
    src_a = os.path.join(BASE, 'soil-moisture-5sagf')
    src_b = os.path.join(BASE, 'soil-moisture-ir')
    dst   = os.path.join(BASE, 'soil-moisture-ir-combined')

    classes = ['soil-moisture-1.0', 'soil-moisture-2.0', 'soil-moisture-3.0',
               'soil-moisture-5.0', 'soil-moisture-8.2']
    nc = 5

    print('\n── Merge 1: soil-moisture-5sagf + soil-moisture-ir ──────────────')
    print(f'  Output: {dst}')

    total_imgs = total_lbls = 0
    for split in SPLITS:
        img_dst = os.path.join(dst, split, 'images')
        lbl_dst = os.path.join(dst, split, 'labels')

        for src in [src_a, src_b]:
            n = copy_images(os.path.join(src, split, 'images'), img_dst)
            m = copy_labels(os.path.join(src, split, 'labels'), lbl_dst)
            total_imgs += n
            total_lbls += m
            print(f'  [{split}] {os.path.basename(src)}: {n} images, {m} labels')

    write_yaml(dst, classes, nc)
    print(f'  Total: {total_imgs} images, {total_lbls} labels')


# ── Merge 2: soil-moisture-v4 + soil_moisture_september ──────────────────────
# Normalized class order: index N = moisture level N (0 through 10).
# Both v4 and september use Roboflow's alphabetical ordering, which must be
# corrected to natural numeric order.
#
# v4 roboflow order (index → name):
#   0→'0'  1→'1'  2→'10'  3→'2'  4→'3'  5→'4'  6→'5'  7→'6'  8→'7'  9→'8'  10→'9'
# Normalized remap (v4_idx → normalized_idx):
V4_TO_NORM = {
    0: 0,   # '0'  → 0
    1: 1,   # '1'  → 1
    2: 10,  # '10' → 10
    3: 2,   # '2'  → 2
    4: 3,   # '3'  → 3
    5: 4,   # '4'  → 4
    6: 5,   # '5'  → 5
    7: 6,   # '6'  → 6
    8: 7,   # '7'  → 7
    9: 8,   # '8'  → 8
    10: 9,  # '9'  → 9
}

# September roboflow order (index → name):
#   0→'0'  1→'1'  2→'10'  3→'3'  4→'4'  5→'5'  6→'7'  7→'8'  8→'9'
# Normalized remap (sept_idx → normalized_idx):
SEPT_TO_NORM = {
    0: 0,   # '0'  → 0
    1: 1,   # '1'  → 1
    2: 10,  # '10' → 10
    3: 3,   # '3'  → 3
    4: 4,   # '4'  → 4
    5: 5,   # '5'  → 5
    6: 7,   # '7'  → 7
    7: 8,   # '8'  → 8
    8: 9,   # '9'  → 9
}


def merge_v4_september():
    src_a = os.path.join(BASE, 'soil-moisture-v4')
    src_b = os.path.join(BASE, 'soil_moisture_september')
    dst   = os.path.join(BASE, 'soil-moisture-v4-september-combined')

    # Natural numeric order: index 0='0', 1='1', ..., 10='10'
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    nc = 11

    print('\n── Merge 2: soil-moisture-v4 + soil_moisture_september ──────────')
    print(f'  Output: {dst}')
    print(f'  Class order: normalized numeric (index 0=moisture 0, ..., index 10=moisture 10)')

    total_imgs = total_lbls = 0
    for split in SPLITS:
        img_dst = os.path.join(dst, split, 'images')
        lbl_dst = os.path.join(dst, split, 'labels')

        # v4: remap from alphabetical to numeric order
        n = copy_images(os.path.join(src_a, split, 'images'), img_dst)
        m = copy_labels(os.path.join(src_a, split, 'labels'), lbl_dst, class_remap=V4_TO_NORM)
        total_imgs += n; total_lbls += m
        print(f'  [{split}] soil-moisture-v4: {n} images, {m} labels (remapped)')

        # september: remap from alphabetical to numeric order
        n = copy_images(os.path.join(src_b, split, 'images'), img_dst)
        m = copy_labels(os.path.join(src_b, split, 'labels'), lbl_dst, class_remap=SEPT_TO_NORM)
        total_imgs += n; total_lbls += m
        print(f'  [{split}] soil_moisture_september: {n} images, {m} labels (remapped)')

    write_yaml(dst, classes, nc)
    print(f'  Total: {total_imgs} images, {total_lbls} labels')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    merge_ir_combined()
    merge_v4_september()
    print('\nDone. New datasets ready for training.')
    print('  Add them to tr.py or train directly with:')
    print('    python train.py --data data/training-data/downloaded/soil-moisture-ir-combined/data.yaml ...')
    print('    python train.py --data data/training-data/downloaded/soil-moisture-v4-september-combined/data.yaml ...')
