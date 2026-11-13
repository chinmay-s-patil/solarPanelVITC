#!/usr/bin/env python3
import os
import shutil
import re
import sys

meshprism_dir = "/home/lavender/OpenFoamUbu/offProjects/solarPanel/meshConv/meshPrism"
base_level3 = os.path.join(meshprism_dir, "level3")

levels_spec = [
    # Level 0 (3 levels below base -> corresponds to meshv2 level 4)
    {"name": "level0", "block": (36, 12, 24), "near": 1, "wake": 1},
    # Level 1 (2 levels below base -> corresponds to meshv2 level 5)
    {"name": "level1", "block": (39, 13, 26), "near": 1, "wake": 1},
    # Level 2 (1 level below base -> corresponds to meshv2 level 6)
    {"name": "level2", "block": (42, 14, 28), "near": 1, "wake": 1},
    # Level 3 (BASE -> corresponds to meshv2 level 7)
    {"name": "level3", "block": (45, 15, 30), "near": 1, "wake": 1},
    # Level 3.5 (Intermediate between level 3 and level 4)
    {"name": "level3.5", "block": (47, 16, 31), "near": 1, "wake": 1},
    # Level 4 (1 level above base -> corresponds to meshv2 level 8)
    {"name": "level4", "block": (49, 16, 33), "near": 2, "wake": 1},
    # Level 4.5 (Intermediate between level 4 and level 5)
    {"name": "level4.5", "block": (51, 17, 35), "near": 2, "wake": 1},
    # Level 5 (2 levels above base -> corresponds to meshv2 level 9)
    {"name": "level5", "block": (54, 18, 36), "near": 2, "wake": 1},
    # Level 6 (3 levels above base -> corresponds to meshv2 level 10)
    {"name": "level6", "block": (60, 20, 40), "near": 2, "wake": 1},
]

def update_block_mesh(file_path, dims):
    with open(file_path, 'r') as f:
        content = f.read()
    pattern = r"hex\s+\(0\s+1\s+2\s+3\s+4\s+5\s+6\s+7\)\s+\(\d+\s+\d+\s+\d+\)"
    replacement = f"hex (0 1 2 3 4 5 6 7) ({dims[0]} {dims[1]} {dims[2]})"
    content = re.sub(pattern, replacement, content)
    with open(file_path, 'w') as f:
        f.write(content)

def update_snappy_mesh(file_path, spec):
    with open(file_path, 'r') as f:
        content = f.read()
    near = spec["near"]
    wake = spec["wake"]
    content = re.sub(r"(nearBodyBox\s*\{\s*mode\s+inside;\s*levels\s*\(\(1E15\s+)\d+(\)\);)", f"\\g<1>{near}\\2", content)
    content = re.sub(r"(wakeBox\s*\{\s*mode\s+inside;\s*levels\s*\(\(1E15\s+)\d+(\)\);)", f"\\g<1>{wake}\\2", content)
    with open(file_path, 'w') as f:
        f.write(content)

def create_level(spec, overwrite=False):
    lvl_name = spec["name"]
    lvl_dir = os.path.join(meshprism_dir, lvl_name)

    if lvl_name == "level3":
        # Level 3 is the base template
        bm = os.path.join(lvl_dir, "system", "blockMeshDict")
        shm = os.path.join(lvl_dir, "system", "snappyHexMeshDict")
        if os.path.exists(bm):
            update_block_mesh(bm, spec["block"])
        if os.path.exists(shm):
            update_snappy_mesh(shm, spec)
        print(f"  [{lvl_name}] Base template verified: block={spec['block']}, near={spec['near']}")
        return

    if os.path.exists(lvl_dir):
        if not overwrite:
            print(f"  [{lvl_name}] Already exists. Skipping.")
            return
        shutil.rmtree(lvl_dir)

    os.makedirs(lvl_dir, exist_ok=True)

    # Copy 0, system, All*
    shutil.copytree(os.path.join(base_level3, "0"), os.path.join(lvl_dir, "0"))
    shutil.copytree(os.path.join(base_level3, "system"), os.path.join(lvl_dir, "system"))

    # Copy constant (excluding polyMesh)
    os.makedirs(os.path.join(lvl_dir, "constant"), exist_ok=True)
    for item in os.listdir(os.path.join(base_level3, "constant")):
        if item == "polyMesh":
            continue
        s_item = os.path.join(base_level3, "constant", item)
        d_item = os.path.join(lvl_dir, "constant", item)
        if os.path.isdir(s_item):
            shutil.copytree(s_item, d_item)
        else:
            shutil.copy2(s_item, d_item)

    for s in ['Allclean', 'Allmesh', 'Allrun', 'Allsolve']:
        s_file = os.path.join(base_level3, s)
        if os.path.exists(s_file):
            shutil.copy2(s_file, os.path.join(lvl_dir, s))
            os.chmod(os.path.join(lvl_dir, s), 0o755)

    bm = os.path.join(lvl_dir, "system", "blockMeshDict")
    shm = os.path.join(lvl_dir, "system", "snappyHexMeshDict")
    update_block_mesh(bm, spec["block"])
    update_snappy_mesh(shm, spec)
    print(f"  [{lvl_name}] Created: block={spec['block']}, near={spec['near']}")

def setup_levels(target_names=None, overwrite=False):
    print("Setting up meshPrism levels...")
    for spec in levels_spec:
        if target_names and spec["name"] not in target_names:
            continue
        create_level(spec, overwrite=overwrite)

if __name__ == "__main__":
    targets = []
    overwrite = False
    for arg in sys.argv[1:]:
        if arg in ["--overwrite", "-overwrite"]:
            overwrite = True
        else:
            if not arg.startswith("level"):
                targets.append(f"level{arg}")
            else:
                targets.append(arg)

    setup_levels(target_names=targets if targets else None, overwrite=overwrite)
