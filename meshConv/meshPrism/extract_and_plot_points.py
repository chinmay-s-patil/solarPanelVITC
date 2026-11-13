#!/usr/bin/env python3
"""
Mesh Convergence Analysis across 4 Spatial Probe Points for meshPrism.
Extracts pressure and velocity at 4 aerodynamic probe locations:
  Point 1: (0.50, 0.55, 0.0)  - Upstream Stagnation / Leading Edge
  Point 2: (0.00, 0.80, 0.0)  - Upper Surface Suction / Boundary Layer
  Point 3: (-0.80, 0.70, 0.0) - Leeward Wake / Recirculation Zone
  Point 4: (0.20, 0.45, 0.0)  - Under-Panel Venturi Flow
Strictly excludes level3AMD.
Saves 4 individual photos and combined comparison plots.
"""

import os
import glob
import re
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RHO_INF = 1.225

PROBE_POINTS = [
    {
        "id": 1,
        "name": "Point 1: Leading-Edge Stagnation",
        "short_name": "Point 1 (Stagnation)",
        "coords": (0.50, 0.55, 0.0),
        "desc": "Upstream stagnation region in front of the leading edge",
        "file_prefix": "point1_stagnation_pressure"
    },
    {
        "id": 2,
        "name": "Point 2: Upper Surface Suction",
        "short_name": "Point 2 (Top Suction)",
        "coords": (0.00, 0.80, 0.0),
        "desc": "Suction expansion over the tilted upper panel face",
        "file_prefix": "point2_top_suction_pressure"
    },
    {
        "id": 3,
        "name": "Point 3: Leeward Wake Separation",
        "short_name": "Point 3 (Separated Wake)",
        "coords": (-0.80, 0.70, 0.0),
        "desc": "Turbulent recirculation wake downstream of trailing edge",
        "file_prefix": "point3_wake_pressure"
    },
    {
        "id": 4,
        "name": "Point 4: Under-Panel Venturi Flow",
        "short_name": "Point 4 (Under-Panel)",
        "coords": (0.20, 0.45, 0.0),
        "desc": "Accelerated flow between ground and panel lower face",
        "file_prefix": "point4_under_panel_pressure"
    }
]

def parse_probe_file(filepath):
    if not os.path.exists(filepath):
        return None, None
    last_line = None
    with open(filepath, 'r') as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
            last_line = line_str
    if not last_line:
        return None, None
    parts = re.split(r'\s+', last_line)
    try:
        time_val = float(parts[0])
        vals = [float(p) for p in parts[1:]]
        return time_val, vals
    except (ValueError, IndexError):
        return None, None

def get_cell_count(case_dir):
    log_cm = os.path.join(case_dir, "log.checkMesh")
    if os.path.exists(log_cm):
        with open(log_cm, 'r') as fp:
            for line in fp:
                if 'cells:' in line:
                    m = re.search(r'cells:\s*(\d+)', line)
                    if m:
                        return int(m.group(1))
    log_shm = os.path.join(case_dir, "log.snappyHexMesh")
    if os.path.exists(log_shm):
        with open(log_shm, 'r') as fp:
            for line in reversed(fp.readlines()):
                if 'cells:' in line:
                    m = re.search(r'cells:\s*(\d+)', line)
                    if m:
                        return int(m.group(1))
    return None

def collect_data(dataset_dir):
    def level_sort_key(name):
        num_str = re.sub(r'^[a-zA-Z]+', '', name)
        return int(num_str) if num_str.isdigit() else 9999

    folders = [d for d in os.listdir(dataset_dir) if d.startswith('level') and os.path.isdir(os.path.join(dataset_dir, d))]
    folders.sort(key=level_sort_key)

    results = []
    for folder in folders:
        if folder == 'level3AMD':
            continue
        case_dir = os.path.join(dataset_dir, folder)
        p_files = glob.glob(os.path.join(case_dir, "postProcessing", "probes", "*", "p"))
        if not p_files:
            continue
        time_p, p_vals = parse_probe_file(p_files[0])
        if p_vals is None or len(p_vals) < 4:
            continue
        cells = get_cell_count(case_dir)
        results.append({
            "level": folder,
            "cells": cells,
            "time": time_p,
            "p_m2_s2": p_vals[:4],
            "p_gauge_Pa": [p * RHO_INF for p in p_vals[:4]]
        })
    return results

def plot_single(pt_idx, data, output_dir):
    pt_meta = PROBE_POINTS[pt_idx]
    fig, ax1 = plt.subplots(figsize=(9.5, 6), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#fbfbfc')

    x_vals = [r["cells"] / 1000.0 for r in data if r["cells"]]
    y_vals = [r["p_m2_s2"][pt_idx] for r in data if r["cells"]]
    labels = [r["level"] for r in data if r["cells"]]

    ax1.plot(x_vals, y_vals, marker='s', markersize=8, linewidth=2.2,
             color='#1b5e20', linestyle='-', label='meshPrism (Prism Layers)', zorder=4)

    for x, y, lvl in zip(x_vals, y_vals, labels):
        ax1.annotate(lvl, (x, y), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=9, fontweight='bold', color='#1b5e20')

    ax1.set_xlabel('Cell Count (Thousands)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'Kinematic Pressure $p$ $[\mathrm{m^2/s^2}]$', fontsize=12, fontweight='bold')
    ax1.set_title(f'{pt_meta["name"]}\nCoords: {pt_meta["coords"]} | {pt_meta["desc"]}',
                  fontsize=12, fontweight='bold', pad=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best', framealpha=0.95)

    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * RHO_INF, y1_max * RHO_INF)
    ax2.set_ylabel(r'Gauge Pressure $P_{\mathrm{gauge}}$ $[\mathrm{Pa}]$', fontsize=11, color='#555555')

    fig.text(0.12, 0.02, '* Excludes level3AMD. Refinement progresses from level0 (coarse) to level6 (fine).',
             fontsize=8.5, color='#777777', style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out_file = os.path.join(output_dir, f"{pt_meta['file_prefix']}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def plot_combined(data, output_dir):
    fig, ax1 = plt.subplots(figsize=(11, 7), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#fbfbfc')

    colors = ['#d95f02', '#7570b3', '#e7298a', '#1b9e77']
    markers = ['o', 's', '^', 'D']

    x_vals = [r["cells"] / 1000.0 for r in data if r["cells"]]

    for i, pt_meta in enumerate(PROBE_POINTS):
        y_vals = [r["p_m2_s2"][i] for r in data if r["cells"]]
        ax1.plot(x_vals, y_vals, marker=markers[i], markersize=7, linewidth=2.0,
                 color=colors[i], label=pt_meta["short_name"])

    ax1.set_xlabel('Cell Count (Thousands)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'Kinematic Pressure $p$ $[\mathrm{m^2/s^2}]$', fontsize=12, fontweight='bold')
    ax1.set_title('meshPrism - Combined 4-Point Mesh Convergence Comparison\n(Excluding level3AMD)',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='best', framealpha=0.95, fontsize=10)

    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * RHO_INF, y1_max * RHO_INF)
    ax2.set_ylabel(r'Gauge Pressure $P_{\mathrm{gauge}}$ $[\mathrm{Pa}]$', fontsize=11, color='#555555')

    plt.tight_layout()
    out_file = os.path.join(output_dir, "combined_points_pressure.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")

def main():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data = collect_data(cur_dir)
    print(f"Loaded {len(data)} levels for meshPrism.")

    for idx in range(4):
        plot_single(idx, data, cur_dir)
    plot_combined(data, cur_dir)

if __name__ == "__main__":
    main()
