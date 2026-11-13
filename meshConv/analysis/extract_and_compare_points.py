#!/usr/bin/env python3
"""
Mesh Convergence Analysis across 4 Spatial Probe Points
Compares meshPrism and meshv2 across all solved levels (excluding level3AMD).
Extracts pressure and velocity at 4 aerodynamic locations around the solar panel:
  Point 1: (0.50, 0.55, 0.0)  - Upstream Stagnation / Leading Edge
  Point 2: (0.00, 0.80, 0.0)  - Upper Surface Suction / Boundary Layer
  Point 3: (-0.80, 0.70, 0.0) - Leeward Wake / Recirculation Zone
  Point 4: (0.20, 0.45, 0.0)  - Under-Panel Venturi Flow
"""

import os
import glob
import re
import csv
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Air density matching OpenFOAM controlDict rhoInf
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
    """Parses OpenFOAM probe file and returns (time, list_of_scalar_values)."""
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

def parse_vector_probe_file(filepath):
    """Parses OpenFOAM vector probe file and returns (time, list_of_(Ux, Uy, Uz)_tuples)."""
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
    
    vec_matches = re.findall(r'\(([^)]+)\)', last_line)
    time_match = re.match(r'^([\d\.\+\-eE]+)', last_line)
    time_val = float(time_match.group(1)) if time_match else 0.0
    
    vectors = []
    for vm in vec_matches:
        v_parts = [float(x) for x in vm.strip().split()]
        vectors.append(tuple(v_parts))
    return time_val, vectors

def get_cell_count(case_dir):
    """Extracts cell count from log.checkMesh or log.snappyHexMesh."""
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
            lines = fp.readlines()
            for line in reversed(lines):
                if 'cells:' in line:
                    m = re.search(r'cells:\s*(\d+)', line)
                    if m:
                        return int(m.group(1))
    return None

def collect_dataset_data(dataset_name, dataset_dir):
    """Gathers all solved levels and their probe data."""
    results = []
    if not os.path.exists(dataset_dir):
        return results

    def level_sort_key(name):
        num_str = re.sub(r'^[a-zA-Z]+', '', name)
        try:
            return float(num_str)
        except ValueError:
            return 9999.0

    folders = [d for d in os.listdir(dataset_dir) if d.startswith('level') and os.path.isdir(os.path.join(dataset_dir, d))]
    folders.sort(key=level_sort_key)

    for folder in folders:
        # Strictly skip level3AMD
        if folder == 'level3AMD':
            continue

        case_dir = os.path.join(dataset_dir, folder)
        probes_dir = os.path.join(case_dir, "postProcessing", "probes")
        
        p_files = glob.glob(os.path.join(probes_dir, "*", "p"))
        if not p_files:
            continue

        p_file = p_files[0]
        time_p, p_vals = parse_probe_file(p_file)
        if p_vals is None or len(p_vals) < 4:
            continue

        u_files = glob.glob(os.path.join(probes_dir, "*", "U"))
        u_vecs = []
        if u_files:
            _, u_vecs = parse_vector_probe_file(u_files[0])

        cells = get_cell_count(case_dir)

        row = {
            "dataset": dataset_name,
            "level": folder,
            "level_idx": level_sort_key(folder),
            "cells": cells,
            "time": time_p,
            "p_m2_s2": p_vals[:4],
            "p_gauge_Pa": [p * RHO_INF for p in p_vals[:4]],
            "u_vectors": u_vecs[:4] if len(u_vecs) >= 4 else [(None, None, None)] * 4,
            "u_mag": [np.sqrt(u[0]**2 + u[1]**2 + u[2]**2) if u[0] is not None else None for u in u_vecs[:4]] if len(u_vecs) >= 4 else [None] * 4
        }
        results.append(row)

    return results

def save_csv_summary(all_rows, output_path):
    """Exports structured probe data across all levels to CSV."""
    with open(output_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        header = [
            "Dataset", "Level", "Cells", "Time",
            "P1_coords", "P1_kinematic_p_m2_s2", "P1_gauge_Pa", "P1_magU_mps",
            "P2_coords", "P2_kinematic_p_m2_s2", "P2_gauge_Pa", "P2_magU_mps",
            "P3_coords", "P3_kinematic_p_m2_s2", "P3_gauge_Pa", "P3_magU_mps",
            "P4_coords", "P4_kinematic_p_m2_s2", "P4_gauge_Pa", "P4_magU_mps"
        ]
        writer.writerow(header)
        for r in all_rows:
            p_k = r["p_m2_s2"]
            p_g = r["p_gauge_Pa"]
            u_m = r["u_mag"]
            row = [
                r["dataset"], r["level"], r["cells"], r["time"],
                str(PROBE_POINTS[0]["coords"]), f"{p_k[0]:.4f}", f"{p_g[0]:.2f}", f"{u_m[0]:.3f}" if u_m[0] else "",
                str(PROBE_POINTS[1]["coords"]), f"{p_k[1]:.4f}", f"{p_g[1]:.2f}", f"{u_m[1]:.3f}" if u_m[1] else "",
                str(PROBE_POINTS[2]["coords"]), f"{p_k[2]:.4f}", f"{p_g[2]:.2f}", f"{u_m[2]:.3f}" if u_m[2] else "",
                str(PROBE_POINTS[3]["coords"]), f"{p_k[3]:.4f}", f"{p_g[3]:.2f}", f"{u_m[3]:.3f}" if u_m[3] else ""
            ]
            writer.writerow(row)
    print(f"Summary CSV exported: {output_path}")

def plot_single_point(pt_idx, prism_data, v2_data, output_dir, file_prefix=None):
    """Plots convergence for a single probe point across both meshPrism and meshv2."""
    pt_meta = PROBE_POINTS[pt_idx]
    coords = pt_meta["coords"]
    pt_name = pt_meta["name"]
    desc = pt_meta["desc"]
    if file_prefix is None:
        file_prefix = pt_meta["file_prefix"]

    fig, ax1 = plt.subplots(figsize=(10, 6.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#fbfbfc')

    prism_color = '#1b5e20'   # Forest Green
    v2_color = '#0d47a1'      # Royal Blue

    # Plot meshv2
    if v2_data:
        x_v2 = [r["cells"] / 1000.0 for r in v2_data if r["cells"]]
        y_v2 = [r["p_m2_s2"][pt_idx] for r in v2_data if r["cells"]]
        levels_v2 = [r["level"] for r in v2_data if r["cells"]]
        ax1.plot(x_v2, y_v2, marker='o', markersize=7, linewidth=2.0,
                 color=v2_color, label='meshv2 (noPrism)', alpha=0.9, zorder=4)
        for x, y, lvl in zip(x_v2, y_v2, levels_v2):
            if lvl in ['level0', 'level4', 'level8', 'level12']:
                ax1.annotate(lvl, (x, y), textcoords="offset points", xytext=(0, 8),
                             ha='center', fontsize=8, color=v2_color, fontweight='bold')

    # Plot meshPrism
    if prism_data:
        x_pr = [r["cells"] / 1000.0 for r in prism_data if r["cells"]]
        y_pr = [r["p_m2_s2"][pt_idx] for r in prism_data if r["cells"]]
        levels_pr = [r["level"] for r in prism_data if r["cells"]]
        ax1.plot(x_pr, y_pr, marker='s', markersize=7, linewidth=2.0,
                 color=prism_color, linestyle='--', label='meshPrism (Prism Layers)', alpha=0.9, zorder=5)
        for x, y, lvl in zip(x_pr, y_pr, levels_pr):
            ax1.annotate(lvl, (x, y), textcoords="offset points", xytext=(0, -13),
                         ha='center', fontsize=8, color=prism_color, fontweight='bold')

    ax1.set_xlabel('Total Cell Count (Thousands)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel(r'Kinematic Pressure $p$ $[\mathrm{m^2/s^2}]$', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_title(f'{pt_name}\nLocation: $(x={coords[0]:.2f}, y={coords[1]:.2f}, z={coords[2]:.2f})\,\mathrm{{m}}$ | {desc}',
                  fontsize=13, fontweight='bold', pad=14)

    ax1.grid(True, linestyle=':', alpha=0.6, color='#bbbbbb')
    ax1.legend(loc='best', frameon=True, framealpha=0.95, facecolor='white', edgecolor='#cccccc', fontsize=10)

    # Secondary Y-axis for Gauge Pressure P = rho * p (Pa)
    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * RHO_INF, y1_max * RHO_INF)
    ax2.set_ylabel(r'Gauge Pressure $P_{\mathrm{gauge}}$ $[\mathrm{Pa}]$ ($\rho = 1.225\,\mathrm{kg/m^3}$)',
                   fontsize=11, color='#555555', labelpad=10)
    ax2.tick_params(colors='#555555')

    fig.text(0.12, 0.02, '* Excluding level3AMD. Refinement progresses from coarse (left) to fine (right).',
             fontsize=8.5, color='#777777', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_file = os.path.join(output_dir, f"{file_prefix}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")
    return out_file

def plot_combined_all_points(prism_data, v2_data, output_dir):
    """Plots all 4 points together on a combined multi-line convergence plot."""
    fig, ax1 = plt.subplots(figsize=(12, 7.5), dpi=300)
    fig.patch.set_facecolor('#ffffff')
    ax1.set_facecolor('#fbfbfc')

    colors = ['#d95f02', '#7570b3', '#e7298a', '#1b9e77']
    markers = ['o', 's', '^', 'D']

    for i, pt_meta in enumerate(PROBE_POINTS):
        c = colors[i]
        m = markers[i]
        label_base = pt_meta["short_name"]

        # meshv2
        if v2_data:
            x_v2 = [r["cells"] / 1000.0 for r in v2_data if r["cells"]]
            y_v2 = [r["p_m2_s2"][i] for r in v2_data if r["cells"]]
            ax1.plot(x_v2, y_v2, marker=m, markersize=6, linewidth=1.8,
                     color=c, label=f'{label_base} (v2)', alpha=0.85)

        # meshPrism
        if prism_data:
            x_pr = [r["cells"] / 1000.0 for r in prism_data if r["cells"]]
            y_pr = [r["p_m2_s2"][i] for r in prism_data if r["cells"]]
            ax1.plot(x_pr, y_pr, marker=m, markersize=6, linewidth=1.8,
                     linestyle='--', color=c, label=f'{label_base} (prism)', alpha=0.95)

    ax1.set_xlabel('Total Cell Count (Thousands)', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_ylabel(r'Kinematic Pressure $p$ $[\mathrm{m^2/s^2}]$', fontsize=12, fontweight='bold', labelpad=8)
    ax1.set_title('Combined 4-Point Mesh Convergence Comparison\n(meshPrism vs meshv2 across all solved levels, excl. level3AMD)',
                  fontsize=13, fontweight='bold', pad=14)

    ax1.grid(True, linestyle=':', alpha=0.6, color='#bbbbbb')
    ax1.legend(loc='center left', bbox_to_anchor=(1.08, 0.5), frameon=True,
               framealpha=0.95, facecolor='white', edgecolor='#cccccc', fontsize=9.5)

    ax2 = ax1.twinx()
    y1_min, y1_max = ax1.get_ylim()
    ax2.set_ylim(y1_min * RHO_INF, y1_max * RHO_INF)
    ax2.set_ylabel(r'Gauge Pressure $P_{\mathrm{gauge}}$ $[\mathrm{Pa}]$ ($\rho = 1.225\,\mathrm{kg/m^3}$)',
                   fontsize=11, color='#555555', labelpad=10)
    ax2.tick_params(colors='#555555')

    fig.text(0.08, 0.02, 'Solid line = meshv2 | Dashed line = meshPrism | Excludes level3AMD',
             fontsize=8.5, color='#777777', style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out_file = os.path.join(output_dir, "combined_points_pressure.png")
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_file}")
    return out_file

def plot_4panel_dashboard(prism_data, v2_data, output_dir):
    """Plots a 2x2 grid displaying each point in its own dedicated subplot."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    fig.patch.set_facecolor('#ffffff')

    prism_color = '#1b5e20'   # Forest Green
    v2_color = '#0d47a1'      # Royal Blue

    for i, pt_meta in enumerate(PROBE_POINTS):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        ax.set_facecolor('#fbfbfc')

        # meshv2
        if v2_data:
            x_v2 = [r["cells"] / 1000.0 for r in v2_data if r["cells"]]
            y_v2 = [r["p_m2_s2"][i] for r in v2_data if r["cells"]]
            ax.plot(x_v2, y_v2, marker='o', markersize=6, linewidth=1.8,
                    color=v2_color, label='meshv2 (noPrism)', alpha=0.9)

        # meshPrism
        if prism_data:
            x_pr = [r["cells"] / 1000.0 for r in prism_data if r["cells"]]
            y_pr = [r["p_m2_s2"][i] for r in prism_data if r["cells"]]
            ax.plot(x_pr, y_pr, marker='s', markersize=6, linewidth=1.8,
                    linestyle='--', color=prism_color, label='meshPrism', alpha=0.9)

        ax.set_title(f'{pt_meta["name"]}\n{pt_meta["coords"]} | {pt_meta["desc"]}',
                     fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel('Cell Count (kCells)', fontsize=10, fontweight='bold')
        ax.set_ylabel(r'$p$ $[\mathrm{m^2/s^2}]$', fontsize=10, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.6, color='#bbbbbb')
        ax.legend(loc='best', fontsize=8.5, framealpha=0.9)

        ax_sec = ax.twinx()
        y1_min, y1_max = ax.get_ylim()
        ax_sec.set_ylim(y1_min * RHO_INF, y1_max * RHO_INF)
        ax_sec.set_ylabel(r'$P_{\mathrm{gauge}}$ $[\mathrm{Pa}]$', fontsize=9, color='#666666')
        ax_sec.tick_params(colors='#666666', labelsize=8)

    plt.suptitle('4-Point Aerodynamic Mesh Convergence Dashboard\nmeshPrism vs meshv2 (Excluding level3AMD)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out_file = os.path.join(output_dir, "combined_4panel_dashboard.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")
    return out_file

def main():
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(analysis_dir)
    prism_dir = os.path.join(base_dir, "meshPrism")
    v2_dir = os.path.join(base_dir, "meshv2")

    print("Collecting probe and cell data for meshPrism...")
    prism_data = collect_dataset_data("prism", prism_dir)
    print(f"Found {len(prism_data)} solved levels in meshPrism.")

    print("Collecting probe and cell data for meshv2...")
    v2_data = collect_dataset_data("v2", v2_dir)
    print(f"Found {len(v2_data)} solved levels in meshv2.")

    all_data = prism_data + v2_data
    csv_out = os.path.join(analysis_dir, "points_pressure_summary.csv")
    save_csv_summary(all_data, csv_out)

    # 1. Save 4 individual point photos
    print("\nGenerating 4 individual point plots...")
    for idx in range(4):
        plot_single_point(idx, prism_data, v2_data, analysis_dir)

    # 2. Save combined plots
    print("\nGenerating combined plots...")
    plot_combined_all_points(prism_data, v2_data, analysis_dir)
    plot_4panel_dashboard(prism_data, v2_data, analysis_dir)

    print("\nAll plots generated successfully in:", analysis_dir)

if __name__ == "__main__":
    main()
