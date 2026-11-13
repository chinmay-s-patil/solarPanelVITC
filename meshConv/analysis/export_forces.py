#!/usr/bin/env python3
import os
import glob
import re
import csv
import sys
import matplotlib
matplotlib.use('Agg')  # Headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Base directories
analysis_dir = os.path.dirname(os.path.abspath(__file__))
meshconv_dir = os.path.dirname(analysis_dir)
meshv2_dir = os.path.join(meshconv_dir, "meshv2")
meshprism_dir = os.path.join(meshconv_dir, "meshPrism")

output_csv = os.path.join(analysis_dir, "summary_forces.csv")
output_img = os.path.join(analysis_dir, "dimensionless_forces_plot.png")
output_img_legacy = os.path.join(analysis_dir, "comparison_forces_plot.png")
artifact_img = None

def get_file_max_time_and_last_line(filepath):
    """Reads a dat file and returns (max_time, header_list, last_data_line_str)."""
    if not os.path.exists(filepath):
        return -1.0, [], None

    headers = []
    last_line = None
    max_time = -1.0

    with open(filepath, 'r') as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith('#'):
                if 'Time' in line_str and ('Cd' in line_str or 'Cl' in line_str or 'total_x' in line_str):
                    headers = [h.strip() for h in re.split(r'\s+', line_str.lstrip('#').strip())]
            else:
                last_line = line_str
                parts = re.split(r'\s+', line_str)
                try:
                    tval = float(parts[0])
                    if tval > max_time:
                        max_time = tval
                except ValueError:
                    pass

    return max_time, headers, last_line

def parse_coefficient_file(filepath):
    """Parses OpenFOAM forceCoeffs coefficient.dat file for latest iteration values."""
    max_t, headers, last_line = get_file_max_time_and_last_line(filepath)
    if not last_line:
        return {}

    values = [v.strip() for v in re.split(r'\s+', last_line)]
    res = {}
    if headers and len(values) == len(headers):
        for h, v in zip(headers, values):
            try:
                res[h] = float(v)
            except ValueError:
                res[h] = v
    else:
        default_headers = ["Time", "Cd", "Cd(f)", "Cd(r)", "Cl", "Cl(f)", "Cl(r)", "CmPitch", "CmRoll", "CmYaw", "Cs", "Cs(f)", "Cs(r)"]
        try:
            floats = [float(v) for v in values]
            for i, h in enumerate(default_headers):
                if i < len(floats):
                    res[h] = floats[i]
        except ValueError:
            pass
    return res

def parse_force_file(filepath):
    """Parses OpenFOAM forces force.dat file for latest iteration values (forces in N)."""
    max_t, headers, last_line = get_file_max_time_and_last_line(filepath)
    if not last_line:
        return {}

    parts = re.split(r'\s+', last_line)
    cleaned_parts = [p.replace('(', '').replace(')', '') for p in parts if p.replace('(', '').replace(')', '') != '']

    try:
        floats = [float(p) for p in cleaned_parts]
        if len(floats) >= 10:
            return {
                "Time": floats[0],
                "Total_Fx": floats[1],
                "Total_Fy": floats[2],
                "Total_Fz": floats[3],
                "Pressure_Fx": floats[4],
                "Pressure_Fy": floats[5],
                "Pressure_Fz": floats[6],
                "Viscous_Fx": floats[7],
                "Viscous_Fy": floats[8],
                "Viscous_Fz": floats[9],
            }
    except ValueError:
        pass
    return {}

def find_best_file(case_dir, sub_type):
    """Finds the dat file with the highest solved Time step under postProcessing/sub_type/."""
    post_dir = os.path.join(case_dir, "postProcessing", sub_type)
    if not os.path.exists(post_dir):
        return None

    candidates = []
    for root, dirs, files in os.walk(post_dir):
        for f in files:
            if f.endswith(".dat"):
                fp = os.path.join(root, f)
                max_t, _, _ = get_file_max_time_and_last_line(fp)
                if max_t >= 0:
                    candidates.append((max_t, fp))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]

def get_cell_count(case_dir):
    """Reads cell count from log.checkMesh if present."""
    log_cm = os.path.join(case_dir, "log.checkMesh")
    if os.path.exists(log_cm):
        with open(log_cm, 'r') as fp:
            for line in fp:
                if 'cells:' in line:
                    m = re.search(r'cells:\s*(\d+)', line)
                    if m:
                        return int(m.group(1))
    return None

def scan_dataset(dataset_name, dataset_dir):
    """Scans all level directories within a dataset directory."""
    if not os.path.exists(dataset_dir):
        return []

    dir_entries = [d for d in os.listdir(dataset_dir) if re.match(r"^level[\d\.]+$", d)]
    dir_entries.sort(key=lambda x: float(re.sub(r"^[a-zA-Z]+", "", x)))

    rho = 1.225
    U = 65.0
    Aref = 1.389
    q_dyn = 0.5 * rho * (U ** 2) * Aref

    rows = []
    for lvl in dir_entries:
        case_dir = os.path.join(dataset_dir, lvl)
        coeff_file = find_best_file(case_dir, "forceCoeffs")
        force_file = find_best_file(case_dir, "forces")
        ncells = get_cell_count(case_dir)

        coeff_data = parse_coefficient_file(coeff_file) if coeff_file else {}
        force_data = parse_force_file(force_file) if force_file else {}

        solved = bool(coeff_data or force_data)

        cd_val = coeff_data.get("Cd", "")
        cl_val = coeff_data.get("Cl", "")
        cs_val = coeff_data.get("Cs", "")

        drag_n = force_data.get("Total_Fx", "")
        lift_n = force_data.get("Total_Fy", "")
        side_n = force_data.get("Total_Fz", "")

        if drag_n == "" and isinstance(cd_val, float):
            drag_n = cd_val * q_dyn
        if lift_n == "" and isinstance(cl_val, float):
            lift_n = cl_val * q_dyn
        if side_n == "" and isinstance(cs_val, float):
            side_n = cs_val * q_dyn

        row = {
            "Dataset": dataset_name,
            "Level": lvl,
            "Cells": ncells if ncells is not None else "",
            "Status": "Solved" if solved else "Unrun",
            "Time": coeff_data.get("Time", force_data.get("Time", "")),
            "Cd": cd_val,
            "Cl": cl_val,
            "Cs": cs_val,
            "CmPitch": coeff_data.get("CmPitch", ""),
            "CmRoll": coeff_data.get("CmRoll", ""),
            "CmYaw": coeff_data.get("CmYaw", ""),
            "Drag_Force_N": drag_n,
            "Lift_Force_N": lift_n,
            "Side_Force_N": side_n,
            "Pressure_Fx_N": force_data.get("Pressure_Fx", ""),
            "Pressure_Fy_N": force_data.get("Pressure_Fy", ""),
            "Pressure_Fz_N": force_data.get("Pressure_Fz", ""),
            "Viscous_Fx_N": force_data.get("Viscous_Fx", ""),
            "Viscous_Fy_N": force_data.get("Viscous_Fy", ""),
            "Viscous_Fz_N": force_data.get("Viscous_Fz", "")
        }
        rows.append(row)

    return rows

def generate_normalized_plots(no_prism_rows, prism_rows, cd0, cl0, drag0, lift0):
    """Generates dimensionless plots (V / V_0) with cell count on the x-axis,

    normalized by meshv2 (noPrism) level 0 baseline values.
    """
    np_solved = [r for r in no_prism_rows if r["Status"] == "Solved" and isinstance(r["Cells"], int)]
    p_solved = [r for r in prism_rows if r["Status"] == "Solved" and isinstance(r["Cells"], int)]

    np_solved.sort(key=lambda r: r["Cells"])
    p_solved.sort(key=lambda r: r["Cells"])

    if not np_solved and not p_solved:
        print("No solved cases available to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    fig.suptitle(
        f"Dimensionless Convergence Comparison: noPrism vs. Prism\n"
        f"Normalized by noPrism Level 0 (Cd0 = {cd0:.5f}, Cl0 = {cl0:.5f}) vs. Cell Count",
        fontsize=13,
        fontweight='bold'
    )

    # Color Palette:
    # noPrism: Blue tones
    # prism: Red / Orange tones
    c_np_cd = '#1f77b4'
    c_np_cl = '#028090'
    c_p_cd = '#d62728'
    c_p_cl = '#e67e22'

    # --- Plot 1: Dimensionless Coefficients (Cd / Cd0 and Cl / Cl0) ---
    if np_solved:
        np_cells = [r["Cells"] for r in np_solved]
        np_cd_norm = [r["Cd"] / cd0 for r in np_solved if isinstance(r["Cd"], (int, float))]
        np_cl_norm = [r["Cl"] / cl0 for r in np_solved if isinstance(r["Cl"], (int, float))]
        ax1.plot(np_cells[:len(np_cd_norm)], np_cd_norm, 'o-', color=c_np_cd, linewidth=2, markersize=7, label=f'noPrism: Cd / Cd0')
        ax1.plot(np_cells[:len(np_cl_norm)], np_cl_norm, 's--', color=c_np_cl, linewidth=2, markersize=7, label=f'noPrism: Cl / Cl0')

    if p_solved:
        p_cells = [r["Cells"] for r in p_solved]
        p_cd_norm = [r["Cd"] / cd0 for r in p_solved if isinstance(r["Cd"], (int, float))]
        p_cl_norm = [r["Cl"] / cl0 for r in p_solved if isinstance(r["Cl"], (int, float))]
        ax1.plot(p_cells[:len(p_cd_norm)], p_cd_norm, 'D-', color=c_p_cd, linewidth=2.5, markersize=8, label=f'prism: Cd / Cd0')
        ax1.plot(p_cells[:len(p_cl_norm)], p_cl_norm, '^--', color=c_p_cl, linewidth=2.5, markersize=8, label=f'prism: Cl / Cl0')

    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.8, linewidth=1.5, label='Baseline (1.0 = noPrism L0)')
    ax1.set_ylabel("Dimensionless Ratio (V / V0)", fontsize=11, fontweight='bold')
    ax1.set_title("Aerodynamic Coefficients (Normalized by noPrism Level 0)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='center right', frameon=True, fontsize=10)

    # --- Plot 2: Dimensionless Physical Forces (Drag / Drag0 and Lift / Lift0) ---
    c_np_drag = '#20639b'
    c_np_lift = '#3caea3'
    c_p_drag = '#c0392b'
    c_p_lift = '#d35400'

    if np_solved:
        np_cells = [r["Cells"] for r in np_solved]
        np_drag_norm = [r["Drag_Force_N"] / drag0 for r in np_solved if isinstance(r["Drag_Force_N"], (int, float))]
        np_lift_norm = [r["Lift_Force_N"] / lift0 for r in np_solved if isinstance(r["Lift_Force_N"], (int, float))]
        ax2.plot(np_cells[:len(np_drag_norm)], np_drag_norm, 'o-', color=c_np_drag, linewidth=2, markersize=7, label=f'noPrism: Drag / Drag0 (Drag0 = {drag0:.1f} N)')
        ax2.plot(np_cells[:len(np_lift_norm)], np_lift_norm, 's--', color=c_np_lift, linewidth=2, markersize=7, label=f'noPrism: Lift / Lift0 (Lift0 = {lift0:.1f} N)')

    if p_solved:
        p_cells = [r["Cells"] for r in p_solved]
        p_drag_norm = [r["Drag_Force_N"] / drag0 for r in p_solved if isinstance(r["Drag_Force_N"], (int, float))]
        p_lift_norm = [r["Lift_Force_N"] / lift0 for r in p_solved if isinstance(r["Lift_Force_N"], (int, float))]
        ax2.plot(p_cells[:len(p_drag_norm)], p_drag_norm, 'D-', color=c_p_drag, linewidth=2.5, markersize=8, label=f'prism: Drag / Drag0')
        ax2.plot(p_cells[:len(p_lift_norm)], p_lift_norm, '^--', color=c_p_lift, linewidth=2.5, markersize=8, label=f'prism: Lift / Lift0')

    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.8, linewidth=1.5, label='Baseline (1.0 = noPrism L0)')
    ax2.set_xlabel("Cell Count", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Dimensionless Ratio (F / F0)", fontsize=11, fontweight='bold')
    ax2.set_title("Physical Forces (Normalized by noPrism Level 0)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='center right', frameon=True, fontsize=10)

    # Format x-axis with k / M notation
    def cell_formatter(x, pos):
        if x >= 1e6:
            return f'{x*1e-6:.2f}M'
        elif x >= 1e3:
            return f'{x*1e-3:.0f}k'
        return f'{x:.0f}'

    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(cell_formatter))

    plt.tight_layout()
    plt.savefig(output_img, dpi=180)
    plt.savefig(output_img_legacy, dpi=180)
    # print(f"Saved dimensionless comparison plot to:\n  {output_img}")

    if artifact_img and os.path.exists(os.path.dirname(artifact_img)):
        plt.savefig(artifact_img, dpi=180)
        # print(f"Saved artifact copy to:\n  {artifact_img}")

    plt.close()

def main():
    print("=" * 85)
    print("  Dimensionless Mesh Convergence: noPrism (meshv2) vs. prism (meshPrism)")
    print("  Normalized by noPrism (meshv2) Level 0 Baseline Values")
    print("=" * 85)

    no_prism_rows = scan_dataset("noPrism", meshv2_dir)
    prism_rows = scan_dataset("prism", meshprism_dir)

    # Identify Level 0 baseline from meshv2 (noPrism)
    base_l0 = None
    for r in no_prism_rows:
        if r["Level"] == "level0" and r["Status"] == "Solved" and isinstance(r["Cd"], float):
            base_l0 = r
            break

    if not base_l0:
        print("ERROR: Could not find solved level0 in meshv2 (noPrism) for baseline normalization!")
        sys.exit(1)

    cd0 = base_l0["Cd"]
    cl0 = base_l0["Cl"]
    drag0 = base_l0["Drag_Force_N"]
    lift0 = base_l0["Lift_Force_N"]

    print(f"\nBaseline Values from noPrism (meshv2) level0 ({base_l0['Cells']:,} cells):")
    print(f"  Cd0    = {cd0:.6f}")
    print(f"  Cl0    = {cl0:.6f}")
    print(f"  Drag0  = {drag0:.3f} N")
    print(f"  Lift0  = {lift0:.3f} N\n")

    # Add normalized columns to all rows
    all_rows = no_prism_rows + prism_rows
    for r in all_rows:
        if r["Status"] == "Solved" and isinstance(r["Cd"], (int, float)):
            r["Cd_norm"] = r["Cd"] / cd0
            r["Cl_norm"] = r["Cl"] / cl0
            r["Drag_norm"] = r["Drag_Force_N"] / drag0 if isinstance(r["Drag_Force_N"], (int, float)) and drag0 else ""
            r["Lift_norm"] = r["Lift_Force_N"] / lift0 if isinstance(r["Lift_Force_N"], (int, float)) and lift0 else ""
        else:
            r["Cd_norm"] = ""
            r["Cl_norm"] = ""
            r["Drag_norm"] = ""
            r["Lift_norm"] = ""

    fieldnames = [
        "Dataset", "Level", "Cells", "Status", "Time",
        "Cd", "Cl", "Cd_norm", "Cl_norm", "Cs",
        "CmPitch", "CmRoll", "CmYaw",
        "Drag_Force_N", "Lift_Force_N", "Drag_norm", "Lift_norm", "Side_Force_N",
        "Pressure_Fx_N", "Pressure_Fy_N", "Pressure_Fz_N",
        "Viscous_Fx_N", "Viscous_Fy_N", "Viscous_Fz_N"
    ]

    with open(output_csv, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Saved summary CSV (with normalized ratios) to:\n  {output_csv}\n")

    # Print clean formatted tables
    print(f"{'Dataset':<9} {'Level':<8} {'Cells':<10} {'Status':<8} {'Time':<6} {'Cd':<10} {'Cd/Cd0':<9} {'Cl':<10} {'Cl/Cl0':<9} {'Drag(N)':<10} {'Lift(N)':<10}")
    print("-" * 105)
    for r in all_rows:
        cell_s = f"{r['Cells']:,}" if isinstance(r['Cells'], int) else (str(r['Cells']) if r['Cells'] != "" else "-")
        cd_s = f"{r['Cd']:.5f}" if isinstance(r['Cd'], float) else (str(r['Cd']) if r['Cd'] != "" else "-")
        cdn_s = f"{r['Cd_norm']:.4f}" if isinstance(r['Cd_norm'], float) else "-"
        cl_s = f"{r['Cl']:.5f}" if isinstance(r['Cl'], float) else (str(r['Cl']) if r['Cl'] != "" else "-")
        cln_s = f"{r['Cl_norm']:.4f}" if isinstance(r['Cl_norm'], float) else "-"
        df_s = f"{r['Drag_Force_N']:.1f}" if isinstance(r['Drag_Force_N'], float) else (str(r['Drag_Force_N']) if r['Drag_Force_N'] != "" else "-")
        lf_s = f"{r['Lift_Force_N']:.1f}" if isinstance(r['Lift_Force_N'], float) else (str(r['Lift_Force_N']) if r['Lift_Force_N'] != "" else "-")
        t_s = f"{r['Time']:.0f}" if isinstance(r['Time'], float) else (str(r['Time']) if r['Time'] != "" else "-")
        print(f"{r['Dataset']:<9} {r['Level']:<8} {cell_s:<10} {r['Status']:<8} {t_s:<6} {cd_s:<10} {cdn_s:<9} {cl_s:<10} {cln_s:<9} {df_s:<10} {lf_s:<10}")

    # Generate normalized plots
    generate_normalized_plots(no_prism_rows, prism_rows, cd0, cl0, drag0, lift0)

    # Print comparison summary for solved levels
    np_solved = [r for r in no_prism_rows if r["Status"] == "Solved" and isinstance(r["Cd"], float)]
    p_solved = [r for r in prism_rows if r["Status"] == "Solved" and isinstance(r["Cd"], float)]

    # print("\n" + "=" * 85)
    # print("  Comparison Summary (Normalized by noPrism Level 0):")
    # print("=" * 85)
    # for pr in p_solved:
    #     closest_np = min(np_solved, key=lambda x: abs(x["Cells"] - pr["Cells"])) if np_solved else None
    #     print(f"Prism {pr['Level']} ({pr['Cells']:,} cells):")
    #     print(f"  Prism:   Cd = {pr['Cd']:.5f} (Cd/Cd0 = {pr['Cd_norm']:.4f}), Cl = {pr['Cl']:.5f} (Cl/Cl0 = {pr['Cl_norm']:.4f})")
    #     if closest_np:
    #         delta_cd_norm = ((pr['Cd_norm'] - closest_np['Cd_norm']) / closest_np['Cd_norm']) * 100
    #         delta_cl_norm = ((pr['Cl_norm'] - closest_np['Cl_norm']) / closest_np['Cl_norm']) * 100
    #         print(f"  Closest noPrism is {closest_np['Level']} ({closest_np['Cells']:,} cells):")
    #         print(f"  noPrism: Cd = {closest_np['Cd']:.5f} (Cd/Cd0 = {closest_np['Cd_norm']:.4f}), Cl = {closest_np['Cl']:.5f} (Cl/Cl0 = {closest_np['Cl_norm']:.4f})")
    #         print(f"  Relative difference: Δ(Cd/Cd0) = {delta_cd_norm:+.2f}%, Δ(Cl/Cl0) = {delta_cl_norm:+.2f}%")
    # print("=" * 85)

if __name__ == "__main__":
    main()
