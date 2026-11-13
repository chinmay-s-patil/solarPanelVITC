#!/usr/bin/env python3
import os
import glob
import re
import csv
import sys
import matplotlib
matplotlib.use('Agg')  # Non-blocking headless rendering
import matplotlib.pyplot as plt

meshv2_dir = "/home/lavender/OpenFoamUbu/offProjects/solarPanel/meshConv/meshv2"
output_csv = os.path.join(meshv2_dir, "summary_forces.csv")
output_img = os.path.join(meshv2_dir, "dimensionless_forces_plot.png")

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

def generate_dimensionless_plot(results):
    """Generates dimensionless plot (V / V_0) across solved levels."""
    solved_rows = [r for r in results if r["Status"] == "Solved"]
    if not solved_rows:
        return

    levels_labels = [r["Level"] for r in solved_rows]
    indices = range(len(solved_rows))

    def safe_get(r, key):
        v = r[key]
        return float(v) if isinstance(v, (int, float)) else None

    cd_vals = [safe_get(r, "Cd") for r in solved_rows]
    cl_vals = [safe_get(r, "Cl") for r in solved_rows]
    cm_vals = [safe_get(r, "CmPitch") for r in solved_rows]
    drag_vals = [safe_get(r, "Drag_Force_N") for r in solved_rows]
    lift_vals = [safe_get(r, "Lift_Force_N") for r in solved_rows]

    cd_0 = cd_vals[0] if cd_vals and cd_vals[0] is not None else 1.0
    cl_0 = cl_vals[0] if cl_vals and cl_vals[0] is not None else 1.0
    cm_0 = cm_vals[0] if cm_vals and cm_vals[0] is not None else 1.0
    drag_0 = drag_vals[0] if drag_vals and drag_vals[0] is not None else 1.0
    lift_0 = lift_vals[0] if lift_vals and lift_vals[0] is not None else 1.0

    cd_norm = [v / cd_0 if v is not None else None for v in cd_vals]
    cl_norm = [v / cl_0 if v is not None else None for v in cl_vals]
    cm_norm = [v / cm_0 if v is not None else None for v in cm_vals]
    drag_norm = [v / drag_0 if v is not None else None for v in drag_vals]
    lift_norm = [v / lift_0 if v is not None else None for v in lift_vals]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Dimensionless Force & Coefficient Convergence (Normalized by Level 0: V / V_0)", fontsize=13, fontweight='bold')

    ax1.plot(indices, cd_norm, 'o-', color='#1f77b4', linewidth=2, markersize=7, label=f'Cd / Cd0 (Cd0 = {cd_0:.4f})')
    ax1.plot(indices, cl_norm, 's-', color='#ff7f0e', linewidth=2, markersize=7, label=f'Cl / Cl0 (Cl0 = {cl_0:.4f})')
    # if any(v is not None for v in cm_norm):
    #     ax1.plot(indices, cm_norm, '^--', color='#2ca02c', linewidth=2, markersize=7, label=f'CmPitch / Cm0 (Cm0 = {cm_0:.4f})')
    ax1.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax1.set_ylabel("Dimensionless Ratio (V / V0)", fontsize=11, fontweight='bold')
    ax1.set_title("Aerodynamic Coefficients (Normalized)", fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='best', frameon=True)

    ax2.plot(indices, drag_norm, 'd-', color='#d62728', linewidth=2, markersize=7, label=f'Drag / Drag0 (Drag0 = {drag_0:.1f} N)')
    ax2.plot(indices, lift_norm, 'v-', color='#9467bd', linewidth=2, markersize=7, label=f'Lift / Lift0 (Lift0 = {lift_0:.1f} N)')
    ax2.axhline(1.0, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel("Mesh Level", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Dimensionless Ratio (F / F0)", fontsize=11, fontweight='bold')
    ax2.set_title("Physical Forces (Normalized)", fontsize=11)
    ax2.set_xticks(list(indices))
    ax2.set_xticklabels(levels_labels, rotation=0, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='best', frameon=True)

    plt.tight_layout()
    plt.savefig(output_img, dpi=150)

    artifact_path = "/home/lavender/.gemini/antigravity/brain/6a87848b-5044-4992-b4cb-41b3c2b8bd0c/dimensionless_forces_plot.png"
    if os.path.exists(os.path.dirname(artifact_path)):
        plt.savefig(artifact_path, dpi=150)
    plt.close()

def main():
    levels = [f"level{i}" for i in range(15)]
    results = []

    rho = 1.225
    U = 65.0
    Aref = 1.389
    q_dyn = 0.5 * rho * (U ** 2) * Aref

    for lvl in levels:
        case_dir = os.path.join(meshv2_dir, lvl)
        if not os.path.exists(case_dir):
            continue

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
        results.append(row)

    fieldnames = [
        "Level", "Cells", "Status", "Time", "Cd", "Cl", "Cs", "CmPitch", "CmRoll", "CmYaw",
        "Drag_Force_N", "Lift_Force_N", "Side_Force_N",
        "Pressure_Fx_N", "Pressure_Fy_N", "Pressure_Fz_N",
        "Viscous_Fx_N", "Viscous_Fy_N", "Viscous_Fz_N"
    ]

    with open(output_csv, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Exported forces summary for meshv2 to:\n  {output_csv}\n")
    print(f"{'Level':<10} {'Cells':<10} {'Status':<10} {'Time':<8} {'Cd':<12} {'Cl':<12} {'Drag (N)':<14} {'Lift (N)':<14}")
    print("-" * 88)
    for r in results:
        cell_s = str(r['Cells']) if r['Cells'] != "" else "-"
        cd_s = f"{r['Cd']:.5f}" if isinstance(r['Cd'], float) else (str(r['Cd']) if r['Cd'] != "" else "-")
        cl_s = f"{r['Cl']:.5f}" if isinstance(r['Cl'], float) else (str(r['Cl']) if r['Cl'] != "" else "-")
        df_s = f"{r['Drag_Force_N']:.3f}" if isinstance(r['Drag_Force_N'], float) else (str(r['Drag_Force_N']) if r['Drag_Force_N'] != "" else "-")
        lf_s = f"{r['Lift_Force_N']:.3f}" if isinstance(r['Lift_Force_N'], float) else (str(r['Lift_Force_N']) if r['Lift_Force_N'] != "" else "-")
        t_s = str(r['Time']) if r['Time'] != "" else "-"
        print(f"{r['Level']:<10} {cell_s:<10} {r['Status']:<10} {t_s:<8} {cd_s:<12} {cl_s:<12} {df_s:<14} {lf_s:<14}")

    # Generate dimensionless plot
    generate_dimensionless_plot(results)

    # Print dimensionless ratio summary table
    solved = [r for r in results if r["Status"] == "Solved"]
    if solved and isinstance(solved[0]["Cd"], float):
        cd0 = solved[0]["Cd"]
        cl0 = solved[0]["Cl"]
        drag0 = solved[0]["Drag_Force_N"]
        lift0 = solved[0]["Lift_Force_N"]

        print("\n" + "=" * 80)
        print("Dimensionless Normalized Ratios (V / V_0 relative to Level 0):")
        print(f"{'Level':<10} {'Cells':<10} {'Cd / Cd0':<12} {'Cl / Cl0':<12} {'Drag / Drag0':<14} {'Lift / Lift0':<14}")
        print("-" * 80)
        for r in solved:
            cell_s = str(r['Cells']) if r['Cells'] != "" else "-"
            cd_r = f"{r['Cd']/cd0:.4f}" if isinstance(r['Cd'], float) and cd0 else "-"
            cl_r = f"{r['Cl']/cl0:.4f}" if isinstance(r['Cl'], float) and cl0 else "-"
            df_r = f"{r['Drag_Force_N']/drag0:.4f}" if isinstance(r['Drag_Force_N'], float) and drag0 else "-"
            lf_r = f"{r['Lift_Force_N']/lift0:.4f}" if isinstance(r['Lift_Force_N'], float) and lift0 else "-"
            print(f"{r['Level']:<10} {cell_s:<10} {cd_r:<12} {cl_r:<12} {df_r:<14} {lf_r:<14}")
        print("=" * 80)

if __name__ == "__main__":
    main()
