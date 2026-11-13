#!/usr/bin/env python3
"""
plot_forces.py
Reads OpenFOAM forces and force coefficients from postProcessing and plots them over time (iterations).
Designed for meshPrism level cases (e.g. level4).

Usage:
    python3 plot_forces.py [options]

Options:
    --case <path>         Case directory (default: directory containing this script)
    --settled-time <val>  Starting iteration for the settled regime plot (default: 100)
    --zoom-last <val>     Number of final iterations for convergence zoom plot (default: 300)
    --output-dir <path>   Directory to save output plots (default: case directory)
"""

import os
import sys
import glob
import re
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for terminal / server execution
import matplotlib.pyplot as plt

def parse_header_and_data(filepath):
    """
    Parses OpenFOAM dat file, correctly extracting column headers and numeric rows.
    Handles multiple time blocks or restarts within a single file.
    """
    if not os.path.exists(filepath):
        return None, None

    headers = []
    data_rows = []

    with open(filepath, 'r') as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                continue

            if line_str.startswith('#'):
                clean_header = line_str.lstrip('#').strip()
                if 'Time' in clean_header and ('total' in clean_header or 'Cd' in clean_header or 'Cl' in clean_header):
                    headers = [h.strip() for h in re.split(r'\s+', clean_header)]
                continue

            # Strip parentheses and split
            cleaned_line = line_str.replace('(', ' ').replace(')', ' ').strip()
            parts = [p for p in re.split(r'\s+', cleaned_line) if p]
            try:
                numeric_vals = [float(p) for p in parts]
                data_rows.append(numeric_vals)
            except ValueError:
                continue

    if not data_rows:
        return None, None

    data_arr = np.array(data_rows, dtype=float)
    # Sort by time (column 0) and remove duplicate time entries (keeping last)
    _, unique_indices = np.unique(data_arr[:, 0], return_index=True)
    data_arr = data_arr[unique_indices]
    data_arr = data_arr[np.argsort(data_arr[:, 0])]

    return headers, data_arr

def load_postprocessing_data(case_dir, subfolder, filename):
    """
    Searches all time subfolders in postProcessing/<subfolder>/*/filename
    and merges them chronologically.
    """
    search_path = os.path.join(case_dir, "postProcessing", subfolder, "*", filename)
    matched_files = sorted(glob.glob(search_path), key=lambda p: os.path.getmtime(p))

    if not matched_files:
        return None, None

    all_data = []
    final_headers = None

    for fpath in matched_files:
        headers, data = parse_header_and_data(fpath)
        if data is not None and len(data) > 0:
            all_data.append(data)
            if headers:
                final_headers = headers

    if not all_data:
        return None, None

    merged = np.vstack(all_data)
    # Deduplicate time stamps
    _, unique_idx = np.unique(merged[:, 0], return_index=True)
    merged = merged[unique_idx]
    merged = merged[np.argsort(merged[:, 0])]

    return final_headers, merged

def plot_forces(time_data, out_path, settled_time=100):
    """
    Plots dimensional forces (Total, Pressure, Viscous) for X, Y, Z components over time.
    Uses a 2-column layout:
      - Left column: Full run history (including startup transient)
      - Right column: Settled regime (t >= settled_time) with clear force resolution.
    """
    time = time_data[:, 0]
    mask_settled = time >= settled_time
    if not np.any(mask_settled):
        mask_settled = np.ones(len(time), dtype=bool)

    t_all = time
    t_set = time[mask_settled]

    total_x_all, total_y_all, total_z_all = time_data[:, 1], time_data[:, 2], time_data[:, 3]
    press_x_all, press_y_all, press_z_all = time_data[:, 4], time_data[:, 5], time_data[:, 6]
    visc_x_all,  visc_y_all,  visc_z_all  = time_data[:, 7], time_data[:, 8], time_data[:, 9]

    total_x_set, total_y_set, total_z_set = time_data[mask_settled, 1], time_data[mask_settled, 2], time_data[mask_settled, 3]
    press_x_set, press_y_set, press_z_set = time_data[mask_settled, 4], time_data[mask_settled, 5], time_data[mask_settled, 6]
    visc_x_set,  visc_y_set,  visc_z_set  = time_data[mask_settled, 7], time_data[mask_settled, 8], time_data[mask_settled, 9]

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex='col')
    fig.patch.set_facecolor('#ffffff')

    c_tot = '#1f77b4'
    c_pres = '#ff7f0e'
    c_visc = '#2ca02c'

    # --- ROW 1: Fx ---
    # Left: All iterations
    axes[0, 0].plot(t_all, total_x_all, label=r'Total $F_x$', color=c_tot, linewidth=1.4)
    axes[0, 0].plot(t_all, press_x_all, label=r'Pressure $F_{p,x}$', color=c_pres, linestyle='--', linewidth=1.1)
    axes[0, 0].plot(t_all, visc_x_all, label=r'Viscous $F_{v,x}$', color=c_visc, linestyle=':', linewidth=1.2)
    axes[0, 0].set_ylabel(r'Force $F_x$ [N]', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)
    axes[0, 0].set_title('Full History (Iterations 1 to End)', fontsize=12, fontweight='bold')

    # Right: Settled regime
    axes[0, 1].plot(t_set, total_x_set, label=r'Total $F_x$', color=c_tot, linewidth=1.5)
    axes[0, 1].plot(t_set, press_x_set, label=r'Pressure $F_{p,x}$', color=c_pres, linestyle='--', linewidth=1.2)
    axes[0, 1].plot(t_set, visc_x_set, label=r'Viscous $F_{v,x}$', color=c_visc, linestyle=':', linewidth=1.4)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)
    axes[0, 1].set_title(f'Settled Regime (Iterations $\\geq$ {int(settled_time)})', fontsize=12, fontweight='bold')

    # --- ROW 2: Fy ---
    # Left: All iterations
    axes[1, 0].plot(t_all, total_y_all, label=r'Total $F_y$', color=c_tot, linewidth=1.4)
    axes[1, 0].plot(t_all, press_y_all, label=r'Pressure $F_{p,y}$', color=c_pres, linestyle='--', linewidth=1.1)
    axes[1, 0].plot(t_all, visc_y_all, label=r'Viscous $F_{v,y}$', color=c_visc, linestyle=':', linewidth=1.2)
    axes[1, 0].set_ylabel(r'Force $F_y$ [N]', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Right: Settled regime
    axes[1, 1].plot(t_set, total_y_set, label=r'Total $F_y$', color=c_tot, linewidth=1.5)
    axes[1, 1].plot(t_set, press_y_set, label=r'Pressure $F_{p,y}$', color=c_pres, linestyle='--', linewidth=1.2)
    axes[1, 1].plot(t_set, visc_y_set, label=r'Viscous $F_{v,y}$', color=c_visc, linestyle=':', linewidth=1.4)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)

    # --- ROW 3: Fz ---
    # Left: All iterations
    axes[2, 0].plot(t_all, total_z_all, label=r'Total $F_z$', color=c_tot, linewidth=1.4)
    axes[2, 0].plot(t_all, press_z_all, label=r'Pressure $F_{p,z}$', color=c_pres, linestyle='--', linewidth=1.1)
    axes[2, 0].plot(t_all, visc_z_all, label=r'Viscous $F_{v,z}$', color=c_visc, linestyle=':', linewidth=1.2)
    axes[2, 0].set_xlabel('Iteration / Time', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel(r'Force $F_z$ [N]', fontsize=11, fontweight='bold')
    axes[2, 0].grid(True, linestyle='--', alpha=0.6)
    axes[2, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Right: Settled regime
    axes[2, 1].plot(t_set, total_z_set, label=r'Total $F_z$', color=c_tot, linewidth=1.5)
    axes[2, 1].plot(t_set, press_z_set, label=r'Pressure $F_{p,z}$', color=c_pres, linestyle='--', linewidth=1.2)
    axes[2, 1].plot(t_set, visc_z_set, label=r'Viscous $F_{v,z}$', color=c_visc, linestyle=':', linewidth=1.4)
    axes[2, 1].set_xlabel('Iteration / Time', fontsize=11, fontweight='bold')
    axes[2, 1].grid(True, linestyle='--', alpha=0.6)
    axes[2, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)

    fig.suptitle('OpenFOAM Dimensional Forces ($F_x, F_y, F_z$) Over Time', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved dimensional forces plot to: {out_path}")

def plot_force_coefficients(coeff_data, out_path, settled_time=100):
    """
    Plots aerodynamic force coefficients (Cd, Cl, Cs, CmPitch) over time.
    Uses a 2-column layout:
      - Left column: Full run history
      - Right column: Settled regime (t >= settled_time) with high-resolution resolution.
    """
    time = coeff_data[:, 0]
    mask_settled = time >= settled_time
    if not np.any(mask_settled):
        mask_settled = np.ones(len(time), dtype=bool)

    t_all = time
    t_set = time[mask_settled]

    Cd_all  = coeff_data[:, 1]
    Cdf_all = coeff_data[:, 2] if coeff_data.shape[1] > 2 else None
    Cdr_all = coeff_data[:, 3] if coeff_data.shape[1] > 3 else None
    Cl_all  = coeff_data[:, 4] if coeff_data.shape[1] > 4 else None
    Clf_all = coeff_data[:, 5] if coeff_data.shape[1] > 5 else None
    Clr_all = coeff_data[:, 6] if coeff_data.shape[1] > 6 else None
    CmP_all = coeff_data[:, 7] if coeff_data.shape[1] > 7 else None
    Cs_all  = coeff_data[:, 10] if coeff_data.shape[1] > 10 else None

    Cd_set  = coeff_data[mask_settled, 1]
    Cdf_set = coeff_data[mask_settled, 2] if coeff_data.shape[1] > 2 else None
    Cdr_set = coeff_data[mask_settled, 3] if coeff_data.shape[1] > 3 else None
    Cl_set  = coeff_data[mask_settled, 4] if coeff_data.shape[1] > 4 else None
    Clf_set = coeff_data[mask_settled, 5] if coeff_data.shape[1] > 5 else None
    Clr_set = coeff_data[mask_settled, 6] if coeff_data.shape[1] > 6 else None
    CmP_set = coeff_data[mask_settled, 7] if coeff_data.shape[1] > 7 else None
    Cs_set  = coeff_data[mask_settled, 10] if coeff_data.shape[1] > 10 else None

    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex='col')
    fig.patch.set_facecolor('#ffffff')

    # --- ROW 1: Cd ---
    # Left: All
    axes[0, 0].plot(t_all, Cd_all, label=r'Total $C_d$', color='#d62728', linewidth=1.5)
    if Cdf_all is not None:
        axes[0, 0].plot(t_all, Cdf_all, label=r'$C_d$(front)', color='#e377c2', linestyle='--', linewidth=1.1)
    if Cdr_all is not None:
        axes[0, 0].plot(t_all, Cdr_all, label=r'$C_d$(rear)', color='#8c564b', linestyle=':', linewidth=1.1)
    axes[0, 0].set_ylabel(r'Drag Coeff $C_d$', fontsize=11, fontweight='bold')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)
    axes[0, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)
    axes[0, 0].set_title('Full History (Iterations 1 to End)', fontsize=12, fontweight='bold')

    # Right: Settled
    axes[0, 1].plot(t_set, Cd_set, label=r'Total $C_d$', color='#d62728', linewidth=1.6)
    if Cdf_set is not None:
        axes[0, 1].plot(t_set, Cdf_set, label=r'$C_d$(front)', color='#e377c2', linestyle='--', linewidth=1.2)
    if Cdr_set is not None:
        axes[0, 1].plot(t_set, Cdr_set, label=r'$C_d$(rear)', color='#8c564b', linestyle=':', linewidth=1.2)
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)
    axes[0, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)
    axes[0, 1].set_title(f'Settled Regime (Iterations $\\geq$ {int(settled_time)})', fontsize=12, fontweight='bold')

    # --- ROW 2: Cl ---
    # Left: All
    if Cl_all is not None:
        axes[1, 0].plot(t_all, Cl_all, label=r'Total $C_l$', color='#1f77b4', linewidth=1.5)
    if Clf_all is not None:
        axes[1, 0].plot(t_all, Clf_all, label=r'$C_l$(front)', color='#17becf', linestyle='--', linewidth=1.1)
    if Clr_all is not None:
        axes[1, 0].plot(t_all, Clr_all, label=r'$C_l$(rear)', color='#9467bd', linestyle=':', linewidth=1.1)
    axes[1, 0].set_ylabel(r'Lift Coeff $C_l$', fontsize=11, fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)
    axes[1, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Right: Settled
    if Cl_set is not None:
        axes[1, 1].plot(t_set, Cl_set, label=r'Total $C_l$', color='#1f77b4', linewidth=1.6)
    if Clf_set is not None:
        axes[1, 1].plot(t_set, Clf_set, label=r'$C_l$(front)', color='#17becf', linestyle='--', linewidth=1.2)
    if Clr_set is not None:
        axes[1, 1].plot(t_set, Clr_set, label=r'$C_l$(rear)', color='#9467bd', linestyle=':', linewidth=1.2)
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)
    axes[1, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)

    # --- ROW 3: CmPitch & Cs ---
    # Left: All
    if CmP_all is not None:
        axes[2, 0].plot(t_all, CmP_all, label=r'Pitch Moment $C_{m,\mathrm{pitch}}$', color='#ff7f0e', linewidth=1.4)
    if Cs_all is not None:
        axes[2, 0].plot(t_all, Cs_all, label=r'Side Coeff $C_s$', color='#2ca02c', linestyle='--', linewidth=1.2)
    axes[2, 0].set_xlabel('Iteration / Time', fontsize=11, fontweight='bold')
    axes[2, 0].set_ylabel(r'Moment / Side Coeff', fontsize=11, fontweight='bold')
    axes[2, 0].grid(True, linestyle='--', alpha=0.6)
    axes[2, 0].legend(loc='upper right', framealpha=0.9, fontsize=9)

    # Right: Settled
    if CmP_set is not None:
        axes[2, 1].plot(t_set, CmP_set, label=r'Pitch Moment $C_{m,\mathrm{pitch}}$', color='#ff7f0e', linewidth=1.5)
    if Cs_set is not None:
        axes[2, 1].plot(t_set, Cs_set, label=r'Side Coeff $C_s$', color='#2ca02c', linestyle='--', linewidth=1.3)
    axes[2, 1].set_xlabel('Iteration / Time', fontsize=11, fontweight='bold')
    axes[2, 1].grid(True, linestyle='--', alpha=0.6)
    axes[2, 1].legend(loc='upper right', framealpha=0.9, fontsize=9)

    fig.suptitle('Aerodynamic Force & Moment Coefficients Over Time', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved aerodynamic coefficients plot to: {out_path}")

def plot_convergence_zoom(coeff_data, out_path, last_n=300):
    """
    Plots asymptotic convergence zoomed into the last N iterations.
    """
    n_pts = len(coeff_data)
    if n_pts < 10:
        return

    n_view = min(n_pts, last_n)
    subset = coeff_data[-n_view:]
    t = subset[:, 0]
    Cd = subset[:, 1]
    Cl = subset[:, 4] if subset.shape[1] > 4 else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.patch.set_facecolor('#ffffff')

    ax1.plot(t, Cd, color='#d62728', marker='.', markersize=3.5, linewidth=1.3)
    ax1.set_ylabel(r'Drag $C_d$', fontsize=12, fontweight='bold')
    ax1.set_title(f'Asymptotic Convergence Stability (Last {n_view} Iterations)', fontsize=14, fontweight='bold', pad=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Annotate mean and std
    cd_mean, cd_std = np.mean(Cd), np.std(Cd)
    ax1.axhline(cd_mean, color='black', linestyle=':', alpha=0.7, label=f'Mean = {cd_mean:.6f} ($\\pm${cd_std:.1e})')
    ax1.legend(loc='upper right', framealpha=0.9)

    if Cl is not None:
        ax2.plot(t, Cl, color='#1f77b4', marker='.', markersize=3.5, linewidth=1.3)
        ax2.set_ylabel(r'Lift $C_l$', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.6)
        cl_mean, cl_std = np.mean(Cl), np.std(Cl)
        ax2.axhline(cl_mean, color='black', linestyle=':', alpha=0.7, label=f'Mean = {cl_mean:.6f} ($\\pm${cl_std:.1e})')
        ax2.legend(loc='upper right', framealpha=0.9)

    ax2.set_xlabel('Iteration / Time', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[✓] Saved convergence zoom plot to: {out_path}")

def print_summary(forces_data, coeff_data, last_n=100):
    """
    Prints terminal summary of final converged values and statistics over the last N iterations.
    """
    print("\n" + "="*70)
    print("                 LATEST FORCES & COEFFICIENTS SUMMARY                  ")
    print("="*70)

    if coeff_data is not None and len(coeff_data) > 0:
        last_row = coeff_data[-1]
        t_final = last_row[0]
        cd_final = last_row[1]
        cl_final = last_row[4] if len(last_row) > 4 else float('nan')
        cmpitch_final = last_row[7] if len(last_row) > 7 else float('nan')
        cs_final = last_row[10] if len(last_row) > 10 else float('nan')

        n_pts = min(len(coeff_data), last_n)
        cd_avg = np.mean(coeff_data[-n_pts:, 1])
        cl_avg = np.mean(coeff_data[-n_pts:, 4]) if coeff_data.shape[1] > 4 else float('nan')

        print(f"Final Iteration / Time       : {t_final:.1f}")
        print(f"Drag Coeff (Cd) [Last Value] : {cd_final:12.6f}  (Last {n_pts} Avg: {cd_avg:.6f})")
        print(f"Lift Coeff (Cl) [Last Value] : {cl_final:12.6f}  (Last {n_pts} Avg: {cl_avg:.6f})")
        print(f"Side Coeff (Cs) [Last Value] : {cs_final:12.6f}")
        print(f"Pitch Moment    [Last Value] : {cmpitch_final:12.6f}")

    if forces_data is not None and len(forces_data) > 0:
        last_f = forces_data[-1]
        n_pts = min(len(forces_data), last_n)
        fx_avg = np.mean(forces_data[-n_pts:, 1])
        fy_avg = np.mean(forces_data[-n_pts:, 2])

        print("-" * 70)
        print(f"Total Force Fx (Drag)        : {last_f[1]:12.2f} N (Last {n_pts} Avg: {fx_avg:.2f} N)")
        print(f"Total Force Fy (Lift)        : {last_f[2]:12.2f} N (Last {n_pts} Avg: {fy_avg:.2f} N)")
        print(f"Total Force Fz (Side)        : {last_f[3]:12.2f} N")
        print(f"Pressure Force (Fx, Fy)      : ({last_f[4]:.2f}, {last_f[5]:.2f}) N")
        print(f"Viscous Force  (Fx, Fy)      : ({last_f[7]:.2f}, {last_f[8]:.2f}) N")
    print("="*70 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Plot OpenFOAM forces and force coefficients over time.")
    parser.add_argument("--case", type=str, default=None,
                        help="Path to OpenFOAM case directory (default: script directory)")
    parser.add_argument("--settled-time", type=float, default=100.0,
                        help="Starting iteration for settled regime panel (default: 100)")
    parser.add_argument("--zoom-last", type=int, default=300,
                        help="Number of final iterations for convergence zoom plot (default: 300)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save generated plots (default: case directory)")

    args = parser.parse_args()

    # Determine case directory
    if args.case:
        case_dir = os.path.abspath(args.case)
    else:
        case_dir = os.path.dirname(os.path.abspath(__file__))

    out_dir = os.path.abspath(args.output_dir) if args.output_dir else case_dir
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Analyzing OpenFOAM case: {case_dir}")

    # Load forces and force coefficients
    _, forces_data = load_postprocessing_data(case_dir, "forces", "force.dat")
    _, coeff_data  = load_postprocessing_data(case_dir, "forceCoeffs", "coefficient.dat")

    if forces_data is None and coeff_data is None:
        print("[!] Error: No force.dat or coefficient.dat files found in postProcessing/!")
        sys.exit(1)

    # Generate plots
    if forces_data is not None:
        out_forces = os.path.join(out_dir, "forces_over_time.png")
        plot_forces(forces_data, out_forces, settled_time=args.settled_time)

    if coeff_data is not None:
        out_coeffs = os.path.join(out_dir, "force_coefficients_over_time.png")
        plot_force_coefficients(coeff_data, out_coeffs, settled_time=args.settled_time)

        out_zoom = os.path.join(out_dir, "convergence_zoom.png")
        plot_convergence_zoom(coeff_data, out_zoom, last_n=args.zoom_last)

    # Print summary
    print_summary(forces_data, coeff_data)

if __name__ == "__main__":
    main()
