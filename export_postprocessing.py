#!/usr/bin/env python3
import os
import glob
import re
import csv

base_dir = "/home/lavender/OpenFoamUbu/offProjects/solarPanel"
output_csv = os.path.join(base_dir, "summary_forces.csv")

def parse_coefficient_file(filepath):
    """Parses OpenFOAM forceCoeffs coefficient.dat file for latest iteration values."""
    if not os.path.exists(filepath):
        return {}
    
    headers = []
    last_line = None
    
    with open(filepath, 'r') as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str:
                continue
            if line_str.startswith('#'):
                # Check for header line
                if 'Time' in line_str and ('Cd' in line_str or 'Cl' in line_str):
                    headers = [h.strip() for h in re.split(r'\s+', line_str.lstrip('#').strip())]
            else:
                last_line = line_str
                
    if not last_line or not headers:
        return {}
        
    values = [v.strip() for v in re.split(r'\s+', last_line)]
    if len(values) != len(headers):
        # Try matching as floats
        try:
            floats = [float(v) for v in values]
            if len(floats) >= len(headers):
                return {headers[i]: floats[i] for i in range(len(headers))}
        except ValueError:
            pass
        return {}
        
    res = {}
    for h, v in zip(headers, values):
        try:
            res[h] = float(v)
        except ValueError:
            res[h] = v
    return res

def parse_force_file(filepath):
    """Parses OpenFOAM forces force.dat file for latest iteration values (forces in N)."""
    if not os.path.exists(filepath):
        return {}
        
    last_line = None
    with open(filepath, 'r') as fp:
        for line in fp:
            line_str = line.strip()
            if not line_str or line_str.startswith('#'):
                continue
            last_line = line_str
            
    if not last_line:
        return {}
        
    # Standard OpenFOAM forces format: Time total(x y z) pressure(x y z) viscous(x y z)
    # Often formatted as: Time fx fy fz px py pz vx vy vz
    parts = re.split(r'\s+', last_line)
    # Clean parentheses if present
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

def find_latest_postprocessing_file(case_dir, sub_type):
    """Finds the latest time-folder file under postProcessing/sub_type/ (e.g. forceCoeffs or forces)."""
    post_dir = os.path.join(case_dir, "postProcessing", sub_type)
    if not os.path.exists(post_dir):
        return None
        
    # Search for numeric time directories under postProcessing/sub_type/
    time_dirs = []
    for d in os.listdir(post_dir):
        dp = os.path.join(post_dir, d)
        if os.path.isdir(dp):
            try:
                tval = float(d)
                time_dirs.append((tval, dp))
            except ValueError:
                pass
                
    if not time_dirs:
        return None
        
    time_dirs.sort(key=lambda x: x[0], reverse=True)
    latest_time_dir = time_dirs[0][1]
    
    # Check for dat file in latest time dir
    for target in ["coefficient.dat", "coefficient_0.dat", "force.dat", "forces.dat", "forceCoeffs.dat"]:
        candidate = os.path.join(latest_time_dir, target)
        if os.path.exists(candidate):
            return candidate
            
    # Fallback to any .dat file
    dats = glob.glob(os.path.join(latest_time_dir, "*.dat"))
    if dats:
        return dats[0]
        
    return None

def collect_all_results():
    results = []
    
    # Search for all directories containing a postProcessing folder
    for root, dirs, files in os.walk(base_dir):
        if "postProcessing" in dirs:
            case_dir = root
            rel_case = os.path.relpath(case_dir, base_dir)
            
            coeff_file = find_latest_postprocessing_file(case_dir, "forceCoeffs")
            force_file = find_latest_postprocessing_file(case_dir, "forces")
            
            if not coeff_file and not force_file:
                continue
                
            coeff_data = parse_coefficient_file(coeff_file) if coeff_file else {}
            force_data = parse_force_file(force_file) if force_file else {}
            
            row = {
                "Case": rel_case,
                "Time": coeff_data.get("Time", force_data.get("Time", "")),
                "Cd": coeff_data.get("Cd", ""),
                "Cl": coeff_data.get("Cl", ""),
                "Cs": coeff_data.get("Cs", ""),
                "CmPitch": coeff_data.get("CmPitch", ""),
                "CmRoll": coeff_data.get("CmRoll", ""),
                "CmYaw": coeff_data.get("CmYaw", ""),
                "Drag_Force_N": force_data.get("Total_Fx", ""),
                "Lift_Force_N": force_data.get("Total_Fy", ""),
                "Side_Force_N": force_data.get("Total_Fz", ""),
                "Pressure_Fx_N": force_data.get("Pressure_Fx", ""),
                "Pressure_Fy_N": force_data.get("Pressure_Fy", ""),
                "Pressure_Fz_N": force_data.get("Pressure_Fz", ""),
                "Viscous_Fx_N": force_data.get("Viscous_Fx", ""),
                "Viscous_Fy_N": force_data.get("Viscous_Fy", ""),
                "Viscous_Fz_N": force_data.get("Viscous_Fz", "")
            }
            results.append(row)
            
    # Sort results by case path
    results.sort(key=lambda x: x["Case"])
    return results

def main():
    results = collect_all_results()
    if not results:
        print("No postProcessing force/coefficient data found.")
        return

    fieldnames = [
        "Case", "Time", "Cd", "Cl", "Cs", "CmPitch", "CmRoll", "CmYaw",
        "Drag_Force_N", "Lift_Force_N", "Side_Force_N",
        "Pressure_Fx_N", "Pressure_Fy_N", "Pressure_Fz_N",
        "Viscous_Fx_N", "Viscous_Fy_N", "Viscous_Fz_N"
    ]
    
    with open(output_csv, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Also save a copy in meshConv if meshConv directory exists
    meshconv_csv = os.path.join(base_dir, "meshConv", "summary_forces.csv")
    if os.path.exists(os.path.dirname(meshconv_csv)):
        with open(meshconv_csv, 'w', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Successfully exported {len(results)} case results to CSV:")
    print(f"  - {output_csv}")
    if os.path.exists(os.path.dirname(meshconv_csv)):
        print(f"  - {meshconv_csv}")
        
    print("\nSummary Table Preview:")
    print(f"{'Case':<40} {'Time':<8} {'Cd':<10} {'Cl':<10} {'Drag(N)':<12} {'Lift(N)':<12}")
    print("-" * 92)
    for r in results[:20]:
        cd_str = f"{r['Cd']:.4f}" if isinstance(r['Cd'], float) else str(r['Cd'])
        cl_str = f"{r['Cl']:.4f}" if isinstance(r['Cl'], float) else str(r['Cl'])
        drag_str = f"{r['Drag_Force_N']:.2f}" if isinstance(r['Drag_Force_N'], float) else str(r['Drag_Force_N'])
        lift_str = f"{r['Lift_Force_N']:.2f}" if isinstance(r['Lift_Force_N'], float) else str(r['Lift_Force_N'])
        print(f"{r['Case']:<40} {str(r['Time']):<8} {cd_str:<10} {cl_str:<10} {drag_str:<12} {lift_str:<12}")

if __name__ == "__main__":
    main()
