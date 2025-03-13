#!/usr/bin/env python
import os
import sys
import pandas as pd
import json
import re

# --- Helper functions for safe conversion ---
def safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(float(x))
    except (ValueError, TypeError):
        return default

def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except (ValueError, TypeError):
        return default

# --- Mapping for resource codes to friendly names ---
resource_map = {
    "progdir": "Program Director",
    "progmgr": "Program Manager",
    "cpm": "CPM",
    "solarch": "Solution Architect",
    "integengin": "Integration Engineer",
    "swdev": "Software Developer",
    "itengtest": "IT Engineering Test",
    "testmgr": "Test Manager",
    "itsysexp": "IT System Expert",
    "techsme": "Technical SME",
    "consult": "Consultant",
    # You can add more mappings if needed.
}

# --- Configuration: update the filename/path as needed ---
filename = r".\Test_Simon\smal_excell_jason.csv"

# --- Check if file exists ---
if not os.path.exists(filename):
    print(f"Error: File '{filename}' not found. Please check the file path.")
    sys.exit(1)

# --- Read the file ---
# We now set header=5 so that row 6 in the raw CSV is used as header.
try:
    df = pd.read_csv(filename, encoding="cp1252", header=5)
except Exception as e:
    print(f"Error reading file '{filename}': {e}")
    sys.exit(1)

# --- Define a function to check if a dependency cell is valid ---
def valid_dependency(dep):
    dep_str = str(dep).strip()
    if dep_str.upper() == "NA":
        return True
    return bool(re.fullmatch(r"\d+(,\s*\d+)*", dep_str))

# --- Filter rows based on valid dependency values in column C (3rd column) ---
df_valid = df[df.iloc[:, 2].apply(valid_dependency)].copy()

# --- Build mapping from original row index to new id and extract basic task info ---
mapping = {}
tasks = []
new_id = 1

for orig_idx, row in df_valid.iterrows():
    mapping[orig_idx] = new_id
    task = {
        "original_index": orig_idx,
        "task_name": row.iloc[0],
        "max": safe_int(row.iloc[3]),
        "min": safe_int(row.iloc[4]),
        "base_effort": safe_float(row.iloc[5]),
        "raw_dependencies": str(row.iloc[2]).strip(),  # Column C: dependencies
        "resource": ""
    }
    tasks.append(task)
    new_id += 1

# --- Update dependencies using the mapping ---
for task in tasks:
    raw_dep = task["raw_dependencies"]
    if raw_dep.upper() == "NA":
        task["dependencies"] = []
    else:
        try:
            deps = [safe_int(x.strip(), default=None) for x in raw_dep.split(",") if x.strip()]
            deps = [d for d in deps if d is not None]
        except Exception as e:
            deps = []
        new_deps = [mapping[dep] for dep in deps if dep in mapping]
        task["dependencies"] = new_deps
    del task["raw_dependencies"]

# --- Process resource allocations from resource columns ---
# Here we assume that resource columns start at column J (index 9)
# and that the header (from row 6 of the raw CSV) holds the resource type.
resource_columns = df.columns[9:]
for idx, (orig_idx, row) in enumerate(df_valid.iterrows()):
    resources = []
    for col in resource_columns:
        cell = row[col]
        if pd.notnull(cell):
            token = str(cell).strip()
            # Skip empty or zero values
            if token in ["", "0", "0.0", "0.0%"]:
                continue

            # If there is a comma, split into tokens (e.g. "0.0%, 40.0%, 60.0%, 100.0%")
            token_list = [t.strip() for t in token.split(",")] if "," in token else [token]
            for t in token_list:
                if t.endswith("%"):
                    try:
                        percent_val = float(t.replace("%", ""))
                    except Exception:
                        percent_val = 0.0
                    if percent_val > 0:
                        # Use the resource type from the header of this column.
                        col_header = str(col).strip()
                        col_header_lower = col_header.lower()
                        mapped_resource = resource_map.get(col_header_lower, col_header)
                        resources.append(mapped_resource)
                else:
                    t_lower = t.lower()
                    mapped_resource = resource_map.get(t_lower, t)
                    resources.append(mapped_resource)
    # Deduplicate while preserving order:
    resources = list(dict.fromkeys(resources))
    final_resource = ", ".join(resources)
    tasks[idx]["resource"] = final_resource

# --- Build the final list of tasks, filtering out those without any allocated resource ---
final_tasks = []
for task in tasks:
    if not task["resource"]:
        continue  # Skip tasks with no resource allocated
    final_tasks.append({
        "id": mapping[task["original_index"]],
        "task_name": task["task_name"],
        "base_effort": task["base_effort"],
        "min": task["min"],
        "max": task["max"],
        "dependencies": task["dependencies"],
        "resource": task["resource"]
    })

# --- Output the JSON to the console for quick review ---
json_output = json.dumps(final_tasks, indent=4)
print(json_output)

# --- Output the final tasks to a new CSV file ---
output_df = pd.DataFrame(final_tasks)
output_filename = "final_tasks.csv"
try:
    output_df.to_csv(output_filename, index=False)
    print(f"Output saved to {output_filename}")
except Exception as e:
    print(f"Error writing to CSV file '{output_filename}': {e}")
