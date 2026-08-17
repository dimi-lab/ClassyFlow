import pandas as pd
import sys
import re
import os
import json

# --- Column cleaning config ---
# remove_columns = ['Batch', 'Histology', 'DAPI_R01', 'DAPI_AF_R01', 'DAPI_R08','DAPI_R25', 'DAPI_R27', 'DAPI_AF_R13']
remove_columns = ['Distance to annotation with Macrophage µm', 'Distance to annotation with Hepatocyte µm', 
    'Distance to annotation with Cholangiocytes µm', 'Distance to annotation with CD4 T Cell µm',
    'Distance to annotation with B Cell µm', 'Distance to annotation with Hepatic Stellate Cell µm', 
    'Distance to annotation with LSEC Endothelial µm' ,'Distance to annotation with Other µm', 
    'Distance to annotation with CD8 T Cell µm',
    'Distance to detection B Cell µm', 'Distance to detection CD4 T Cell µm', 'Distance to detection CD8 T Cell µm', 
    'Distance to detection Cholangiocytes µm', 'Distance to detection Hepatic Stellate Cell µm', 
    'Distance to detection Hepatocyte µm', 'Distance to detection LSEC Endothelial µm', 
    'Distance to detection Macrophage µm', 'Distance to detection Other µm']

rename_columns = {
    'OldName1': 'NewName1'
    # Add more renaming rules as needed
}
marker_map = {'ASMA': 'aSMA',
            'CD11B': 'CD11b',  'CD11C': 'CD11c',
            'TenC':'TENASCINC', 'Gal9':'GALECTIN9',
            'HLAI':'HLA_I' ,'HLAII':'HLA_II',
            'NAK':'NAKATPASE',
            'FOXP3':'FoxP3', 
            'PANCK':'PanCK', 
            'GP100':'gp100',
            'B2MG' : 'B2M',
            'GZB' : 'GzB',
            'tryptase' : 'TRYPTASE'
}  # Add more as needed

# --- Classification cleaning config ---
upgradeMacs = True  # Set to False to downgrade M1/M2 to Macrophage

lookup = {
    "Ignore*": "",
    "Melanocytes": "Tumor",
    "Tumor Cell": "Tumor",
    "NRP2+TumC": "Tumor",
    "NRP+TumC": "Tumor",
    "Bcell": "B Cell",
    "CLT": "CytoT",
    "CTL": "CytoT",
    "Cytotoxic T Cell": "CytoT",
    "Helper T": "HelperT",
    "Thelper": "HelperT",
    "Helper T Cell": "HelperT",
    "Helper T-cell": "HelperT",
    "Helper T-cell": "HelperT",
    "M1": "M1 Macrophage",
    "Macrophage": "M1 Macrophage",
    "M1 Macrophages": "M1 Macrophage",
    "CD68+": "M1 Macrophage",
    "M2": "M2 Macrophage",
    "MPO": "Neutro",
    "Neuto": "Neutro",
    "Neutro": "Neutro",
    "Neutrophil": "Neutro",
    "Epithelial Cell": "Epithelial",
    "Stroma": "Epithelial",
    "SM": "Epithelial",
    "Treg": "T Reg",
    "Regulatory T Cell": "T Reg",
    "TumC": "Tumor", 
    "TumGas": "Tumor",
    #"Vascul": "Vasculature",
    "Vascul": "Endothelial",
    "Vasculature": "Endothelial",
    "Endo": "Endothelial",
    "Dendritic Cell": "DC",
    "CD11c@": "",
    "CD11c+": "DC",
    "Monocyte":"Non-classical monocytes",  
    "Int-M":"Non-classical monocytes",  
    "NCM":"Non-classical monocytes",  
    "CM":"Classical Monocytes",  
    "TAM" : "M2 Macrophage", # BE WARE - PROJECT SPECIFIC Recodification
    "MDSC" : "M2 Macrophage", # BE WARE - PROJECT SPECIFIC Recodification
    "NK" : "NK Cell",
    "CD56" : "NK Cell",
    "<TBD>": "",
    "B2M": "",
    "NRP2+": "",
    "NRP+": "",
    "NRP": "",
    "GZB": "",
    "CD28": "",
    "FAP": "",
    "IDO": "",
    "NKG2D": "",
    "NKGD2": "",
    "Gal3": "",
    "Gal9": "",
    "GzB": "",
    "HER2": "",
    "HLA1": "",
    "HLAI": "",
    "HLA2": "",
    "HLAII": "",
    "ILT4": "",
    "LAG3": "",
    "LAG3+": "",    
    "FAP=": "",
    "PDL1": "",
    "PDL1": "",
    "Survivin": "",
    "TIGIT": "",
    "TIM3": "",
    "CTLA4": "",
    "CTLA4+": "",
    "EGFR": "",
    "CD013A": "",
    "CD103A": "",
    "CD130A": "",
    "NLG2D": "",
    "HLAII": "",
    "HLAI": "",
    "CD141": "",
    "CD103a": "",
    "SURVIVIN+": "",
    "NKG2D+": "",
    "CD28+": "",
    "PDL1+": "",
    "EGFR+": "",
    "B2M+": "",
    "Tim3": "",
    "Her2+": "",
    "Gal9+": "",
    "Gal3+": "",
    "ILT4+": "",
    "aSMA@": ""
}

# --- Helper functions ---
def rename_marker(df, marker_map):
    # marker_map: {old_marker: new_marker}
    newcols = []
    for col in df.columns:
        replaced = False
        for old, new in marker_map.items():
            if col.startswith(old + ":"):
                newcols.append(new + col[len(old):])
                replaced = True
                break
        if not replaced:
            newcols.append(col)
    df.columns = newcols
    return df

def remove_dash_suffix(val):
    """Remove any values that end with '-' (even after '|'), converting them to blank."""
    if pd.isna(val):
        return val
    parts = [v.strip() for v in str(val).split('|')]
    cleaned = [p for p in parts if not re.search(r'-$', p)]
    return '|'.join(cleaned) if cleaned else ""

def upgrade_macrophages(df):
    # Find columns containing both 'CD68' & 'Median', and 'CD206' & 'Median'
    cd68_col = next((col for col in df.columns if 'CD68' in col and 'Median' in col), None)
    cd206_col = next((col for col in df.columns if 'CD206' in col and 'Median' in col), None)
    if cd68_col and cd206_col:
        def upgrade_row(row):
            if re.search(r'Macrophage', str(row['Classification'])):
                try:
                    cd68 = float(row[cd68_col])
                    cd206 = float(row[cd206_col])
                    if cd206 > cd68:
                        return 'M2 Macrophage'
                    else:
                        return 'M1 Macrophage'
                except Exception:
                    return row['Classification']
            return row['Classification']
        df['Classification'] = df.apply(upgrade_row, axis=1)
    return df

def main(input_path, log_path):
    # Read the TSV file
    df = pd.read_csv(input_path, sep='\t', low_memory=False)

    # --- Column cleaning ---
    for col in remove_columns:
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.rename(columns={k: v for k, v in rename_columns.items() if k in df.columns})
    df = rename_marker(df, marker_map)

    # --- Classification cleaning ---
    if "Classification" in df.columns:
        # Remove values ending with '-'
        df["Classification"] = df["Classification"].apply(remove_dash_suffix)

        # Remap using lookup
        not_found = set()
        def remap(val):
            if pd.isna(val):
                return val
            parts = [v.strip() for v in str(val).split('|')]
            new_parts = []
            for p in parts:
                if p in lookup:
                    if lookup[p]:
                        new_parts.append(lookup[p])
                else:
                    not_found.add(p)
                    new_parts.append(p)
            return '|'.join(new_parts)
        df["Classification"] = df["Classification"].apply(remap)

        if upgradeMacs:
            df = upgrade_macrophages(df)

        print("\nUnique counts after remapping:")
        print(df["Classification"].value_counts())


    # Save back to the same file
    df.to_csv(input_path, sep='\t', index=False)

    # --- Logging value counts to JSON ---
    base = os.path.splitext(os.path.basename(input_path))[0]
    counts = df["Classification"].value_counts().to_dict()
    # Read existing log file if present
    try:
        with open(log_path, "r") as f:
            log_data = json.load(f)
    except Exception:
        log_data = {}
    # Update log with new counts for this file
    log_data[base] = counts
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python fixup_columns.py <input.tsv> <log.json>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
