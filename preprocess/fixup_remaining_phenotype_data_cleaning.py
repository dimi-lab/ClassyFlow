import pandas as pd
import sys
import re

def remove_dash_suffix(val):
    """Remove any values that end with '-' (even after '|'), converting them to blank."""
    if pd.isna(val):
        return val
    parts = [v.strip() for v in str(val).split('|')]
    cleaned = [p for p in parts if not re.search(r'-$', p)]
    return '|'.join(cleaned) if cleaned else ""

lookup = {
    "Ignore*": "",
    "Melanocytes": "Tumor",
    "B cell": "B Cell",
    "Bcell": "B Cell",
    "CLT": "CytoT",
    "CTL": "CytoT",
    "Helper T": "HelperT",
    "M1": "M1 Macrophage",
    "M2": "M2 Macrophage",
    "MPO": "Neutro",
    "Neuto": "Neutro",
    "Neutro": "Neutro",
    "Stroma": "Epithelial",
    "Treg": "T Reg",
    "TumC": "Tumor", 
    "TumGas": "Tumor",
    "Vascul": "Endothelial",
    "Endo": "Endothelial",
    "<TBD>": "",
    "B2M": "",
    "GZB": "",
    "CD28": "",
    "ExstdT": "",
    "FAP": "",
    "IDO": "",
    "NKG2D": "",
    "NKGD2": "",
    "Gal3": "",
    "Gal9": "",
    "GzB": "",
    "HER2": "",
    "HLA1": "",
    "HLA2": "",
    "ILT4": "",
    "LAG3": "",
    "LAG3+": "",    
    "FAP=": "",
    "PD1": "",
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
    "PD1+": "",
    "EGFR+": "",
    "B2M+": "",
    "Tim3": "",
    "Her2+": "",
    "Gal9+": "",
    "Gal3+": "",
    "ILT4+": ""
}

def main(tsv_path):
    df = pd.read_csv(tsv_path, sep='\t')
    if "Classification" not in df.columns:
        print("Error: 'Classification' column not found.")
        sys.exit(1)

    # Remove values ending with '-'
    df["Classification"] = df["Classification"].apply(remove_dash_suffix)

    # Split by '|' and flatten
    all_classes = df["Classification"].dropna().astype(str).str.split('|').explode()
    unique_classes = set(all_classes)

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

    #print("Unique values in file not found in lookup dict:")
    #for v in sorted(not_found):
    #    print(f"  {v}")

    print("\nUnique counts after remapping:")
    print(df["Classification"].value_counts())

    # Overwrite the input file with the cleaned data
    df.to_csv(tsv_path, sep='\t', index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fixup_remaining_phenotype_data_cleaning.py <input.tsv>")
        sys.exit(1)
    main(sys.argv[1])