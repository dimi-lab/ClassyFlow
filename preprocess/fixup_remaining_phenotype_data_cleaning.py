import pandas as pd
import sys
import re

upgradeMacs = True  # Set to False to downgrade M1/M2 to Macrophage

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
    "Tumor Cell": "Tumor",
    "Bcell": "B Cell",
    "CLT": "CytoT",
    "CTL": "CytoT",
    "Cytotoxic T Cell": "CytoT",
    "Helper T": "HelperT",
    "Helper T Cell": "HelperT",
    "M1": "M1 Macrophage",
    "M2": "M2 Macrophage",
    "MPO": "Neutro",
    "Neuto": "Neutro",
    "Neutro": "Neutro",
    "Neutrophil": "Neutro",
    "Stroma": "Epithelial",
    "SM": "Epithelial",
    "Treg": "T Reg",
    "Regulatory T Cell": "T Reg",
    "TumC": "Tumor", 
    "TumGas": "Tumor",
    "Vascul": "Vasculature",
    "Endo": "Endothelial",
    "Dendritic Cell": "DC",
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

# Optionally upgrade "Macrophage" to M1/M2 based on CD163/CD206 medians
def upgrade_macrophages(df):
    # Find columns containing both 'CD163' & 'Median', and 'CD206' & 'Median'
    cd163_col = next((col for col in df.columns if 'CD163' in col and 'Median' in col), None)
    cd206_col = next((col for col in df.columns if 'CD206' in col and 'Median' in col), None)
    if cd163_col and cd206_col:
        def upgrade_row(row):
            if row['Classification'] == 'Macrophage':
                try:
                    cd163 = float(row[cd163_col])
                    cd206 = float(row[cd206_col])
                    if cd206 > cd163:
                        return 'M2 Macrophage'
                    else:
                        return 'M1 Macrophage'
                except Exception:
                    return row['Classification']
            return row['Classification']
        df['Classification'] = df.apply(upgrade_row, axis=1)
    return df

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

    if upgradeMacs:
        df = upgrade_macrophages(df)

    print("\nUnique counts after remapping:")
    print(df["Classification"].value_counts())

    # Overwrite the input file with the cleaned data
    df.to_csv(tsv_path, sep='\t', index=False)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fixup_remaining_phenotype_data_cleaning.py <input.tsv>")
        sys.exit(1)
    main(sys.argv[1])