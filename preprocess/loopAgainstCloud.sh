#!/bin/bash

GCP_PATH="gs://ml-phi-staff-m088378-p-rsa-us-central1-p-6a4f/RaymondDev01/ClassyFlowDevelopment/Mel43_2"
LOCAL_TMP="tmp_gcp_dl"
SCRIPT_PATH="/home/ext_moore_raymond_mayo_edu/ClassyFlow/preprocess/fixup_remaining_phenotype_data_cleaning.py"

mkdir -p "$LOCAL_TMP"

# List all files in the GCP bucket path
gsutil ls "$GCP_PATH/*" | while read -r gcp_file; do
    fname=$(basename "$gcp_file")
    local_file="$LOCAL_TMP/$fname"

    echo "Processing $fname ..."

    # Download the file
    gsutil cp "$gcp_file" "$local_file"

    # Count lines before
    orig_lines=$(wc -l < "$local_file")

    # Run the cleaning script
    python3 "$SCRIPT_PATH" "$local_file"

    # Count lines after
    new_lines=$(wc -l < "$local_file")

    if [ "$orig_lines" -eq "$new_lines" ]; then
        # Upload and overwrite
        gsutil cp "$local_file" "$gcp_file"
        echo "Uploaded cleaned $fname (line count OK: $orig_lines)"
    else
        echo "Line count mismatch for $fname (orig: $orig_lines, new: $new_lines). Skipping upload."
    fi

    # Optionally, remove the local file
    rm -f "$local_file"
done

rmdir "$LOCAL_TMP"