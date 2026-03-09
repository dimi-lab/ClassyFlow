#!/bin/bash

GCP_PATH="gs://ml-phi-staff-m088378-p-rsa-us-central1-p-6a4f/RaymondDev01/ClassyFlowDevelopment/*GM_3b*/*" # Get from EXPORT environment
LOCAL_TMP="/fslustre/qhs/ext_moore_raymond_mayo_edu/tmp_gcp_d2"
LOCAL_LOG="/home/ext_moore_raymond_mayo_edu/MyClassyConfigs/mixedLN_fixup.json"

mkdir -p "$LOCAL_TMP"

# List all files in the GCP bucket path
gsutil ls "$GCP_PATH" | while read -r gcp_file; do
    fname=$(basename "$gcp_file")
    # Skip empty filenames
    if [ -z "$fname" ]; then
        continue
    fi

    local_file="$LOCAL_TMP/$fname"
    echo "Processing $fname ..."

    # Download the file
    gsutil cp "$gcp_file" "$local_file"

    # Wait for the file to exist (max 30s)
    for i in {1..30}; do
        if [ -f "$local_file" ]; then
            break
        fi
        sleep 1
    done
    if [ ! -f "$local_file" ]; then
        echo "File $local_file did not appear after download. Skipping."
        continue
    fi

    # Count lines before
    orig_lines=$(wc -l < "$local_file")

    # Run the cleaning script
    echo -e "Begin cleaning $fname ..."
    # Change values in column 5 from 'TAM' to 'M2 Macrophage'
    awk -F '\t' 'BEGIN{OFS=FS} {$5 = ($5 == "TAM" ? "M2 Macrophage" : $5); print}' "$local_file" > "$local_file.tmp"
    mv "$local_file.tmp" "$local_file"

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