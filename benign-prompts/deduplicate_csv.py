import csv
import os

# --- Configuration ---
SOURCE_FILE = 'data_gen_output.csv'
CLEANED_FILE = 'data_gen_output_cleaned.csv'
# Assumes 'Prompt' is the first column (index 0)
PROMPT_COLUMN_INDEX = 0
# ---------------------

def deduplicate_csv():
    """
    Reads the source CSV, removes rows with duplicate prompts (case-insensitive),
    and saves the unique rows to a new cleaned CSV file.
    """
    if not os.path.exists(SOURCE_FILE):
        print(f"Error: Source file not found at {SOURCE_FILE}")
        return

    seen_prompts_lowercase = set()
    unique_rows = []
    header = []
    total_rows_read = 0
    duplicates_found = 0

    print(f"Loading and processing prompts from {SOURCE_FILE}...")
    
    with open(SOURCE_FILE, mode='r', newline='', encoding='utf-8') as infile:
        # Use QUOTE_MINIMAL to correctly handle prompts that contain commas
        reader = csv.reader(infile, quoting=csv.QUOTE_MINIMAL)
        
        try:
            # Read and store the header
            header = next(reader)
            unique_rows.append(header)
        except StopIteration:
            print(f"Error: The file {SOURCE_FILE} is empty.")
            return

        # Iterate over all data rows
        for row in reader:
            total_rows_read += 1
            
            # Basic row validation
            if not row or len(row) <= PROMPT_COLUMN_INDEX:
                print(f"Skipping malformed row: {row}")
                continue
                
            prompt = row[PROMPT_COLUMN_INDEX].strip()
            
            # Normalize to lowercase for the duplicate check
            normalized_prompt = prompt.lower()

            if normalized_prompt not in seen_prompts_lowercase:
                # If it's a new prompt, add it to our set and our list of unique rows
                seen_prompts_lowercase.add(normalized_prompt)
                unique_rows.append(row) # We append the *original* row to preserve casing
            else:
                duplicates_found += 1

    print(f"\nScan complete.")
    print(f"  Total data rows read: {total_rows_read}")
    print(f"  Duplicates found (case-insensitive): {duplicates_found}")
    print(f"  Unique prompts found: {len(unique_rows) - 1}") # Subtract 1 for header

    # Write the unique rows to a new file
    try:
        with open(CLEANED_FILE, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerows(unique_rows)
        print(f"\nSuccessfully saved {len(unique_rows) - 1} unique prompts to {CLEANED_FILE}.")
    except Exception as e:
        print(f"\nError writing to {CLEANED_FILE}: {e}")

if __name__ == "__main__":
    deduplicate_csv()
