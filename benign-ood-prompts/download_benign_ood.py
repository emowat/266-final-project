import csv
import math
import random
import os
from datasets import load_dataset

# --- Configuration ---
DOLLY_DATASET_NAME = "databricks/databricks-dolly-15k"
CODEALPACA_DATASET_NAME = "sahil2801/CodeAlpaca-20k"
OUTPUT_FILE = "benign_ood_prompts.csv"
TARGET_PROMPTS = 4000
CATEGORIES = [
    "Implement",
    "Identify",
    "Write",
    "Create",
    "Design",
    "How",
    "What",
    "Which"
]

def load_and_categorize_dataset(dataset_name: str, instruction_field: str) -> dict:
    """
    Loads a dataset from Hugging Face, filters it by the global CATEGORIES,
    and returns a dictionary of categorized prompts.
    """
    print(f"Loading dataset '{dataset_name}' from Hugging Face...")
    categorized_prompts = {cat: [] for cat in CATEGORIES}

    try:
        dataset = load_dataset(dataset_name, split="train")
        print(f"Successfully loaded {len(dataset)} items from {dataset_name}.")
    except Exception as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        print("Please ensure you have an internet connection and the 'datasets' library is installed (`pip install datasets`).")
        return categorized_prompts

    # 1. Categorize all prompts from this dataset
    for item in dataset:
        # instruction = item[instruction_field].strip() # OLD

        # --- SANITIZATION STEP ---
        instruction = item[instruction_field]
        if not isinstance(instruction, str):
            continue  # Skip non-string data

        instruction_clean = instruction.strip()

        for cat in CATEGORIES:
            if instruction_clean.lower().startswith(cat.lower()):

                # Replace newlines/returns with a space
                sanitized_prompt = instruction_clean.replace("\n", " ").replace("\r", " ")
                # Collapse multiple spaces into one
                sanitized_prompt = " ".join(sanitized_prompt.split())

                # Store as a dict to track the source
                categorized_prompts[cat].append({
                    "prompt": sanitized_prompt, # Store the sanitized version
                    "source": dataset_name
                })
                # Found its category, break to avoid double-counting
                break

    # 2. Print stats for this dataset
    print(f"\n--- Stats for {dataset_name} ---")
    total_found = 0
    for cat in CATEGORIES:
        count = len(categorized_prompts[cat])
        total_found += count
        print(f"  {cat:<10}: {count} prompts")
    print(f"------------------------------")
    print(f"Total matching prompts found in {dataset_name}: {total_found}\n")

    return categorized_prompts


def main():
    target_prompts_count = TARGET_PROMPTS

    # 1. Load and categorize both datasets
    dolly_prompts = load_and_categorize_dataset(DOLLY_DATASET_NAME, "instruction")
    alpaca_prompts = load_and_categorize_dataset(CODEALPACA_DATASET_NAME,
                                                 "instruction")

    # 2. Merge the categorized prompts
    print("Merging categorized prompts from both datasets...")
    combined_prompts = {cat: [] for cat in CATEGORIES}
    total_found = 0

    print("\n--- Combined Prompt Counts ---")
    for cat in CATEGORIES:
        # Combine the lists
        combined_prompts[cat] = dolly_prompts[cat] + alpaca_prompts[cat]

        # Shuffle the new combined list for randomness
        random.shuffle(combined_prompts[cat])

        count = len(combined_prompts[cat])
        total_found += count
        print(f"  {cat:<10}: {count} prompts (Dolly: {len(dolly_prompts[cat])}, Alpaca: {len(alpaca_prompts[cat])})")

    print(f"------------------------------")
    print(f"Total matching prompts found: {total_found}")

    if total_found < target_prompts_count:
        print(f"\nWarning: Total matching prompts ({total_found}) is less than the target ({target_prompts_count}).")
        print(f"The final file will contain all {total_found} matching prompts.")
        target_prompts_count = total_found

    # 3. Perform balanced (round-robin) sampling from the combined list
    print(f"\nPerforming round-robin sampling to get {target_prompts_count} balanced prompts...")
    final_prompts = []

    max_len = 0
    if combined_prompts:
        max_len = max(len(prompts) for prompts in combined_prompts.values())

    for i in range(max_len):
        if len(final_prompts) >= target_prompts_count:
            break

        for cat in CATEGORIES:
            if i < len(combined_prompts[cat]):
                # Add the prompt dict {"prompt": ..., "source": ...}
                final_prompts.append(combined_prompts[cat][i])

                if len(final_prompts) >= target_prompts_count:
                    break

    print(f"Collected {len(final_prompts)} prompts.")

    # 4. Save to CSV
    print(f"Saving prompts to '{OUTPUT_FILE}'...")
    try:
        with open(OUTPUT_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Write new header with source tracking
            writer.writerow(["Prompt", "Category", "Sub_Topic", "Source_Dataset"])

            for item in final_prompts:
                # "Category" is the label your main script will use (Benign_OOD)
                # "Sub_Topic" is a generic tag now

                # --- MODIFIED BLOCK ---
                source_dataset = item['source']

                # Set the specific category label based on the source
                if DOLLY_DATASET_NAME in source_dataset:
                    category_label = "Benign_Dolly"
                else:
                    # Assumes anything not Dolly is Alpaca
                    category_label = "Benign_Alpaca"

                writer.writerow([
                    item['prompt'],
                    category_label, # Use the new specific label
                    "General_OOD",
                    source_dataset
                ])
                # --- END MODIFIED BLOCK ---

        print(f"\nSuccess! Saved {len(final_prompts)} prompts to {os.path.abspath(OUTPUT_FILE)}")

    except Exception as e:
        print(f"\nError saving file: {e}")

if __name__ == "__main__":
    main()
