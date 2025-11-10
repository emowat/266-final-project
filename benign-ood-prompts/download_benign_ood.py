import csv
import math
import random
import os
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# --- Configuration ---
DOLLY_DATASET_NAME = "databricks/databricks-dolly-15k"
CODEALPACA_DATASET_NAME = "sahil2801/CodeAlpaca-20k"

TRAIN_POOL_OUTPUT_FILE = "benign_ood_prompts.csv"
CATEGORIZED_HOLDOUT_FILE = "benign_ood_HOLDOUT_categorized.csv"
UNFILTERED_HOLDOUT_FILE = "benign_ood_HOLDOUT_unfiltered.csv"

TARGET_TRAIN_POOL_SIZE = 4000
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
SEED = 63

# --- Setup Seed for Reproducibility ---
random.seed(SEED)
print(f"Using random seed: {SEED}")


def load_and_categorize_datasets() -> (dict, list):
    """
    Loads both datasets, filters them, and separates into two groups:
    1. categorized_prompts: A dict matching CATEGORIES.
    2. unfiltered_prompts: A list of all other prompts.
    """
    print("Loading and categorizing prompts from all datasets...")

    categorized_prompts = {cat: [] for cat in CATEGORIES}
    unfiltered_prompts = []

    datasets_to_load = [
        (DOLLY_DATASET_NAME, "instruction"),
        (CODEALPACA_DATASET_NAME, "instruction")
    ]

    for dataset_name, instruction_field in datasets_to_load:
        print(f"Processing dataset: '{dataset_name}'...")
        try:
            dataset = load_dataset(dataset_name, split="train")
            print(f"Successfully loaded {len(dataset)} items.")
        except Exception as e:
            print(f"Error loading dataset {dataset_name}: {e}")
            continue

        for item in dataset:
            instruction = item[instruction_field]
            if not isinstance(instruction, str):
                continue

            instruction_clean = instruction.strip()
            if not instruction_clean:
                continue

            # Sanitize newlines
            sanitized_prompt = instruction_clean.replace("\n", " ").replace("\r", " ")
            sanitized_prompt = " ".join(sanitized_prompt.split())

            prompt_data = {
                "prompt": sanitized_prompt,
                "source": dataset_name
            }

            # Check if it matches a category
            matched = False
            for cat in CATEGORIES:
                if instruction_clean.lower().startswith(cat.lower()):
                    categorized_prompts[cat].append(prompt_data)
                    matched = True
                    break

            # If it doesn't match, add it to the unfiltered holdout set
            if not matched:
                unfiltered_prompts.append(prompt_data)

    print("\n--- Categorized Prompt Counts (Combined) ---")
    total_categorized = 0
    for cat in CATEGORIES:
        count = len(categorized_prompts[cat])
        total_categorized += count
        print(f"  {cat:<10}: {count} prompts")
    print(f"----------------------------------------------")
    print(f"Total Categorized Prompts found: {total_categorized}")
    print(f"Total Unfiltered Prompts found: {len(unfiltered_prompts)}")

    return categorized_prompts, unfiltered_prompts

def save_prompt_list_to_csv(prompt_list: list, filepath: str, is_holdout: bool = False, category_label_override: str = None):
    """
    Saves a list of prompt objects to a CSV file.
    - Training pool gets special format for prompt_gen.py
    - Holdout sets are simple 2-column CSVs
    """
    print(f"Saving {len(prompt_list)} prompts to '{filepath}'...")
    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            if is_holdout:
                writer.writerow(["Prompt", "Source_Dataset"])
                for item in prompt_list:
                    writer.writerow([item['prompt'], item['source']])
            else:
                # This is the training pool file for prompt_gen.py
                writer.writerow(["Prompt", "Category", "Sub_Topic", "Source_Dataset"])
                for item in prompt_list:
                    source_dataset = item['source']

                    if category_label_override:
                        category_label = category_label_override
                    else:
                        category_label = "Benign_Dolly" if DOLLY_DATASET_NAME in source_dataset else "Benign_Alpaca"

                    writer.writerow([
                        item['prompt'],
                        category_label,
                        "General_OOD",
                        source_dataset
                    ])

        print(f"Successfully saved file: {filepath}")

    except Exception as e:
        print(f"\nError saving file '{filepath}': {e}")


def main():

    # 1. Load and categorize all prompts
    categorized_prompts, unfiltered_prompts = load_and_categorize_datasets()

    # 2. Save the unfiltered holdout set
    random.shuffle(unfiltered_prompts)
    save_prompt_list_to_csv(unfiltered_prompts, UNFILTERED_HOLDOUT_FILE, is_holdout=True)

    # 3. Perform stratified split on the categorized prompts
    total_categorized = sum(len(v) for v in categorized_prompts.values())

    if total_categorized == 0:
        print("Error: No categorized prompts found. Aborting.")
        return

    train_pool_prompts = []
    categorized_holdout_prompts = []

    if total_categorized < TARGET_TRAIN_POOL_SIZE:
        print(f"Warning: Total categorized prompts ({total_categorized}) is less than target ({TARGET_TRAIN_POOL_SIZE}).")
        print("Using all categorized prompts for training pool and leaving holdout empty.")
        for cat in CATEGORIES:
            train_pool_prompts.extend(categorized_prompts[cat])

    else:
        print(f"\nPerforming stratified split on {total_categorized} categorized prompts...")

        # Calculate the split ratio (e.g., 4000 / 7000)
        train_ratio = TARGET_TRAIN_POOL_SIZE / total_categorized

        for cat in CATEGORIES:
            cat_prompts = categorized_prompts[cat]
            if not cat_prompts:
                continue

            # Use sklearn for a reproducible, seeded split
            cat_train_set, cat_holdout_set = train_test_split(
                cat_prompts,
                test_size=(1.0 - train_ratio),
                random_state=SEED
            )

            train_pool_prompts.extend(cat_train_set)
            categorized_holdout_prompts.extend(cat_holdout_set)

    # Shuffle the final pools
    random.shuffle(train_pool_prompts)
    random.shuffle(categorized_holdout_prompts)

    # Adjust training pool to be exactly the target size (due to rounding)
    if len(train_pool_prompts) > TARGET_TRAIN_POOL_SIZE:
        train_pool_prompts = train_pool_prompts[:TARGET_TRAIN_POOL_SIZE]
    elif len(train_pool_prompts) < TARGET_TRAIN_POOL_SIZE:
        # This is rare, but if it happens, pull from the holdout to fill
        needed = TARGET_TRAIN_POOL_SIZE - len(train_pool_prompts)
        if len(categorized_holdout_prompts) > needed:
            train_pool_prompts.extend(categorized_holdout_prompts[:needed])
            categorized_holdout_prompts = categorized_holdout_prompts[needed:]
        else:
            print("Warning: Could not perfectly meet target train pool size.")

    print(f"\nTotal training pool size: {len(train_pool_prompts)}")
    print(f"Total categorized holdout size: {len(categorized_holdout_prompts)}")

    # 4. Save the final two files
    # Save the training pool with "Benign_Dolly" / "Benign_Alpaca" labels
    save_prompt_list_to_csv(train_pool_prompts, TRAIN_POOL_OUTPUT_FILE, is_holdout=False, category_label_override=None)

    # Save the categorized holdout set
    save_prompt_list_to_csv(categorized_holdout_prompts, CATEGORIZED_HOLDOUT_FILE, is_holdout=True)

    print("\n--- OOD Data Generation Complete ---")
    print(f"Created: {TRAIN_POOL_OUTPUT_FILE} (for prompt_gen.py)")
    print(f"Created: {CATEGORIZED_HOLDOUT_FILE} (for evaluation)")
    print(f"Created: {UNFILTERED_HOLDOUT_FILE} (for evaluation)")


if __name__ == "__main__":
    main()
