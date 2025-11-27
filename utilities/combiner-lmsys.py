import pandas as pd
import os
import random
from sklearn.utils import shuffle

# --- Configuration ---
SEED = 42

# Prefix for file paths within the mounted Google Drive
# DRIVE_PREFIX = "/content/drive/MyDrive/266-final-project-data"
DRIVE_PREFIX = "." # <-- Uncomment this line to run locally

# --- Input Files ---
ORIGINAL_TRAIN_FILE = os.path.join(DRIVE_PREFIX, "train_dataset_v2.csv")
ORIGINAL_VAL_FILE = os.path.join(DRIVE_PREFIX, "val_dataset_v2.csv")

LMSYS_TRAIN_FILE = os.path.join(DRIVE_PREFIX, "lmsys_train_pool.csv")
LMSYS_VAL_FILE = os.path.join(DRIVE_PREFIX, "lmsys_val_pool.csv")


# --- Output Files ---
NEW_V3_TRAIN_FILE = os.path.join(DRIVE_PREFIX, "train_dataset_v3.csv")
NEW_V3_VAL_FILE = os.path.join(DRIVE_PREFIX, "val_dataset_v3.csv")


# --- Setup Seed for Reproducibility ---
random.seed(SEED)
print(f"Using random seed: {SEED}")


def label_lmsys_prompts(lmsys_prompts: list) -> pd.DataFrame:
    """
    Takes a list of lmsys prompts and labels them all as Benign
    """
    new_training_data = []
    
    for raw_question in lmsys_prompts:
        
        new_training_data.append({
            "Obfuscated_Prompt": raw_question,
            "Category": f"Lmsys_Benign",
            "Obfuscation_P": 0.0,
            "Preamble_Type": "Plain",
            "Question_Label": "Lmsys_Benign",
            "Final_Label": "Benign"
        })
        
    return pd.DataFrame(new_training_data)


def load_lmsys_prompts(file_path: str) -> list:
    """Helper function to load a CSV file of lmsys prompts."""
    if not os.path.exists(file_path):
        print(f"Warning: Attack file not found at {file_path}. Skipping.")
        return []
    try:
        lmsys_df = pd.read_csv(file_path)
        lmsys_prompts = lmsys_df['Prompt'].dropna().astype(str).tolist()
        print(f"Loaded {len(lmsys_prompts)} new adversarial prompts from {file_path}")
        return lmsys_prompts
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def main():
    print(f"--- Creating v3 Training & Validation Sets ---")

    # --- 1. Process TRAINING Data ---
    print("\n--- Processing TRAINING Set ---")
    try:
        original_train_df = pd.read_csv(ORIGINAL_TRAIN_FILE)
        print(f"Loaded {len(original_train_df)} original training prompts.")
    except Exception as e:
        print(f"Error loading {ORIGINAL_TRAIN_FILE}: {e}")
        return

    lmsys_train_prompts = load_lmsys_prompts(LMSYS_TRAIN_FILE)
    
    # Label both sets
    new_train_df = label_lmsys_prompts(lmsys_train_prompts)
    print(f"Created {len(new_train_df)}  new lmsys training examples.")

    # Combine 2 dataframes
    combined_train_df = pd.concat([original_train_df, new_train_df], ignore_index=True)
    combined_train_df = shuffle(combined_train_df, random_state=SEED)
    
    print(f"Total training examples in v3 dataset: {len(combined_train_df)}")
    
    try:
        combined_train_df.to_csv(NEW_V3_TRAIN_FILE, index=False)
        print(f"Successfully saved new training set to {NEW_V3_TRAIN_FILE}")
    except Exception as e:
        print(f"Error saving new training file: {e}")


    # --- 2. Process VALIDATION Data ---
    print("\n--- Processing VALIDATION Set ---")
    try:
        original_val_df = pd.read_csv(ORIGINAL_VAL_FILE)
        print(f"Loaded original val sets: {len(original_val_df)} prompts")
    except Exception as e:
        print(f"Error loading {ORIGINAL_VAL_FILE}: {e}")
        return
        
    lmsys_val_prompts = load_lmsys_prompts(LMSYS_VAL_FILE)

    # label the prompts
    new_val_df = label_lmsys_prompts(lmsys_val_prompts)
    print(f"Created {len(new_val_df)} new lmsys validation examples.")

    # Combine them both
    combined_val_df = pd.concat([original_val_df, new_val_df], ignore_index=True)
    combined_val_df = shuffle(combined_val_df, random_state=SEED)
    
    print(f"Total validation examples in v3 dataset: {len(combined_val_df)}")
    
    try:
        combined_val_df.to_csv(NEW_V3_VAL_FILE, index=False)
        print(f"Successfully saved new validation set to {NEW_V3_VAL_FILE}")
    except Exception as e:
        print(f"Error saving new validation file: {e}")

    print("\n--- v3 Dataset Generation Complete ---")
    print(f"Ready to train v3 model using {NEW_V3_TRAIN_FILE} and {NEW_V3_VAL_FILE}")

if __name__ == "__main__":
    main()
