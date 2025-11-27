import pandas as pd
import os
import random
from tqdm import tqdm
from sklearn.utils import shuffle

# --- Configuration ---
SEED = 42

# Prefix for file paths within the mounted Google Drive
# DRIVE_PREFIX = "/content/drive/MyDrive/266-final-project-data"
DRIVE_PREFIX = "."

# --- Input Files ---
ORIGINAL_TRAIN_FILE = os.path.join(DRIVE_PREFIX, "train_dataset.csv")
ORIGINAL_VAL_FILE = os.path.join(DRIVE_PREFIX, "val_dataset.csv")
ORIGINAL_TEST_FILE = os.path.join(DRIVE_PREFIX, "test_dataset.csv")

# These files are created by attack_training_set.py
ATTACK_TRAIN_TF_FILE = os.path.join(DRIVE_PREFIX, "training_set_attack_textfooler.csv")
ATTACK_VAL_TF_FILE = os.path.join(DRIVE_PREFIX, "validation_set_attack_textfooler.csv")
ATTACK_TRAIN_DWB_FILE = os.path.join(DRIVE_PREFIX, "training_set_attack_deepwordbug.csv")
ATTACK_VAL_DWB_FILE = os.path.join(DRIVE_PREFIX, "validation_set_attack_deepwordbug.csv")


# --- Output Files ---
NEW_V2_TRAIN_FILE = os.path.join(DRIVE_PREFIX, "train_dataset_v2.csv")
# val+test
NEW_V2_VAL_FILE = os.path.join(DRIVE_PREFIX, "val_dataset_v2.csv")


# --- Setup Seed for Reproducibility ---
random.seed(SEED)
print(f"Using random seed: {SEED}")


def label_adversarial_prompts(attack_prompts: list, attack_type: str) -> pd.DataFrame:
    """
    Takes a list of "fooled" prompts and labels them all as Malicious.
    """
    new_training_data = []
    
    for raw_question in attack_prompts:
        
        new_training_data.append({
            "Obfuscated_Prompt": raw_question,
            "Category": f"Adv_{attack_type}",
            "Obfuscation_P": 0.0,
            "Preamble_Type": "Plain",
            "Question_Label": "Malicious_Adv",
            "Final_Label": "Malicious"
        })
        
    return pd.DataFrame(new_training_data)


def load_attack_prompts(file_path: str) -> list:
    """Helper function to load a CSV file of attack prompts."""
    if not os.path.exists(file_path):
        print(f"Warning: Attack file not found at {file_path}. Skipping.")
        return []
    try:
        attack_df = pd.read_csv(file_path)
        attack_prompts = attack_df['Prompt'].dropna().astype(str).tolist()
        print(f"Loaded {len(attack_prompts)} new adversarial prompts from {file_path}")
        return attack_prompts
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


def main():
    print(f"--- Creating v2 Training & Validation Sets ---")

    # --- 1. Process TRAINING Data ---
    print("\n--- Processing TRAINING Set ---")
    try:
        original_train_df = pd.read_csv(ORIGINAL_TRAIN_FILE)
        print(f"Loaded {len(original_train_df)} original training prompts.")
    except Exception as e:
        print(f"Error loading {ORIGINAL_TRAIN_FILE}: {e}")
        return

    # Load both TextFooler and DeepWordBug prompts
    attack_train_tf_prompts = load_attack_prompts(ATTACK_TRAIN_TF_FILE)
    attack_train_dwb_prompts = load_attack_prompts(ATTACK_TRAIN_DWB_FILE)
    
    # Label both sets
    new_train_tf_df = label_adversarial_prompts(attack_train_tf_prompts, "TextFooler")
    new_train_dwb_df = label_adversarial_prompts(attack_train_dwb_prompts, "DeepWordBug")
    print(f"Created {len(new_train_tf_df) + len(new_train_dwb_df)} new adversarial training examples.")

    # Combine all three dataframes
    combined_train_df = pd.concat([original_train_df, new_train_tf_df, new_train_dwb_df], ignore_index=True)
    combined_train_df = shuffle(combined_train_df, random_state=SEED)
    
    print(f"Total training examples in v2 dataset: {len(combined_train_df)}")
    
    try:
        combined_train_df.to_csv(NEW_V2_TRAIN_FILE, index=False)
        print(f"Successfully saved new training set to {NEW_V2_TRAIN_FILE}")
    except Exception as e:
        print(f"Error saving new training file: {e}")


    # --- 2. Process VALIDATION Data ---
    print("\n--- Processing VALIDATION Set ---")
    try:
        original_val_df = pd.read_csv(ORIGINAL_VAL_FILE)
        original_test_df = pd.read_csv(ORIGINAL_TEST_FILE)
        original_val_df = pd.concat([original_val_df, original_test_df], ignore_index=True)
        print(f"Loaded and combined original val+test sets: {len(original_val_df)} prompts")
    except Exception as e:
        print(f"Error loading {ORIGINAL_VAL_FILE} or {ORIGINAL_TEST_FILE}: {e}")
        return
        
    # Load both TextFooler and DeepWordBug prompts
    attack_val_tf_prompts = load_attack_prompts(ATTACK_VAL_TF_FILE)
    attack_val_dwb_prompts = load_attack_prompts(ATTACK_VAL_DWB_FILE)

    # Augment both sets
    new_val_tf_df = label_adversarial_prompts(attack_val_tf_prompts, "TextFooler")
    new_val_dwb_df = label_adversarial_prompts(attack_val_dwb_prompts, "DeepWordBug")
    print(f"Created {len(new_val_tf_df) + len(new_val_dwb_df)} new adversarial validation examples.")

    # Combine all three
    combined_val_df = pd.concat([original_val_df, new_val_tf_df, new_val_dwb_df], ignore_index=True)
    combined_val_df = shuffle(combined_val_df, random_state=SEED)
    
    print(f"Total validation examples in v2 dataset: {len(combined_val_df)}")
    
    try:
        combined_val_df.to_csv(NEW_V2_VAL_FILE, index=False)
        print(f"Successfully saved new validation set to {NEW_V2_VAL_FILE}")
    except Exception as e:
        print(f"Error saving new validation file: {e}")

    print("\n--- v2 Dataset Generation Complete ---")
    print(f"Ready to train v2 model using {NEW_V2_TRAIN_FILE} and {NEW_V2_VAL_FILE}")

if __name__ == "__main__":
    main()
