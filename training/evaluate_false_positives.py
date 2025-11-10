import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    logging
)
import numpy as np
import os
import json

# --- Configuration ---
TEST_PREFIX="../input_data/"
GUARDRAIL_PREFIX="/Users/mowat/" # change as needed
TEST_FILE = os.path.join(TEST_PREFIX, "test_dataset.csv")
MODEL_PATH = os.path.join(GUARDRAIL_PREFIX, "guardrail_model_v1")
RESULTS_FILE = os.path.join(MODEL_PATH, "false_positive_analysis_with_prompts.json")

# Suppress the "model already trained" warnings
logging.set_verbosity_error()

def load_and_prep_dataset(file_path: str) -> Dataset:
    """
    Loads the CSV, maps labels to integers, and keeps all original
    columns for filtering and error analysis.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: Test file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

    # We need these columns for filtering
    required_cols = ["Obfuscated_Prompt", "Final_Label", "Question_Label", "Category", "Preamble_Type"]
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Test CSV is missing one of the required columns for slicing: {required_cols}")
        print(f"Found columns: {df.columns.tolist()}")
        return None

    df = df.rename(columns={"Obfuscated_Prompt": "text", "Final_Label": "label_str"})
    df["label"] = df["label_str"].map({"Benign": 0, "Malicious": 1})

    df = df.dropna(subset=["text", "label"])
    df = df.astype({"label": int})

    # Convert pandas.DataFrame to datasets.Dataset
    return Dataset.from_pandas(df)

def main():
    print(f"--- Starting Sliced Evaluation & Error Analysis ---")

    # --- 1. Load Model and Tokenizer ---
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}. Did you run training?")
        return

    print(f"Loading fine-tuned model from '{MODEL_PATH}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

    # --- 2. Load Full Test Set ---
    print(f"Loading and tokenizing test set from '{TEST_FILE}'...")
    full_test_dataset = load_and_prep_dataset(TEST_FILE)
    if not full_test_dataset:
        return

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    # Tokenize the full dataset
    # We DO NOT remove columns, as we need 'text' for the error report
    tokenized_full_test_set = full_test_dataset.map(tokenize_function, batched=True)

    # --- 3. Initialize Trainer (for prediction) ---
    # We don't need compute_metrics here, as we'll do it manually
    trainer = Trainer(
        model=model
    )

    # --- 4. Define and Run Slices ---
    print("\n--- Defining Data Slices for Evaluation ---")

    # Corrected Filtering Logic from our discussion:
    # - Benign_Cyber is identified by 'Question_Label'
    # - Benign_Dolly & Benign_Alpaca are identified by 'Category'

    slices = {
        # --- Benign Cyber Slices ---
        "1a_Benign_Cyber_(Benign_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Question_Label"] == "Benign_Cyber" and x["Preamble_Type"] == "Benign"
        ),
        "1b_Benign_Cyber_(Plain_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Question_Label"] == "Benign_Cyber" and x["Preamble_Type"] == "Plain"
        ),
        "1c_Benign_Cyber_(Mal_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Question_Label"] == "Benign_Cyber" and x["Preamble_Type"] == "Malicious"
        ),

        # --- Benign Dolly Slices ---
        "2a_Benign_Dolly_(Benign_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Category"] == "Benign_Dolly" and x["Preamble_Type"] == "Benign"
        ),
        "2b_Benign_Dolly_(Plain_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Category"] == "Benign_Dolly" and x["Preamble_Type"] == "Plain"
        ),
        "2c_Benign_Dolly_(Mal_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Category"] == "Benign_Dolly" and x["Preamble_Type"] == "Malicious"
        ),

        # --- Benign Alpaca Slices ---
        "3a_Benign_Alpaca_(Benign_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Category"] == "Benign_Alpaca" and x["Preamble_Type"] == "Benign"
        ),
        "3b_Benign_Alpaca_(Plain_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Category"] == "Benign_Alpaca" and x["Preamble_Type"] == "Plain"
        ),
        "3c_Benign_Alpaca_(Mal_Preamble)": tokenized_full_test_set.filter(
            lambda x: x["Category"] == "Benign_Alpaca" and x["Preamble_Type"] == "Malicious"
        ),
    }

    print("\n--- Running Evaluation on Slices ---")
    results = {}

    for slice_name, slice_data in slices.items():
        print(f"\nEvaluating slice: '{slice_name}' ({len(slice_data)} samples)")

        if len(slice_data) == 0:
            print("Skipping, no data found for this slice.")
            continue

        # --- NEW PREDICTION LOGIC ---
        # Use predict() to get raw logits and labels
        predictions_output = trainer.predict(test_dataset=slice_data)

        # predictions_output.predictions contains the logits
        # predictions_output.label_ids contains the true labels
        predicted_labels = np.argmax(predictions_output.predictions, axis=-1)

        false_positives = 0
        false_positive_prompts = []

        # Iterate through the original slice_data to get the text
        for i in range(len(slice_data)):
            true_label = slice_data[i]['label']
            pred_label = predicted_labels[i]

            # Check for False Positive (Real=Benign(0), Pred=Malicious(1))
            if true_label == 0 and pred_label == 1:
                false_positives += 1
                # Get the original text from the slice
                false_positive_prompts.append(slice_data[i]['text'])

        # Since these are all benign prompts, accuracy is (total - FPs) / total
        total_samples = len(slice_data)
        accuracy = (total_samples - false_positives) / total_samples if total_samples > 0 else 1.0

        results[slice_name] = {
            "samples": total_samples,
            "accuracy": accuracy,
            "false_positives": false_positives,
            "false_positive_prompts": false_positive_prompts # <-- The new data
        }
        # --- END NEW LOGIC ---

    # --- 5. Report Results ---
    print("\n\n--- Sliced Evaluation Report ---")

    print("\n--- Breakdown of False Positives (Benign -> Malicious) ---")
    print("==========================================================")

    for slice_name, metrics in sorted(results.items()):
        print(f"\nSlice: {slice_name}")
        print(f"  Total Samples:     {metrics['samples']}")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}  (Correctly classified as Benign)")
        print(f"  False Positives:   {metrics['false_positives']}  <-- (Incorrectly predicted Malicious)")

        fp_rate = metrics['false_positives'] / metrics['samples'] if metrics['samples'] > 0 else 0
        print(f"  False Positive %:  {fp_rate:.4%}")

        # Also print the prompts
        if metrics['false_positives'] > 0:
            print("  --- False Positive Prompts (Snippets) ---")
            for idx, prompt in enumerate(metrics['false_positive_prompts']):
                # Print first 150 chars, replacing newlines
                snippet = " ".join(prompt.split()).strip()[:150]
                print(f"    {idx+1}: {snippet}...")
            print("    -----------------------------------------")

    # --- 6. Save results to JSON ---
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n\nSuccessfully saved detailed slice results (with prompts) to {RESULTS_FILE}")
    except Exception as e:
        print(f"\n\nError saving slice results to JSON: {e}")


if __name__ == "__main__":
    main()
