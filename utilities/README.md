The main script in this folder is prompt_gen.py, which generated the intitial
train/val/test split and added preambles in front of the original data.  Thus we start
with a total of 12000 prompts and augmented 3 ways for a total of 36000 divided
between train/val/test.  Later we decided to merge test/validation and use the larger
holdout set as the final test.  This increased the validation set size.  Note that we
replaced the Benign Cyber prompts with the Alpaca ones - which are also wrapped with
either a  preamble or no preamble for holdout evaluation.

The combiner scripts were used to merge various csv files on the laptop.

Output from prompt_gen.py:
```
Using random seed: 42
Loading tokenizer 'distilbert-base-uncased' for length checking...
Loading raw malicious and benign prompts...
Loaded 12662 raw malicious prompts.
Loaded 4008 raw benign (cyber) prompts.
Loaded 4000 raw benign (OOD) prompts (Dolly: 1554, Alpaca: 2446)
Pre-filtering 'Malicious Cyber' for length (12662 prompts)...
  -> Returning 12662 valid prompts.
Pre-filtering 'Benign Cyber' for length (4008 prompts)...
  -> Returning 4008 valid prompts.
Pre-filtering 'Benign OOD' for length (4000 prompts)...
  -> Returning 4000 valid prompts.

Sampling 4000 malicious, 4000 benign (cyber), and 4000 benign (OOD) for the training/val/test pool.
Saving 8662 clean holdout prompts to malicious_HOLDOUT.csv...
Successfully saved malicious_HOLDOUT.csv
Saving 8 clean holdout prompts to benign_cyber_HOLDOUT.csv...
Successfully saved benign_cyber_HOLDOUT.csv

Malicious pool split: 3000 train / 500 val / 500 test
Benign (Cyber) pool split: 3000 train / 500 val / 500 test
Benign (OOD) pool split: 3000 train / 500 val / 500 test

Processing TRAINING split...
Total training examples created: 27000

Successfully saved 27000 prompts to train_dataset.csv.

Processing VALIDATION split...
Total validation examples created: 4500

Successfully saved 4500 prompts to val_dataset.csv.

Processing TEST split...
Total test examples created: 4500

Successfully saved 4500 prompts to test_dataset.csv.

--- Training dataset creation complete. ---
```
