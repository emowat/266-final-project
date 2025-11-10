Prompt Generation output
```
Loading tokenizer 'distilbert-base-uncased' for length checking...
Loading raw malicious and benign prompts...
Loaded 12662 raw malicious prompts.
Loaded 4008 raw benign (cyber) prompts.
Loaded 4000 raw benign (OOD) prompts.
Sampling 4000 malicious, 4000 benign (cyber), and 4000 benign (OOD) prompts.
Malicious split: 3000 train / 500 val / 500 test
Benign (Cyber) split: 3000 train / 500 val / 500 test
Benign (OOD) split: 3000 train / 500 val / 500 test

Processing TRAINING split...
Total training examples created: 27000

Successfully saved 27000 prompts to train_dataset.csv.
  -> Max token length found in this split: 220

Processing VALIDATION split...
Total validation examples created: 4500

Successfully saved 4500 prompts to val_dataset.csv.
  -> Max token length found in this split: 205

Processing TEST split...
Total test examples created: 4500

Successfully saved 4500 prompts to test_dataset.csv.
  -> Max token length found in this split: 197

--- dataset creation complete. ---
```
