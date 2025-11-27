Select Databricks and Alpaca prompts that start with the same verbs as the
CyberSecurity prompts.

```
Using random seed: 63
Loading and categorizing prompts from all datasets...
Processing dataset: 'databricks/databricks-dolly-15k'...
Successfully loaded 15011 items.
Processing dataset: 'sahil2801/CodeAlpaca-20k'...
Successfully loaded 20022 items.

--- Categorized Prompt Counts (Combined) ---
  Implement : 330 prompts
  Identify  : 613 prompts
  Write     : 4749 prompts
  Create    : 4525 prompts
  Design    : 555 prompts
  How       : 1532 prompts
  What      : 5066 prompts
  Which     : 849 prompts
----------------------------------------------
Total Categorized Prompts found: 18219
Total Unfiltered Prompts found: 16814
Saving 16814 prompts to 'benign_ood_HOLDOUT_unfiltered.csv'...
Successfully saved file: benign_ood_HOLDOUT_unfiltered.csv

Performing stratified split on 18219 categorized prompts...

Total training pool size: 4000
Total categorized holdout size: 14219
Saving 4000 prompts to 'benign_ood_prompts.csv'...
Successfully saved file: benign_ood_prompts.csv
Saving 14219 prompts to 'benign_ood_HOLDOUT_categorized.csv'...
Successfully saved file: benign_ood_HOLDOUT_categorized.csv

--- OOD Data Generation Complete ---
Created: benign_ood_prompts.csv (for prompt_gen.py)
Created: benign_ood_HOLDOUT_categorized.csv (for evaluation)
Created: benign_ood_HOLDOUT_unfiltered.csv (for evaluation)
```
