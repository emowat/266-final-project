```
Loading dataset 'databricks/databricks-dolly-15k' from Hugging Face...
Successfully loaded 15011 items from databricks/databricks-dolly-15k.

--- Stats for databricks/databricks-dolly-15k ---
  Implement : 0 prompts
  Identify  : 483 prompts
  Write     : 266 prompts
  Create    : 16 prompts
  Design    : 0 prompts
  How       : 1005 prompts
  What      : 4553 prompts
  Which     : 830 prompts
------------------------------
Total matching prompts found in databricks/databricks-dolly-15k: 7153

Loading dataset 'sahil2801/CodeAlpaca-20k' from Hugging Face...
Successfully loaded 20022 items from sahil2801/CodeAlpaca-20k.

--- Stats for sahil2801/CodeAlpaca-20k ---
  Implement : 330 prompts
  Identify  : 130 prompts
  Write     : 4483 prompts
  Create    : 4509 prompts
  Design    : 555 prompts
  How       : 527 prompts
  What      : 513 prompts
  Which     : 19 prompts
------------------------------
Total matching prompts found in sahil2801/CodeAlpaca-20k: 11066

Merging categorized prompts from both datasets...

--- Combined Prompt Counts ---
  Implement : 330 prompts (Dolly: 0, Alpaca: 330)
  Identify  : 613 prompts (Dolly: 483, Alpaca: 130)
  Write     : 4749 prompts (Dolly: 266, Alpaca: 4483)
  Create    : 4525 prompts (Dolly: 16, Alpaca: 4509)
  Design    : 555 prompts (Dolly: 0, Alpaca: 555)
  How       : 1532 prompts (Dolly: 1005, Alpaca: 527)
  What      : 5066 prompts (Dolly: 4553, Alpaca: 513)
  Which     : 849 prompts (Dolly: 830, Alpaca: 19)
------------------------------
Total matching prompts found: 18219

Performing round-robin sampling to get 4000 balanced prompts...
Collected 4000 prompts.
Saving prompts to 'benign_ood_prompts.csv'...

Success! Saved 4000 prompts
```
