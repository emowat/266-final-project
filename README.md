# 266-final-project
Lightweight LLM Cybersecurity Guardrails

This repository contains the notebooks, utilities and data sets used to generate the
initial train/validation/holdout prompts.

benign-prompts: contains the scripts and data used to generate 4000 benign
cybersecurity prompts using the methodology in described by the CySecBench Paper

benign-ood-prompts: script and data to build the initial set of out-of-distribution
prompts using Databricks Dolly and Code_Alpaca datasets.

notebooks:
* Guardrail_Model_Trainer_V4.ipynb - this is the main training model that evaluate
each of the 3 BERT models against the train/validation data
* Holdout_Set_Evaluator.ipynb - computes accuracy and performace for each of the
different holdout categories.
* LatencyCalc.ipynb - Since the LlamaGuard/WildGard timings included tokenization,
compute tokenization timings for the BERT models so they can be added to the final
tallys.
* LlamaGuard3.ipynb - training notebook based on Guardrail trainer for LlamaGuard3
* Adversarial_Training_Set_Attacker.ipynb - using the V1 DistilBERT model, attack
prompts in the train/val using TextFooler and DeepWordBug.
* Adversarial_Validator.ipynb - Validate attacked prompts using LLM-as-a-Judge that
they retain their original malicous intent
* LMSYS_Benign_Validator.ipynb - attempt to collect 13000 prompts benign prompts from
the LMSYS dataset using LLM-as-a-Judge for validation
* utilities: Prompt generation and combiner scripts used to create datasets on the laptop

Data Files are located at:
https://drive.google.com/drive/folders/1UoCgWYPguyHSE-mVj9Ch6DcDBQtI4F3i?usp=sharing

* HoldoutResults - contain the data used in the paper
* guardrail_model* - v4 and v1 are used for the paper
* train_dataset_v4.csv / val_dataset_v4.csv is the final dataset used for the V4
model
