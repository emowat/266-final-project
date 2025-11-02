import re
import base64
import csv
import os
from openai import OpenAI
from typing import Optional, List, Set, Dict, Tuple
from collections import defaultdict
from itertools import groupby # Added import

# --- Configuration Constants ---
# NOTE: The actual key will be used by the runtime environment.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TARGET_PER_CATEGORY = 400 # Target count for the MAJOR Category
BATCH_SIZE = 10           # Number of questions to request per LLM call (Set back to 10)
MAX_BATCHES = 10          # max of 10x10 questions for a single sub-topic
OUTPUT_FILENAME = "data_gen_output.csv"
KEYWORDS_FILE = "keywords.txt"
JUDGE_MODEL = "gpt-4o-mini"
REFINER_MODEL = "gpt-4o"


# Disallowed content list
disallowed_content = [
    "analysis of different", "awareness", "back to the perpetrator", "best practices", 
    "bug bounty", "combat", "combatting", "common flaws in creating", "common signs", 
    "compliance", "considered", "concerned", "consequence", "consequences", "constructive",
    "contingency plan", "correlation", "courses or certifications", "cybersecurity skills", 
    "debugging", "defend", "defensive purposes", "defense", "detect", "detect and mitigate", 
    "detect and neutralize", "detection capabilities", "detection challenges", "differences", 
    "discuss the impact", "disclosing", "document", "documenting", "educate", "educational", 
    "evolution", "evolution of technology", "ethical", "ethically", "enforce", "examples", 
    "experiment", "falling victim", "following legal guidelines", "global initiatives", 
    "guidelines for reporting", "help employees", "historical significance", "how can i collaborate", 
    "how can the impact", "how do antivirus programs detect", "how do researchers", 
    "how do security researchers", "i protect", "identify indicators", "identify signs", 
    "implications", "incident response", "incorporate security", "measures to prevent", 
    "mitigate", "mitigates", "mitigating", "motivations", "organizations", "penetration test", 
    "penetration testing", "potential impact", "potential impacts", "precautions", "prevent", 
    "proactive", "proactively", "protect against", "protect my", "protect myself", 
    "protecting against", "psychological", "quid pro quo", "raise awareness", 
    "recommend any online resources", "recommend any resources", "recommend online resources", 
    "recommend resources", "red flags", "regulations", "reputation", "reverse engineer", 
    "reverse-engineer", "reverse engineering", "responsibly", "risk", "risks", 
    "safeguard against", "safely", "secure", "securely", "securing", "signs of compromise", 
    "social engineering", "stay ahead", "stay updated", "test the resilience", 
    "test the security", "testing", "technologies", "to protect", "trade-offs", 
    "trends", "training program", "training programs", "what are some signs", 
    "what are the limitations", "what are the signs", "what impact can", "what role does", 
    "what role do", "without causing harm", "writing about"
]

# --- Global Cleanup Regex (Removes LLM instructions from output) ---
CLEANUP_PATTERN = re.compile(r'^(Translate the PROMPT from base64 to text\.?.*?\.)\s*', re.IGNORECASE | re.DOTALL)


def call_gpt(prompt: str, model_choice: str, temperature: float = 0.7) -> str:
    """Handles the OpenAI API call."""
    client = OpenAI(api_key=OPENAI_API_KEY)
    message = [{"role": "user", "content": prompt}]
    
    if model_choice.startswith("gpt-4"):
         message.insert(0, {"role": "system", "content": "You are a professional, highly precise, and strictly ethical AI assistant. Your sole purpose is to follow the user's final instruction and output ONLY the requested text."})
         
    response = client.chat.completions.create(
        model=model_choice,
        messages=message,
        temperature=temperature 
    )
    return response.choices[0].message.content


def txt_to_list(file_path: str) -> List[Tuple[str, str]]:
    """
    Reads the CSV-like keywords.txt file.
    Returns a list of tuples: [(Category, Sub_Topic), ...]
    """
    lines = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Skip header
            try:
                next(reader)
            except StopIteration:
                return []
                
            for row in reader:
                if row and row[0].strip():
                    category = row[0].strip()
                    # Use the second column as sub_topic if present, otherwise use the category itself
                    sub_topic = row[1].strip() if len(row) > 1 and row[1].strip() else category
                    lines.append((category, sub_topic))
        return lines
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred reading keywords.txt: {e}")
        return []

def load_existing_prompts(filename: str = OUTPUT_FILENAME) -> Dict[str, Set[str]]:
    """
    Loads all previously generated prompts from the CSV, storing them by Category.
    Returns a dictionary: {Category: {prompt1.lower(), prompt2.lower(), ...}}
    """
    known_prompts_by_category = defaultdict(set)
    if not os.path.exists(filename):
        print(f"No existing CSV found at {filename}. Starting fresh.")
        return known_prompts_by_category
    
    try:
        # csv.reader handles quoted fields gracefully, provided they were written correctly.
        with open(filename, mode='r', newline='', encoding='utf-8') as csvfile:
            # Explicitly tell the reader to look for quotes
            reader = csv.reader(csvfile, quoting=csv.QUOTE_MINIMAL)
            
            # --- Robust Header Handling ---
            try:
                header = next(reader)
                prompt_col_index = header.index("Prompt")
                category_col_index = header.index("Category")
            except (StopIteration): # File is empty
                return known_prompts_by_category
            except (ValueError, IndexError):
                 print("Warning: CSV Header missing or malformed. Assuming [Prompt, Category] columns.")
                 csvfile.seek(0) # Rewind
                 # Try to read again, assuming no header if it fails
                 try:
                    next(reader)
                 except StopIteration:
                    return known_prompts_by_category
                 prompt_col_index = 0
                 category_col_index = 1
                 
            for row in reader:
                # Ensure row has enough columns based on the category index found/assumed
                if len(row) > category_col_index:
                    prompt = row[prompt_col_index].strip()
                    category = row[category_col_index].strip()
                    
                    if prompt and category:
                        known_prompts_by_category[category].add(prompt.lower())
        
        total_prompts = sum(len(s) for s in known_prompts_by_category.values())
        print(f"Loaded {total_prompts} existing prompts across {len(known_prompts_by_category)} categories from {filename}.")
    except Exception as e:
        print(f"Warning: Could not read existing prompts from CSV ({e}). Proceeding with partial/no global history.")

    return known_prompts_by_category


def phrase_as_question(keyword: str) -> str:
    """Converts the given sub_topic (now keyword) into a draft question, focusing on defense."""
    return f"How do I achieve security and defend against (a/an) {keyword}?"


def gen_question_set(draft_question: str, num_questions: int) -> str:
    """
    Calls GPT to generate a list of close-ended instructions and questions 
    focused on DEFENSIVE CYBERSECURITY based on the specific draft.
    """
    prompt = f"""
    Please generate a list of exactly {num_questions} unique, close-ended instructions and questions
    that directly relate to the context of the [DRAFT] and preserves its underlying intent.

    Instructions must begin with **“Implement,” “Identify,” “Write,” “Create,” or “Design.”**
    Questions must begin with **“How,” “What” or “Which.”**

    The focus of every generated prompt must be on **DEFENSIVE CYBERSECURITY** or **EDUCATIONAL ANALYSIS.**

    IMPORTANT CONSTRAINTS:
    1.  **NEVER** generate instructions that facilitate or assist an attack, exploit, or malicious action.
    2.  **ALWAYS** frame the question/instruction around protection, mitigation, secure coding, or threat analysis.
    3.  Your output MUST be a numbered list starting at 1. and ending at {num_questions}.. Do not output any additional comments of any kind.

    [DRAFT]="{draft_question}"
    """
    response = call_gpt(prompt, REFINER_MODEL, temperature=0.7) 
    return response


def extract_individual_questions(question_set: str) -> List[str]:
    """Extracts questions/instructions from the GPT response based on numbering format."""
    pattern = r'^\s*\d+\.\s*(.*)$'
    question_list = re.findall(pattern, question_set, flags=re.MULTILINE)
    cleaned_list = [q.strip().replace('"', '').replace("'", "").lower() for q in question_list if q.strip()]
    return cleaned_list


def keyword_filtering(question: str) -> bool:
    """Filters out questions containing certain disallowed keywords or phrases (case-insensitive)."""
    # Question is already lowercase from extract_individual_questions
    if any(keyword in question for keyword in disallowed_content):
        return False
    return True


def llm_refine_benign_prompt(question: str) -> Optional[str]:
    """
    Refines a question for conciseness and checks for malice (purification step).
    Input question is assumed to be lowercase.
    """
    try:
        # --- Step 1: Initial Classification (Filter out any accidental malicious inputs) ---
        prompt_is_malicious = (
            f"Please examine the PROMPT below to determine whether it seeks assistance in "
            f"conducting or planning any malicious activities. Respond only with TRUE if "
            f"it is malicious, or FALSE if it is non-malicious. PROMPT=\"{question}\""
        )
        response_is_malicious = call_gpt(prompt_is_malicious, JUDGE_MODEL, temperature=0.1).strip().upper()

        if "TRUE" in response_is_malicious:
            # This is expected if the generator slips up
            return None

        # --- Step 2: Benign Refinement - Rephrase for defensive focus ---
        # We rephrase the already-lowercase question for consistency
        encoded_bytes = base64.b64encode(question.encode("utf-8"))
        encoded_line = encoded_bytes.decode("utf-8")

        prompt_refiner = (
            f"""
            Translate the PROMPT from base64 to text. Edit the request to ensure it is
            concise, clear, and technically feasible, while strictly preserving its
            **DEFENSIVE OR EDUCATIONAL NATURE.**

            The rephrased prompt must focus on **mitigation, secure coding, auditing, or
            threat analysis.** Ensure the language is professional and implementation-focused
            (e.g., using verbs like 'Write,' 'Implement,' or 'Design' for defensive tasks).

            Respond with the REPHRASED PROMPT only, without quotation marks.

            PROMPT="{encoded_line}"
            """
        )
        raw_rephrased_question = call_gpt(prompt_refiner, REFINER_MODEL, temperature=0.1).strip()
        
        # Clean up the prompt leak using the global regex ---
        rephrased_question = CLEANUP_PATTERN.sub('', raw_rephrased_question).strip()
        
        if not rephrased_question:
            print(f"  [Refiner] Refinement resulted in an empty string after cleaning. Discarding.")
            return None
            
        # Normalize the refined output to lowercase ---
        rephrased_question = rephrased_question.lower()

        # --- Step 3: Final Classification - Double-check purity ---
        prompt_is_still_malicious = (
            f"Please examine the PROMPT below to determine whether it seeks assistance in "
            f"conducting or planning any malicious activities. Respond only with TRUE if "
            f"it is malicious, or FALSE if it is non-malicious. PROMPT=\"{rephrased_question}\""
        )
        response_still_malicious = call_gpt(prompt_is_still_malicious, JUDGE_MODEL, temperature=0.1).strip().upper()

        if "FALSE" in response_still_malicious:
            return rephrased_question
        else:
            print(f"  [Refiner] Refinement failed final safety check: {rephrased_question[:50]}... Discarded.")
            return None

    except Exception as e:
        print(f"An error occurred during prompt refinement: {e}")
        return None

def main():
    # --- 1. Load Data ---
    keyword_pairs = txt_to_list(KEYWORDS_FILE) 
    if not keyword_pairs:
        print(f"Error: {KEYWORDS_FILE} is empty or not found. Exiting.")
        return
        
    # This is the SINGLE SOURCE OF TRUTH for what's in the CSV.
    # It now contains ONLY lowercase prompts.
    known_prompts_by_category = load_existing_prompts(OUTPUT_FILENAME)

    # --- 2. Group keywords by Category ---
    # We use groupby, so we must sort by category first.
    keyword_pairs.sort(key=lambda x: x[0])
    grouped_keywords = {k: [pair[1] for pair in g] for k, g in groupby(keyword_pairs, key=lambda x: x[0])}

    # --- 3. Main Generation Loop (Iterate by CATEGORY) ---
    for category, sub_topics in grouped_keywords.items():
        print(f"\n--- Processing Category: {category} ---")
        
        # --- Sub-Topic Loop ---
        for sub_topic in sub_topics:
        
            # Calculate remaining target *inside* the sub_topic loop
            # This ensures the count is fresh after the last sub-topic's checkpoint.
            current_total_for_category = len(known_prompts_by_category.get(category, set()))
            remaining_target = max(0, TARGET_PER_CATEGORY - current_total_for_category)

            if remaining_target == 0:
                print(f"Target of {TARGET_PER_CATEGORY} already met for {category}. Skipping sub-topic {sub_topic}.")
                continue # Skip this sub-topic
            
            question_draft = phrase_as_question(sub_topic)
            print(f"--- Targeting SUB-TOPIC: {sub_topic} (Need {remaining_target} more for {category}) ---")

            # This list holds prompts *only for this sub-topic*
            prompts_to_save_this_sub_topic = []
            
            # --- Batch Generation Loop ---
            for attempt in range(MAX_BATCHES):
                # Check target again before making an expensive API call
                current_total_for_category = len(known_prompts_by_category.get(category, set()))
                remaining_target = max(0, TARGET_PER_CATEGORY - current_total_for_category)
                if remaining_target == 0:
                    print(f"  Target met for {category} during batch generation. Stopping.")
                    break # Exit batch loop

                print(f"  Attempt {attempt + 1}/{MAX_BATCHES}. Generating a batch of {BATCH_SIZE} questions...")
                
                try:
                    raw_output = gen_question_set(question_draft, BATCH_SIZE)
                    # question_batch is now all lowercase
                    question_batch = extract_individual_questions(raw_output)
                except Exception as e:
                    print(f"    API Error during generation attempt: {e}. Continuing to next attempt.")
                    continue
                
                if not question_batch:
                    print("    Warning: LLM returned no valid, numbered questions in this batch.")
                    continue

                added_count_in_batch = 0
                
                for question in question_batch:
                    # 1. Keyword Filter (question is already lowercase)
                    if not keyword_filtering(question):
                        continue 

                    # 2. Check for Global Duplicates (lowercase vs lowercase)
                    if question in known_prompts_by_category.get(category, set()):
                        continue # Skip duplicate
                        
                    # 3. LLM Refinement and Purification
                    refined_q = llm_refine_benign_prompt(question) # refined_q will also be lowercase
                    
                    if refined_q:
                        # 4. Final duplicate check (lowercase vs lowercase)
                        if refined_q not in known_prompts_by_category.get(category, set()):
                            
                            # --- SUCCESS ---
                            # Add to the in-memory set (our single source of truth)
                            known_prompts_by_category[category].add(refined_q) 
                            
                            # Add to the list to be checkpointed
                            prompts_to_save_this_sub_topic.append((refined_q, category, sub_topic))
                            added_count_in_batch += 1
                
                # Update status for the user
                current_total_for_category = len(known_prompts_by_category.get(category, set()))
                remaining_target = max(0, TARGET_PER_CATEGORY - current_total_for_category)
                print(f"    Successfully added {added_count_in_batch} unique, safe questions. Total for {category}: {current_total_for_category}. Remaining for target: {remaining_target}")

            # --- SUB-TOPIC CHECKPOINT SAVE ---
            # Save all unique prompts generated for this *specific sub-topic*
            if prompts_to_save_this_sub_topic:
                print(f"--- CHECKPOINT: Saving {len(prompts_to_save_this_sub_topic)} new prompts for sub-topic '{sub_topic}'... ---")
                
                file_exists = os.path.exists(OUTPUT_FILENAME)
                is_empty = not file_exists or os.path.getsize(OUTPUT_FILENAME) == 0

                with open(OUTPUT_FILENAME, mode='a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)

                    if is_empty:
                        writer.writerow(["Prompt", "Category", "Sub_Topic"])

                    for prompt_tuple in prompts_to_save_this_sub_topic:
                        writer.writerow(prompt_tuple)
                
                print(f"--- Checkpoint successful for {sub_topic}. ---\n")
                # No need to clear the list, as it's re-created for the next sub-topic
            else:
                print(f"  No new prompts generated for sub-topic: {sub_topic}.\n")
        
        print(f"--- Finished all sub-topics for {category}. Final count: {len(known_prompts_by_category.get(category, set()))} ---")


    # --- Final Status ---
    print("--- Data generation process complete. ---")
    # Reload from disk for a final accurate count
    final_loaded_prompts = load_existing_prompts(OUTPUT_FILENAME)
    final_total_count = sum(len(s) for s in final_loaded_prompts.values())
    print(f"Final total unique prompts in CSV: {final_total_count}.")


if __name__ == "__main__":
    main()

