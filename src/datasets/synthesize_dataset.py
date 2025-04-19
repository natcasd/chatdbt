import json
import random
from llms.llm_interaction import OpenAIClient, GroqClient, AnthropicClient
from tqdm import tqdm

def load_s_regexes(json_file):
    with open(json_file, 'r') as file:
        return json.load(file).get('patterns', [])

def generate_initial_dataset(output_file):
    llm = OpenAIClient(model="gpt-4o", temperature=0.7)
    
    s_regexes = load_s_regexes('datasets/s_regex.json')
    base_prompt = "Create a realistic patient record with sections [HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, ASSESSMENT] describing a patient visit. Do not include a chief complaint, markdown formatting, or bullet points. Make it around 150 words.  Most important: I want it to contain the following semantic regex pattern: "
    records = []
    
    for s_regex in s_regexes:
        full_prompt = base_prompt + s_regex
        generated_text = llm.generate(prompt=full_prompt, max_tokens=300)
        record = {
            "s_regex": s_regex,
            "record": generated_text,
            "match": True
        }
        records.append(record)
        print(f"Generated record for pattern: {s_regex}")
    
    with open(output_file, 'w') as file:
        json.dump(records, file, indent=2)
    print(f"Dataset generation complete. Saved {len(records)} records to {output_file}")

def test_generations():
    s_regexes = load_s_regexes('datasets/s_regex.json')
    openai_client = OpenAIClient(model="gpt-4o")
    
    # List of Llama models to test
    llama_models = [
        "llama-3.2-11b-vision-preview",
        "llama3-70b-8192",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "llama-guard-3-8b",
        "llama-3.2-3b-preview",
        "llama-3.2-90b-vision-preview",
        "llama-3.2-1b-preview",
        "llama-3.3-70b-specdec",
        "llama3-8b-8192"
    ]
    
    s_regex = s_regexes[4]
    print(f"Semantic regex pattern: {s_regex}\n\n")
    # base_prompt = "Create a realistic patient record with sections [HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, ASSESSMENT] describing a patient visit. Do not include a chief complaint, markdown formatting, or bullet points. Make it around 150 words. Make the response not contain anything else besides the record.  Most important: I want it to contain the following semantic regex pattern: "
    prompt = f"""
    Generate a realistic clinical patient record summarizing a hospital encounter. Structure the record using the following clearly labeled sections: ADMISSION DIAGNOSIS, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, SOCIAL HISTORY, PHYSICAL EXAMINATION, LABORATORY DATA, and HOSPITAL COURSE.

    Do not use markdown, bullets, numbered lists, or PHI tags. Write in a semi-structured, natural clinical narrative approximately 300–500 words long. Ensure variability and realism in the patient's demographics, medical conditions, medications, diagnostic results, treatments, clinical progression, and outcomes.

    Critically, include the following semantic pattern naturally integrated into the narrative, ensuring clinical coherence and plausibility:  
    {s_regex}

    Replace placeholders within the semantic pattern with realistic, clinically appropriate details.
    """
    
    # Test each Llama model
    for model in llama_models:
        try:
            print(f"\n{'=' * 50}")
            print(f"Testing model: {model}")
            print(f"{'=' * 50}\n")
            llm = GroqClient(model=model)
            generated_text = llm.generate(prompt=prompt, max_tokens=700)
            print(f"Generated text:\n\n{generated_text}\n")
        except Exception as e:
            print(f"Error with model {model}: {str(e)}\n")
    
    # Also test GPT-4o for comparison
    print(f"\n{'=' * 50}")
    print("Testing model: GPT-4o (OpenAI)")
    print(f"{'=' * 50}\n")
    generated_text = openai_client.generate(prompt=prompt, max_tokens=700)
    print(f"Generated text:\n\n{generated_text}\n")

def test_negative_generations():
    llm = GroqClient(model="llama-3.3-70b-versatile")
    s_regexes = load_s_regexes('datasets/s_regex.json')
    s_regex = s_regexes[16]
    print(f"Semantic regex pattern: {s_regex}\n\n")
    prompt = f"""
    Generate a realistic clinical patient record summarizing a hospital encounter. Structure the record using the following clearly labeled sections: ADMISSION DIAGNOSIS, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, SOCIAL HISTORY, PHYSICAL EXAMINATION, LABORATORY DATA, and HOSPITAL COURSE.

    Do not use markdown, bullets, numbered lists, or PHI tags. Write in a semi-structured, natural clinical narrative approximately 300–500 words long. Ensure variability and realism in the patient's demographics, medical conditions, medications, diagnostic results, treatments, clinical progression, and outcomes.

    Critically, the narrative must **NOT** include the complete semantic pattern specified below:
    {s_regex}

    However, it **should include some (but not all)** of the individual semantic components within the specified pattern, placed naturally and realistically throughout the text. Carefully ensure that the complete pattern as given never appears in full, maintaining clinical coherence and plausibility.
    """

    generated_text = llm.generate(prompt=prompt, max_tokens=700)
    print(f"Generated text:\n\n{generated_text}\n")

def testing_for_variability():
    # llm = GroqClient(model="llama-3.3-70b-versatile", temperature=2)
    llm = OpenAIClient(model="gpt-4o", temperature=1)
    s_regexes = load_s_regexes('s_regex3.json')
    s_regex = s_regexes[0]
    print(f"Semantic regex pattern: {s_regex}\n\n")
    prompt_positive = f"""
    Generate a realistic clinical patient record summarizing a hospital encounter. Structure the record using these labeled sections: ADMISSION DIAGNOSIS, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, SOCIAL HISTORY, PHYSICAL EXAMINATION, LABORATORY DATA, and HOSPITAL COURSE.

    Do not use markdown, bullets, numbered lists, or PHI tags. Write in a semi-structured, natural clinical narrative approximately 300–500 words long.

    Ensure significant variability and realism in patient demographics (age, occupation), lifestyle factors, and clinical scenarios. Generate records for patients across different age groups, explicitly avoiding repetitive scenarios. Include diverse medical histories, medication usage, diagnostic tests, and treatment outcomes.

    Integrate naturally into the narrative the following semantic pattern, ensuring clinical plausibility:
    {s_regex}

    Replace placeholders within the semantic pattern realistically and contextually. Ensure each generated example is distinct and clinically coherent.
    """

    prompt_negative = f"""
    Generate a realistic clinical patient record summarizing a hospital encounter. Structure the record using these labeled sections: ADMISSION DIAGNOSIS, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, SOCIAL HISTORY, PHYSICAL EXAMINATION, LABORATORY DATA, and HOSPITAL COURSE.

    Do not use markdown, bullets, numbered lists, or PHI tags. Write in a semi-structured, natural clinical narrative approximately 300–500 words long.

    Ensure significant variability and realism in patient demographics (age, occupation), lifestyle factors, and clinical scenarios. Generate records for patients across different age groups, explicitly avoiding repetitive scenarios. Include diverse medical histories, medication usage, diagnostic tests, and treatment outcomes.

    The narrative must **NOT** include the complete semantic pattern below:
    {s_regex}

    However, it **should include some (but not all)** individual semantic components from the specified pattern, placed naturally and realistically throughout the text. Ensure the complete pattern never fully appears.
    """

    for i in range(4):
        print(f"Positive generation {i+1}:")
        print(llm.generate(prompt=prompt_positive, max_tokens=700))
        print("\n")
        print(f"Negative generation {i+1}:")
        print(llm.generate(prompt=prompt_negative, max_tokens=700))
        print("\n")


def generate_dataset_2(output_file):
    s_regexes = load_s_regexes('s_regex3.json')
    records = []
    llm = OpenAIClient(model="gpt-4o", temperature=1)
    for s_regex in tqdm(s_regexes, desc="Generating records"):
        prompt_positive = f"""
        Generate a realistic clinical patient record summarizing a hospital encounter. Structure the record using these labeled sections: ADMISSION DIAGNOSIS, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, SOCIAL HISTORY, PHYSICAL EXAMINATION, LABORATORY DATA, and HOSPITAL COURSE.

        Do not use markdown, bullets, numbered lists, or PHI tags. Write in a semi-structured, natural clinical narrative approximately 300–500 words long.

        Ensure significant variability and realism in patient demographics (age, occupation), lifestyle factors, and clinical scenarios. Generate records for patients across different age groups, explicitly avoiding repetitive scenarios. Include diverse medical histories, medication usage, diagnostic tests, and treatment outcomes.

        Integrate naturally into the narrative the following semantic pattern, ensuring clinical plausibility:
        {s_regex}

        Replace placeholders within the semantic pattern realistically and contextually. Ensure each generated example is distinct and clinically coherent.
        """

        prompt_negative = f"""
        Generate a realistic clinical patient record summarizing a hospital encounter. Structure the record using these labeled sections: ADMISSION DIAGNOSIS, HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, SOCIAL HISTORY, PHYSICAL EXAMINATION, LABORATORY DATA, and HOSPITAL COURSE.

        Do not use markdown, bullets, numbered lists, or PHI tags. Write in a semi-structured, natural clinical narrative approximately 300–500 words long.

        Ensure significant variability and realism in patient demographics (age, occupation), lifestyle factors, and clinical scenarios. Generate records for patients across different age groups, explicitly avoiding repetitive scenarios. Include diverse medical histories, medication usage, diagnostic tests, and treatment outcomes.

        The narrative must **NOT** include the complete semantic pattern below:
        {s_regex}

        However, it **should include some (but not all)** individual semantic components from the specified pattern, placed naturally and realistically throughout the text. Ensure the complete pattern never fully appears.
        """

        positive_generated_text = llm.generate(prompt=prompt_positive, max_tokens=1000)
        negative_generated_text = llm.generate(prompt=prompt_negative, max_tokens=1000)

        record1 = {
            "s_regex": s_regex,
            "record": positive_generated_text,
            "match": True
        }
        records.append(record1)
        record2 = {
            "s_regex": s_regex,
            "record": negative_generated_text,
            "match": False
        }
        records.append(record2)
        print(f"Generated positive and negative records for pattern: {s_regex}")
    
    with open(output_file, 'w') as file:
        json.dump(records, file, indent=2)
    print(f"Dataset generation complete. Saved {len(records)} records to {output_file}")

def generate_nl_query(input_filepath, output_filepath):
    """
    Generate a natural language query for each semantic regex pattern in a pre-generated dataset. Save to new file specified in argument.
    """
    with open(input_filepath, 'r') as f:
        old_data = json.load(f)
    llm = OpenAIClient(model="gpt-4o", temperature=1)
    for record in old_data:
        s_regex = record['s_regex']
        prompt = f"""
        Generate a natural language query for the following semantic regex pattern:
        {s_regex}
        For example, if the semantic regex pattern is "<diagnostic_test><patient><misinterpretation>", the natural language query could be "Is there an occurence of the patient having a misinterpretation of their diagnostic test results?"
        Important: Only return the natural language query, nothing else.
        """
        nl_query = llm.generate(prompt=prompt, max_tokens=100)
        print(nl_query)
        record['nl_query'] = nl_query
    with open(output_filepath, 'w') as f:
        json.dump(old_data, f, indent=2)
    
def generate_semantic_symbol_vocab():
    llm = OpenAIClient(model="gpt-4o", temperature=1)

    generated_responses = []

    for _ in range(60):
        system_prompt = f"""
        You are an expert medical semantic regex generator.
        """

        user_prompt = f"""
        A semantic regex is a sequence of 3–5 semantic symbols representing a meaningful clinical progression in a patient's record. Each symbol is wrapped in angle brackets (e.g., <TREAT:INSULIN>).

        Do **not** always begin with a diagnosis. Vary the **entry point** of the regex — e.g., start with:
        - symptom onset (<ONSET_CHEST_PAIN>)
        - emergency events (<ER_VISIT>)
        - procedures (<SURGERY>)
        - follow-ups (<FOLLOWUP_VISIT>)
        - missed or delayed care (<MISSED_APPT>)
        - chronic monitoring (<LAB_MONITORING>)

        Vary the structure of patient journeys. Include different episode types:
        - acute incidents (e.g., stroke, MI)
        - chronic management (e.g., diabetes, bipolar)
        - interrupted care (e.g., relapse, nonadherence)
        - procedural recovery (e.g., post-op follow-up)
        - multimodal care (e.g., chemo + radiation)

        Use a mix of structured tags (e.g., <DIAG:HTN>, <TREAT:STATINS>) and descriptive tags (e.g., <HospitalAdmission>, <MissedFollowup>).

        Avoid administrative tags like <PatientID>, <DOB>, <Vitals>.

        Instruction: Generate a **realistic and narratively diverse** semantic regex with 3–5 symbols. Do not always begin with diagnosis. Avoid repetition.

        Important: Only return the semantic regex. No markdown, no explanation.
        """

        response = llm.generate(system_prompt=system_prompt, prompt=user_prompt, max_tokens=100)
        print(response)
        generated_responses.append(response)

    with open('s_regex3.json', 'w') as f:
        json.dump(generated_responses, f, indent=2)
    

def combine_datasets(input_file1, input_file2, output_file):
    with open(input_file1, 'r') as f:
        data1 = json.load(f)
    with open(input_file2, 'r') as f:
        data2 = json.load(f)
    combined_data = data1 + data2

    # shuffle the combined data
    random.shuffle(combined_data)

    with open(output_file, 'w') as f:
        json.dump(combined_data, f, indent=2)

if __name__ == "__main__":
    # generate_initial_dataset('datasets/patient_records2.json')
    # test_generations()
    # test_negative_generations()
    # testing_for_variability()
    # generate_dataset_2('patient_records3.json')
    # generate_nl_query('patient_records3.json', 'patient_records3_nl_query.json')
    # generate_semantic_symbol_vocab()
    combine_datasets('patient_records2_nl_query.json', 'patient_records3_nl_query.json', 'patient_records4_nl_query.json')