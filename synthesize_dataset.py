import json

from llms.llm_interaction import OpenAIClient, GroqClient, AnthropicClient

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
    llm = GroqClient(model="llama-3.3-70b-versatile", temperature=2)
    s_regexes = load_s_regexes('datasets/s_regex.json')
    s_regex = s_regexes[16]
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
    s_regexes = load_s_regexes('datasets/s_regex2.json')
    records = []
    llm = GroqClient(model="llama-3.3-70b-versatile", temperature=2)
    for s_regex in s_regexes:
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

        positive_generated_text = llm.generate(prompt=prompt_positive, max_tokens=700)
        negative_generated_text = llm.generate(prompt=prompt_negative, max_tokens=700)

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



if __name__ == "__main__":
    # generate_initial_dataset('datasets/patient_records2.json')
    # test_generations()
    # test_negative_generations()
    # testing_for_variability()
    generate_dataset_2('datasets/patient_records2.json')