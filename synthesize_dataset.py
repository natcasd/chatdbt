import json

from llms.llm_interaction import OpenAIClient

def load_s_regexes(json_file):
    with open(json_file, 'r') as file:
        return json.load(file).get('patterns', [])

def main(output_file):
    llm = OpenAIClient(model="gpt-4o", temperature=0.7)
    
    s_regexes = load_s_regexes('s_regex.json')
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


if __name__ == "__main__":
    main('datasets/patient_records2.json')
