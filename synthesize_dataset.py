import os
from openai import OpenAI
import json
from dotenv import load_dotenv


load_dotenv()


client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def load_s_regexes(json_file):
    with open(json_file, 'r') as file:
        return json.load(file).get('patterns', [])

def generate_text(prompt, max_tokens=300):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

# def main():
#     prompt = "Create a realistic patient record with sections [HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, ASSESSMENT] describing a patient visit. Do not include a chief complaint, markdown formatting, or bullet points. Make it around 150 words.  Most important: I want it to contain the following semantic regex pattern: <medication><patient><ineffectiveness>"
#     generated_text = generate_text(prompt)
#     print(generated_text)

def main2(output_file):
    s_regexes = load_s_regexes('s_regex.json')
    base_prompt = "Create a realistic patient record with sections [HISTORY OF PRESENT ILLNESS, PAST MEDICAL HISTORY, ASSESSMENT] describing a patient visit. Do not include a chief complaint, markdown formatting, or bullet points. Make it around 150 words.  Most important: I want it to contain the following semantic regex pattern: "
    records = []
    for s_regex in s_regexes:
        full_prompt = base_prompt + s_regex
        generated_text = generate_text(full_prompt)
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
    # main2('patient_records1.json')
