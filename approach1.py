from llms.llm_interaction import GroqClient
import json

def main():
    with open('datasets/patient_records1.json', 'r') as file:
        records = json.load(file)

    llm = GroqClient(model="llama3-8b-8192")
    acc = 0
    for record in records:
        regex = record['s_regex']
        record_text = record['record']
        label  = record['match']
        prompt = f"Given the following patient record, identify if it matches the following semantic regex pattern: {regex} Return either true OR false, nothing else.\n\nPatient Record: {record_text}"
        response = llm.generate(prompt)
        response_bool = response.strip().lower() == "true"
        acc += 1 if label == response_bool else 0
      
    print(f'accuracy of model over {len(records)} generated records: {acc/len(records)}')


if __name__ == "__main__":
    main()
