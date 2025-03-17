from llms.llm_interaction import GroqClient
import json
import ast


def pattern_identification(symbols, regex):
    # FSM implementation
    pass

def main():
    with open('datasets/patient_records1.json', 'r') as file:
        records = json.load(file)

    llm = GroqClient(model="llama3-8b-8192")

    for record in records[0:2]:
        regex = record['s_regex']
        record_text = record['record']
        label  = record['match']
        prompt = f"Given the following patient record, extract the following semantic symbols if they exist: {regex}. Return a machine parseable dictionary with <symbol>: extracted text pairs. Only return the dictionary, nothing else.\n\nPatient Record: {record_text}"
        response = llm.generate(prompt)
        response_dict = ast.literal_eval(response)
        print(response_dict)
        pattern_identification(response_dict, regex)
    
     

if __name__ == "__main__":
    main()