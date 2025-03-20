from llms.llm_interaction import GroqClient
import json
import ast
import re

from transitions import Machine

class PatternFSM:
    def __init__(self, pattern):
        self.pattern = pattern
        self.states = ["START", "EXTRACTING", "ACCEPTED", "REJECTED"]
        self.machine = Machine(model=self, states=self.states, initial="START")

        # Define transitions
        self.machine.add_transition(trigger="start_extraction", source="START", dest="EXTRACTING")
        self.machine.add_transition(trigger="accept", source="EXTRACTING", dest="ACCEPTED")
        self.machine.add_transition(trigger="reject", source="EXTRACTING", dest="REJECTED")

    def match(self, record_text):
        # Move to extracting state
        self.start_extraction()  

        # Extract text values
        extracted_symbols = self.extract_symbols(record_text)  

        if extracted_symbols:
            # If we found all required symbols, accept
            self.accept()  
            print("Extracted Symbols:", extracted_symbols)
            return extracted_symbols
        else:
            # If symbols are missing, reject
            self.reject()  
            print("No valid symbols found.")
            return None
    def extract_symbols(self, record_text):
        return None

def pattern_identification(symbols, regex):
    fsm = PatternFSM(re.compile(regex))
    fsm.match(symbols)

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