from llms.llm_interaction import GroqClient
import json
import ast
import re

from transitions import Machine

class PatternFSM:
    def __init__(self, pattern):
        self.pattern = pattern
        self.states = ["START", "MATCHING", "ACCEPTED", "REJECTED"]
        self.machine = Machine(model=self, states=self.states, initial="START")

        # Define transitions
        self.machine.add_transition(trigger="start_matching", source="START", dest="MATCHING")
        self.machine.add_transition(trigger="accept", source="MATCHING", dest="ACCEPTED")
        self.machine.add_transition(trigger="reject", source="MATCHING", dest="REJECTED")

    def match(self, symbols):
        """Simulate FSM processing extracted symbols."""
        self.start_matching()
        input_string = ' '.join(symbols.values())
        
        if self.pattern.fullmatch(input_string):  # Using regex internally
            self.accept()
            print("Pattern Matched!")
        else:
            self.reject()
            print("Pattern Not Matched!")
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