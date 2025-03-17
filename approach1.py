from llms.llm_interaction import GroqClient
import json
import ast  # Add this import for parsing dictionary strings

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

    # Example of converting LLM-generated dictionary text to Python dictionary
    dict_prompt = "Generate a dictionary with patient information including name, age, and diagnosis"
    dict_response = llm.generate(dict_prompt)
    print("\nOriginal LLM dictionary text:")
    print(dict_response)
    
    try:
        # Method 1: Using ast.literal_eval (safest approach)
        # This only works if the response is a valid Python literal (like a dictionary)
        dict_object = ast.literal_eval(dict_response.strip())
        print("\nConverted to Python dictionary using ast.literal_eval:")
        print(dict_object)
        print(f"Type: {type(dict_object)}")
        
        # You can now access dictionary elements
        if isinstance(dict_object, dict):
            print(f"\nAccessing values: Name = {dict_object.get('name', 'N/A')}")
    except (SyntaxError, ValueError) as e:
        print(f"\nError parsing with ast.literal_eval: {e}")
        
        # Method 2: Alternative approach - try json.loads if it's JSON formatted
        try:
            dict_object = json.loads(dict_response)
            print("\nConverted to Python dictionary using json.loads:")
            print(dict_object)
        except json.JSONDecodeError as e:
            print(f"\nError parsing with json.loads: {e}")
            print("\nYou may need to clean/format the text before parsing")


if __name__ == "__main__":
    main()
