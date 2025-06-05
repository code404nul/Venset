import csv
import json

input_file = 'dataset.csv'
output_file = 'dataset_alpaca.jsonl'

with open(input_file, 'r', encoding='utf-8') as csvfile, open(output_file, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        emotion_in = row.get('emmotion_in', '').strip()
        user_input = row.get('input', '').strip()
        context = row.get('context', '').strip()
        output = row.get('output', '').strip()

        instruction = f"The user feels an emotion of '{emotion_in}' and said : \"{user_input}\""
        input_text = (f"\nContext : {context}" if context else "")
        
        prompt = {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
        jsonlfile.write(json.dumps(prompt, ensure_ascii=False) + '\n')
