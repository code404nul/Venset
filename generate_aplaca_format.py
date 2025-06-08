import csv
import json
from emotion import emotion_classify

input_file = 'dataset.csv'
output_file = 'dataset_alpaca.jsonl'

with open(input_file, 'r', encoding='utf-8') as csvfile, open(output_file, 'w', encoding='utf-8') as jsonlfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        user_input = row.get('input', '').strip()
        context = row.get('context', '').strip()
        output = row.get('output', '').strip()


        max_emotion = max([e for e in emotion_classify(user_input)], key=lambda x: x['score'])
        instruction = f"The user feels an emotion of '{max_emotion["label"]}' and said : \"{user_input}\""
        input_text = (f"\nContext : {context}" if context else "")
        
        prompt = {
            "instruction": instruction,
            "input": input_text,
            "output": output
        }
        jsonlfile.write(json.dumps(prompt, ensure_ascii=False) + '\n')
