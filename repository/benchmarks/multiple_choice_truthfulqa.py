import csv
import json
import random

def convert_csv_to_json(csv_file):
    data = []
    with open(csv_file, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            question = row['Question']
            best_answer = row['Best Answer']
            correct_answers = row['Correct Answers'].split('; ')
            incorrect_answers = row['Incorrect Answers'].split('; ')
            
            all_answers = correct_answers + incorrect_answers
            random.shuffle(all_answers)
            
            # Ensure only five options
            all_answers = all_answers[:4] if len(all_answers) > 4 else all_answers
            
            if best_answer not in all_answers:
                all_answers.append(best_answer)
            
            random.shuffle(all_answers)
            
            # Map answers to letters
            answer_mapping = {chr(65 + i): ans for i, ans in enumerate(all_answers)}
            correct_letter = next(key for key, value in answer_mapping.items() if value == best_answer)
            
            multiple_choice_question = f"{question}\n"
            for key, value in answer_mapping.items():
                multiple_choice_question += f"{key}. {value}\n"
            
            data.append({
                "question_en": multiple_choice_question.strip(),
                "answer_en": correct_letter
            })
    
    return data

# Convert CSV to JSON
csv_file_path = r'multilingual-llm-trustworthiness\repository\benchmarks\TruthfulQA.csv'  # Update this to your CSV file path
json_data = convert_csv_to_json(csv_file_path)

# Write to JSON file
with open(r'multilingual-llm-trustworthiness\repository\benchmarks\truthfulqa_mc.csv', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print("Conversion complete. Data saved to output.json")
