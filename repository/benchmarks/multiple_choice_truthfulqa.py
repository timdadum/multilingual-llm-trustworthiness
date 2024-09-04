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
            
            # Ensure that there are at least four incorrect answers
            if len(incorrect_answers) < 3:
                continue
            
            # Ensure that the best answer is in the correct answers
            if best_answer not in correct_answers:
                correct_answers.append(best_answer)
            
            # Select four incorrect answers
            selected_incorrect_answers = random.sample(incorrect_answers, 3)
            
            # Combine the best answer with the selected incorrect answers
            all_answers = [best_answer] + selected_incorrect_answers
            
            # Shuffle the answers
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
csv_file_path = r'multilingual-llm-trustworthiness\repository\benchmarks\TruthfulQA.csv'
json_data = convert_csv_to_json(csv_file_path)

# Write to JSON file
with open(r'multilingual-llm-trustworthiness\repository\benchmarks\clean\truthfulqa_mc.json', 'w') as json_file:
    json.dump(json_data, json_file, indent=2)

print("Conversion complete. Data saved to output.json")
