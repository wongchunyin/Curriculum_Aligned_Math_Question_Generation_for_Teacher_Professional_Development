from flask import Flask, request, jsonify
from flask_cors import CORS # To handle Cross-Origin Resource Sharing
import sys
from pathlib import Path

# Get the absolute path to the code directory
code_dir = Path(__file__).parent.parent.parent
# Add it to Python path
sys.path.append(str(code_dir))


from trainer import LanguageModelTrainer
from utils import get_device
from math_question_generator import prepare_prompt, prepare_generator, generate, generate_multiple
from transformers import GenerationConfig
import warnings

app = Flask(__name__)
CORS(app, 
     origins=['http://localhost:3000', 'http://127.0.0.1:3000'],
     methods=['GET', 'POST', 'OPTIONS'],
     allow_headers=['Content-Type', 'Authorization'],
     supports_credentials=True) # Enable CORS for frontend

def load_llm():
    """
    Simulates loading a pre-trained LLM.
    In a real scenario, this would load an actual model.
    """
    print("Loading pre-trained LLM model (placeholder)...")
    path = Path("../../phi2_final_model/model").resolve()
    trainer = LanguageModelTrainer.load_model(
        device=get_device(),
        path=str(path),
    )
    return trainer.model, trainer.tokenizer

model, tokenizer = load_llm() # Load the "LLM" when the app starts

def parse_question_answer(raw_output):
    """
    Parse raw model output to extract question and answer, removing [ES] and [AS] prefixes
    """
    import re
    
    # Extract first sentence before any [ES] or [AS] as the question
    first_bracket_match = re.search(r'\[ES\]|\[AS\]', raw_output)
    if first_bracket_match:
        question_part = raw_output[:first_bracket_match.start()].strip()
        # Remove any "answer:" text from question part
        question_part = re.sub(r'\banswer\s*:\s*$', '', question_part, flags=re.IGNORECASE).strip()
        explanation_start = first_bracket_match.start()
    else:
        question_part = raw_output.strip()
        explanation_start = 0
    
    # Look for "Final answer:" pattern
    final_answer_match = re.search(r'Final answer\s*:\s*(.+?)$', raw_output, re.IGNORECASE | re.MULTILINE)
    
    if final_answer_match:
        # Extract explanation between brackets and final answer
        explanation_text = raw_output[explanation_start:final_answer_match.start()]
        explanation_text = re.sub(r'\[ES\]|\[AS\]', '', explanation_text).strip()
        
        final_answer = final_answer_match.group(1).strip()
        
        # Combine explanation and final answer
        answer_part = f"{explanation_text}\nFinal answer: {final_answer}"
        
        return {
            'question': question_part,
            'answer': answer_part
        }
    
    return {
        'question': question_part,
        'answer': ''
    }

def clean_tutor_prefix(text):
    """
    Remove "Tutor:" prefix from the text, especially after <|question_end|> markers
    """
    import re
    
    # Remove "Tutor:" at the start of lines (case insensitive)
    pattern = r'^\s*Tutor\s*:\s*'
    text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    # Remove "Tutor:" that appears anywhere in the text (not just at line start)
    pattern = r'\bTutor\s*:\s*'
    text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    return text.strip()

def clean_operation_prefixes(text):
    """
    Remove operation prefixes like "Addition:", "Subtraction:", etc. from the text
    """
    import re
    
    # Remove operation prefixes at the start of lines
    operations = ['Addition', 'Subtraction', 'Multiplication', 'Division']
    for operation in operations:
        # Remove "Operation:" at the start of lines (case insensitive)
        pattern = rf'^\s*{operation}\s*:\s*'
        text = re.sub(pattern, '', text, flags=re.MULTILINE | re.IGNORECASE)
    
    return text.strip()

def prepare_custom_prompt(year=4, transcript=""):
    """
    Use the exact same prompt format as math_question_generator.py
    """
    task = f"generate mathematical question with answer for year {year} student"
    prompt = (
        f"{task}"
        f"\n{transcript}"
    )
    return prompt

def generate_random_student_levels():
    """
    Same function as in math_question_generator.py
    """
    import random
    aspects = ["Addition", "Subtraction", "Multiplication", "Division"]
    levels = ["Strong", "Normal", "Weak"]
    
    transcript = {aspect: random.choice(levels) for aspect in aspects}
    
    if "Weak" not in transcript.values():
        random_aspect = random.choice(aspects)
        transcript[random_aspect] = "Weak"
    
    txt = "\n".join([f"{k}:{v}" for k, v in transcript.items()])
    return txt

def generate_llm_response(model, tokenizer, transcript, year, num_questions=5):
    """
    Generate questions using the exact same approach as math_question_generator
    """
    import torch
    import random
    import time
    
    # Use current time for randomness
    base_seed = int(time.time() * 1000) % 10000
    torch.manual_seed(base_seed)
    random.seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)
    
    questions_data = []
    
    prompt = prepare_custom_prompt(year, transcript)
    print(f"3. FINAL PROMPT SENT TO MODEL:")
    print(f"   {repr(prompt)}")
    print()
    
    for i in range(num_questions):
        # Use different seed for each question
        torch.manual_seed(base_seed + i * 1000 + random.randint(1, 999))
        
        gen_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_new_tokens': 256,
            'repetition_penalty': 1.1,
            'do_sample': True,
            'pad_token_id': tokenizer.eos_token_id,
        }
        
        single_response = generate(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            gen_config=gen_config
        )
        
        print(f"   RAW MODEL OUTPUT: {repr(single_response)}")
        
        # Clean basic formatting
        cleaned_response = single_response.lstrip(': ').strip()
        cleaned_response = cleaned_response.replace('<|endofgeneration|>', '').strip()
        cleaned_response = cleaned_response.replace('<|question_end|>', '').strip()
        cleaned_response = clean_operation_prefixes(cleaned_response)
        cleaned_response = clean_tutor_prefix(cleaned_response)
        cleaned_response = cleaned_response.replace('(needs to use estimation)', '').strip()
        
        # Parse question and answer, removing [ES] and [AS] prefixes
        parsed_data = parse_question_answer(cleaned_response)
        
        print(f"   PARSED DATA: {repr(parsed_data)}")
        
        # Skip if output is too short
        if len(parsed_data['question']) < 10:
            print(f"   SKIPPING: Question too short")
            continue
        
        questions_data.append(parsed_data)
    
    print(f"4. GENERATED QUESTIONS DATA:")
    print(f"   Total Questions Generated: {len(questions_data)}")
    for i, q_data in enumerate(questions_data, 1):
        print(f"   Question {i}: {q_data['question'][:50]}...")
        print(f"   Answer {i}: {q_data['answer']}")
    
    return questions_data, prompt

@app.route('/assess', methods=['POST'])
def assess_student():
    data = request.json
    print("\n" + "="*80)
    print("BACKEND REQUEST PROCESSING")
    print("="*80)
    print(f"1. RAW INPUT FROM FRONTEND:")
    print(f"   {data}")
    print()

    transcript = data.get('transcript', {})
    scoreLabels = data.get('scoreLabels', {})
    passing_requirement = data.get('passingRequirement', 50)
    year = data.get('year', 4)
    num_questions = data.get('numQuestions', 5)

    addition_score = transcript.get('addition', 0)
    subtraction_score = transcript.get('subtraction', 0)
    multiplication_score = transcript.get('multiplication', 0)
    division_score = transcript.get('division', 0)

    lbl_addition = scoreLabels.get('addition',0)
    lbl_subtraction = scoreLabels.get('subtraction', 0)
    lbl_multiplication = scoreLabels.get('multiplication', 0)
    lbl_division = scoreLabels.get('division', 0)
    
    # Determine pass/fail for each aspect
    addition_status = "Pass" if addition_score >= passing_requirement else "Fail"
    subtraction_status = "Pass" if subtraction_score >= passing_requirement else "Fail"
    multiplication_status = "Pass" if multiplication_score >= passing_requirement else "Fail"
    division_status = "Pass" if division_score >= passing_requirement else "Fail"

    transcript = (
        f"transcript:\n"
        f"Addition:{lbl_addition}\n"
        f"Subtraction:{lbl_subtraction}\n"
        f"Multiplication:{lbl_multiplication}\n"
        f"Division:{lbl_division}"
    )
    
    print(f"2. PROCESSED TRANSCRIPT:")
    print(f"   {transcript}")
    print()
    
    # Generate questions using LLM
    questions_data, used_prompt = generate_llm_response(
        model=model, 
        tokenizer=tokenizer, 
        transcript=transcript, 
        year=year, 
        num_questions=num_questions
    )
    
    return jsonify({
        'status': 'success',
        'questions': questions_data,
        'transcript': transcript,
        'year': year
    })

# Add CORS support for Colab
from flask_cors import CORS
CORS(app)

if __name__ == '__main__':
    app.run(debug=True) # Run the Flask app in debug mode