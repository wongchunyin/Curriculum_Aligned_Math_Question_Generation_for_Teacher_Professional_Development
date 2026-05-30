import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Tokenizer, GPT2Model, GPT2LMHeadModel, pipeline, GenerationConfig
from utils import get_device
import re
import random
import warnings
import numpy as np
import json
from typing import Union, List
from PHI2_Utils import PHI2_Utils

class MathQuestionGenerator:
    def __init__(self, model_name="microsoft/phi-2"):
        """
        Initialize the MathQuestionGenerator with Phi-2 model
        """
        # utils = utils()
        self.device = get_device()
        
        # Load model and tokenizer
        if re.search("gpt", model_name) : # string contains "gpt"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(   
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True
            )
        
        self.model.to(self.device) 
        # Set pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    
def prepare_generator(model, tokenizer):
        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=get_device()
        )


def generate(prompt, model, gen_config=None, tokenizer=None):

        default_gen_config = GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                # num_beams=5,
                max_new_tokens=256,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 2. Merge custom config if provided
        if gen_config is not None:
            if isinstance(gen_config, dict):
                gen_config = GenerationConfig(**gen_config)
            
            # Convert both to dicts and merge carefully
            default_dict = default_gen_config.to_dict()
            custom_dict = gen_config.to_dict()
            
            # Remove None values from custom config to avoid overriding with None
            custom_dict = {k: v for k, v in custom_dict.items() if v is not None}
            
            # Merge dictionaries (custom config overrides defaults)
            merged_dict = {**default_dict, **custom_dict}
            
            # Create final config
            final_config = GenerationConfig(**merged_dict)
        else:
            final_config = default_gen_config


        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.shape[1]  # Length of the prompt in tokens

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            outputs = model.generate(
                **inputs,
                generation_config=final_config,
            )

        return tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
             )


def load_embeddings_by_year():
    """
    Load and organize embeddings by year for efficient access.
    
    Returns:
        Dict with year as key and single scope embedding as value
    """
    import json
    with open("./curriculum_Desc/embedded_desc.json", "r") as f:
        embedded_desc = json.load(f)
    
    embeddings_by_year = {}
    for desc in embedded_desc:
        desc_year = desc['year'].split()[-1] if 'year' in desc else None
        if desc_year:
            year_int = int(desc_year)
            embeddings_by_year[year_int] = np.array(desc['scope'])
    
    return embeddings_by_year


def generate_multiple(
    prompt: str,
    model,
    tokenizer,
    num_variants: int = 3,
    gen_config: Union[GenerationConfig, dict, None] = None,
    batch_size: int = None,
    year: int = None,
    scope_embedding = None,
    max_attempts: int = None
) -> List[dict]:
    """
    Generates multiple distinct responses for a single prompt with similarity filtering.
    
    Args:
        prompt: Input prompt (str)
        num_variants: Number of different responses to generate
        batch_size: Number of parallel generations (None for auto)
        year: Year level to match scope descriptions
        scope_embedding: Pre-loaded scope embedding for the year
    
    Returns:
        List of dicts with 'question' and 'similarity' keys
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize PHI2_Utils
    phi2_utils = PHI2_Utils()
    
    # Use provided embedding or load it
    if scope_embedding is None:
        embeddings_by_year = load_embeddings_by_year()
        scope_embedding = embeddings_by_year.get(year, None)
    
    # 1. Configure generation for diversity
    default_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        max_new_tokens=256,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Merge custom config
    final_config = default_config
    if gen_config:
        if isinstance(gen_config, dict):
            gen_config = GenerationConfig(**gen_config)
        final_config = GenerationConfig(
            **{**default_config.to_dict(), 
               **gen_config.to_dict()}
        )
    
    # 2. Optimized batch generation
    if batch_size is None:
        batch_size = min(4, num_variants)
        
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)
    
    # 3. Generate with similarity filtering
    valid_outputs = []
    if max_attempts is None:
        max_attempts = num_variants * 10
    attempts = 0
    
    print(f"Found scope embedding for year {year}: {scope_embedding is not None}")
    
    while len(valid_outputs) < num_variants and attempts < max_attempts:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=final_config,
                num_return_sequences=batch_size
            )
        
        decoded = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # Check similarity for each generated text
        for text in decoded:
            if text not in [output['question'] for output in valid_outputs]:
                if scope_embedding is not None:
                    text_embedding = phi2_utils.embed(text)
                    similarity = phi2_utils.cosine_similarity(scope_embedding, text_embedding)
                    print(f"Similarity: {similarity:.3f}")
                    if similarity > 0.7:
                        valid_outputs.append({
                            'question': text,
                            'similarity': similarity
                        })
                        print(f"✓ Added text with similarity {similarity:.3f}")
                    else:
                        print(f"✗ Text rejected - similarity {similarity:.3f} < 0.7")
                else:
                    print("Warning: No scope embedding found, adding without filtering")
                    valid_outputs.append({
                        'question': text,
                        'similarity': None
                    })
                
                if len(valid_outputs) >= num_variants:
                    break
        
        attempts += 1
    
    print(f"Found {len(valid_outputs)} valid outputs after {attempts} attempts")
    return valid_outputs[:num_variants]



def prepare_prompt(year=4, transcript={}):
    task = f"generate mathematical question with answer for year {year} student"
    if transcript == {}:
        transcript = f"{generate_random_student_levels()}"
    prompt = (
        f"{task}"
        f"\n{transcript}"
    )
    return prompt



def generate_random_student_levels() -> dict:
    """
    Randomly assigns proficiency levels (Strong, Normal, Weak) to each arithmetic aspect.
    """
    aspects = ["Addition", "Subtraction", "Multiplication", "Division"]
    levels = ["Strong", "Normal", "Weak"]

    # randomly assign levels to all aspects
    transcript = {aspect: random.choice(levels) for aspect in aspects}

    # force at least one weak if none exists
    if "Weak" not in transcript.values():
        random_aspect = random.choice(aspects)
        transcript[random_aspect] = "Weak"
    
     # Step 3: Format as a transcript string
    txt = "\n".join([f"{k}:{v}" for k, v in transcript.items()])
    return txt


def generate_questions_for_specific_years(model, tokenizer, questions_per_year=10, target_year=None, max_total_generations=1000):
    """
    Generate questions for a specific year with >0.7 similarity.
    
    Args:
        max_total_generations: Maximum total generations to prevent infinite looping
    
    Returns:
        Dict with year as key and list of questions as value
    """
    # Load all embeddings once
    print("Loading embeddings by year...")
    embeddings_by_year = load_embeddings_by_year()
    
    all_questions = {}
    
    print(f"\n=== Generating {questions_per_year} questions for Year {target_year} (max {max_total_generations} generations) ===")
        
    prompt = prepare_prompt(year=target_year)
    scope_embedding = embeddings_by_year.get(target_year, None)
    
    # Override max_attempts based on max_total_generations
    max_attempts = max_total_generations // 4  # Assuming batch_size of 4
        
    questions = generate_multiple(
        prompt=prompt,
        model=model,
        tokenizer=tokenizer,
        num_variants=questions_per_year,
        year=target_year,
        scope_embedding=scope_embedding,
        max_attempts=max_attempts
    )
        
    all_questions[target_year] = questions
    print(f"Successfully generated {len(questions)} questions for Year {target_year}")
    
    if len(questions) < questions_per_year:
        print(f"Warning: Only found {len(questions)}/{questions_per_year} valid questions within generation limit")
    
    return all_questions





# Example usage
if __name__ == "__main__":
    generator = MathQuestionGenerator()
    
    # Generate a single algebra question
    print("Generating a single algebra question:")
    question = generator.generate_math_question(topic="algebra", difficulty="medium")
    print("\nQuestion:", question["question"])
    print("Answer:", question["answer"])
    print("Explanation:", question["explanation"])
    
    # Generate multiple geometry questions
    print("\nGenerating 3 geometry questions:")
    questions = generator.generate_multiple_questions(n=3, topic="geometry", difficulty="easy")
    for i, q in enumerate(questions, 1):
        print(f"\nQuestion {i}:", q["question"])
        print("Answer:", q["answer"])