import torch
import json
from datasets import Dataset
from trainer import LanguageModelTrainer
from math_question_generator import generate_questions_for_specific_years, MathQuestionGenerator, prepare_prompt
from PHI2_Utils import PHI2_Utils

def collect_feedback_data(model, tokenizer, years=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], samples_per_year=100):
    """
    Collect training data with similarity feedback for all years.
    
    Args:
        model: The model to generate questions
        tokenizer: The tokenizer
        years: List of years to generate data for
        samples_per_year: Number of samples per year
    
    Returns:
        List of training examples with similarity scores
    """
    phi2_utils = PHI2_Utils()
    training_data = []
    
    # Load embeddings once
    with open("embedded_desc.json", "r") as f:
        embedded_desc = json.load(f)
    
    embeddings_by_year = {}
    for desc in embedded_desc:
        desc_year = desc['year'].split()[-1] if 'year' in desc else None
        if desc_year:
            year_int = int(desc_year)
            embeddings_by_year[year_int] = desc['scope']
    
    for year in years:
        print(f"Collecting data for year {year}...")
        scope_embedding = embeddings_by_year.get(year)
        
        if scope_embedding is None:
            print(f"No scope embedding found for year {year}, skipping...")
            continue
            
        for i in range(samples_per_year):
            prompt = prepare_prompt(year=year)
            
            # Generate question
            from math_question_generator import generate
            question = generate(prompt, model, tokenizer=tokenizer)
            
            # Calculate similarity
            question_embedding = phi2_utils.embed(question)
            similarity = phi2_utils.cosine_similarity(scope_embedding, question_embedding)
            
            # Create training example
            training_data.append({
                "input": prompt,
                "output": question,
                "similarity": similarity,
                "year": year,
                "reward": similarity  # Use similarity as reward signal
            })
            
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{samples_per_year} samples for year {year}")
    
    return training_data

def create_preference_pairs(training_data, similarity_threshold=0.7):
    """
    Create preference pairs for DPO-style training.
    
    Args:
        training_data: List of training examples with similarity scores
        similarity_threshold: Threshold to separate good/bad examples
    
    Returns:
        Dataset with chosen/rejected pairs
    """
    preference_pairs = []
    
    # Group by prompt (year)
    by_prompt = {}
    for example in training_data:
        prompt = example["input"]
        if prompt not in by_prompt:
            by_prompt[prompt] = []
        by_prompt[prompt].append(example)
    
    # Create pairs within each prompt group
    for prompt, examples in by_prompt.items():
        good_examples = [ex for ex in examples if ex["similarity"] > similarity_threshold]
        bad_examples = [ex for ex in examples if ex["similarity"] <= similarity_threshold]
        
        # Create pairs: good vs bad
        for good in good_examples:
            for bad in bad_examples:
                preference_pairs.append({
                    "input": prompt,
                    "chosen": good["output"],
                    "rejected": bad["output"],
                    "chosen_similarity": good["similarity"],
                    "rejected_similarity": bad["similarity"]
                })
    
    return Dataset.from_list(preference_pairs)

def create_supervised_dataset(training_data, similarity_threshold=0.7):
    """
    Create supervised fine-tuning dataset with only high-similarity examples.
    
    Args:
        training_data: List of training examples with similarity scores
        similarity_threshold: Minimum similarity to include
    
    Returns:
        Dataset for supervised fine-tuning
    """
    filtered_data = [
        {"input": ex["input"], "output": ex["output"]}
        for ex in training_data 
        if ex["similarity"] > similarity_threshold
    ]
    
    return Dataset.from_list(filtered_data)

def load_existing_generation_data(data_file_path, default_year=5):
    """
    Load existing generated data from generate_questions_for_specific_years.
    
    Args:
        data_file_path: Path to JSON file with generated questions and similarities
        default_year: Default year to use when year info is missing
    
    Returns:
        List of training examples
    """
    with open(data_file_path, "r") as f:
        data = json.load(f)
    
    training_data = []
    
    # Handle different data formats
    if isinstance(data, dict):
        # Format: {"year": [{"question": ..., "similarity": ...}]}
        for year_str, questions in data.items():
            year = int(year_str)
            for q_data in questions:
                if isinstance(q_data, dict):
                    question_text = q_data["question"]
                    similarity = q_data["similarity"]
                else:
                    question_text = q_data
                    similarity = None
                
                training_data.append({
                    "input": prepare_prompt(year=year),
                    "output": question_text,
                    "similarity": similarity,
                    "year": year,
                    "reward": similarity
                })
    
    elif isinstance(data, list):
        # Format: [{"question": ..., "similarity": ..., "year": ...}]
        for item in data:
            if isinstance(item, dict):
                question_text = item.get("question", "")
                similarity = item.get("similarity", None)
                year = item.get("year", default_year)  # Use provided default year
                
                # Generate a generic prompt since we don't know the original prompt
                prompt = prepare_prompt(year=year)
                
                training_data.append({
                    "input": prompt,
                    "output": question_text,
                    "similarity": similarity,
                    "year": year,
                    "reward": similarity
                })
    
    return training_data

def feedback_fine_tune_phi2(
    base_model_path="microsoft/phi-2",
    output_dir="./phi2_feedback_model",
    years=[5, 6, 7, 8, 9, 10],
    samples_per_year=50,
    similarity_threshold=0.7,
    training_method="supervised",  # "supervised" or "preference"
    existing_data_path=None,  # Path to existing Generation V3 data
    default_year=5  # Default year for existing data without year info
):
    """
    Fine-tune Phi-2 using feedback from similarity scores.
    
    Args:
        base_model_path: Path to base model
        output_dir: Output directory for fine-tuned model
        years: Years to collect training data for
        samples_per_year: Number of samples per year
        similarity_threshold: Similarity threshold for filtering
        training_method: "supervised" or "preference"
        existing_data_path: Path to existing generated data (if None, generates new data)
    """
    
    if existing_data_path:
        print(f"Loading existing data from {existing_data_path}...")
        training_data = load_existing_generation_data(existing_data_path, default_year)
    else:
        # Initialize base model for data collection
        print("Initializing base model for data collection...")
        generator = MathQuestionGenerator(base_model_path)
        
        # Collect feedback data
        print("Collecting feedback data...")
        training_data = collect_feedback_data(
            generator.model, 
            generator.tokenizer, 
            years=years, 
            samples_per_year=samples_per_year
        )
    
    # Save raw training data
    with open(f"{output_dir}_training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Collected {len(training_data)} training examples")
    print(f"High similarity examples (>{similarity_threshold}): {sum(1 for ex in training_data if ex['similarity'] > similarity_threshold)}")
    
    # Create dataset based on training method
    if training_method == "supervised":
        dataset = create_supervised_dataset(training_data, similarity_threshold)
        print(f"Created supervised dataset with {len(dataset)} examples")
        
        # Initialize trainer with LoRA
        lora_config = {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        }
        
        trainer = LanguageModelTrainer(
            device="cuda" if torch.cuda.is_available() else "cpu",
            model_name=base_model_path,
            lora_config=lora_config
        )
        
        trainer.setup(trust_remote_code=True)
        
        # Tokenize dataset
        tokenized_dataset = trainer.tokenize_dataset(dataset)
        
        # Training arguments
        training_args = {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "learning_rate": 1e-4,
            "warmup_steps": 100,
            "logging_steps": 50,
            "save_strategy": "epoch"
        }
        
        # Train
        trainer.train(
            train_dataset=tokenized_dataset,
            output_dir=output_dir,
            training_args=training_args
        )
        
    elif training_method == "preference":
        preference_dataset = create_preference_pairs(training_data, similarity_threshold)
        print(f"Created preference dataset with {len(preference_dataset)} pairs")
        
        # For preference learning, you would need DPO implementation
        # This is a placeholder - you'd need to implement DPO trainer
        print("Preference learning not implemented yet. Use supervised method.")
        return
    
    print(f"Fine-tuning completed! Model saved to {output_dir}")

if __name__ == "__main__":
    # Run feedback fine-tuning with existing Generation V3 data
    feedback_fine_tune_phi2(
        base_model_path="microsoft/phi-2",
        output_dir="./phi2_feedback_model",
        similarity_threshold=0.7,
        training_method="supervised",
        existing_data_path="generation_v3_data.json"  # Use existing data
    )
    
    # Or generate new data:
    # feedback_fine_tune_phi2(
    #     base_model_path="microsoft/phi-2",
    #     output_dir="./phi2_feedback_model",
    #     years=[5, 6, 7, 8, 9, 10],
    #     samples_per_year=30,
    #     similarity_threshold=0.7,
    #     training_method="supervised"
    # )