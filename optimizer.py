from transformers import TrainingArguments
import optuna
from functools import partial
import json
import os
import logging

class HyperparameterOptimizer:
    def __init__(self, trainer_class, device, model_name, tokenizer_name=None):
        """
        Initialize the hyperparameter optimizer for TinyLlama or Phi-2 models.
        
        Args:
            trainer_class: The LanguageModelTrainer class to optimize
            device: Target device for training
            model_name: Pretrained model name/path (must contain 'phi' or 'llama')
            tokenizer_name: Optional different tokenizer name/path
        """
        self.trainer_class = trainer_class
        self.device = device
        self.model_name = model_name.lower()
        self.tokenizer_name = tokenizer_name
        
        # Configure model family and specific parameters
        if "phi" in self.model_name:
            self.model_family = "phi"
            self.default_batch_sizes = [2, 4, 8]  # Phi-2 typically needs smaller batches
        elif "llama" in self.model_name:
            self.model_family = "llama"
            self.default_batch_sizes = [4, 8, 16]  # TinyLlama can handle larger batches
        else:
            raise ValueError("Model must be either Phi-2 or TinyLlama")
            
        self.best_params = None
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)


    def get_model_specific_target_modules(self, trial):
        """
        Return appropriate target modules based on model family.
        """
        if self.model_family == "phi":
            # Phi-2 architecture specific modules
            return trial.suggest_categorical(
                "target_modules", 
                [
                    ["Wqkv", "out_proj"],  # Attention layers
                    ["fc1", "fc2"],        # Feed-forward layers
                    ["Wqkv", "out_proj", "fc1", "fc2"]  # All key layers
                ]
            )
        else:  # TinyLlama
            # TinyLlama architecture specific modules
            return trial.suggest_categorical(
                "target_modules", 
                [
                    ["q_proj", "v_proj"],               # Minimal attention
                    ["q_proj", "k_proj", "v_proj"],     # Full attention
                    ["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention with output
                    ["gate_proj", "up_proj", "down_proj"]  # FFN layers
                ]
            )
        

    def objective(self, trial, train_dataset, eval_dataset, output_dir_base):
        """
        Objective function for Optuna to optimize.
        """
        try:
            # Define LoRA hyperparameters to optimize
            lora_config = {
                "r": trial.suggest_categorical("lora_r", [4, 8, 16, 32]),
                "lora_alpha": trial.suggest_categorical("lora_alpha", [8, 16, 32, 64]),
                "lora_dropout": trial.suggest_float("lora_dropout", 0.01, 0.2),
                "bias": trial.suggest_categorical("lora_bias", ["none", "all", "lora_only"]),
                "target_modules": self.get_model_specific_target_modules(trial)
            }
            
            # Define training hyperparameters to optimize
            training_args = {
                "per_device_train_batch_size": trial.suggest_categorical("batch_size", self.default_batch_sizes),
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True),
                "num_train_epochs": trial.suggest_int("num_epochs", 1, 5),
                "gradient_accumulation_steps": trial.suggest_categorical("grad_accum", [1, 2, 4]),
                "warmup_steps": trial.suggest_int("warmup_steps", 0, 500),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "fp16": True if "cuda" in str(self.device).lower() else False,
                "optim": trial.suggest_categorical("optimizer", ["adamw_torch", "adamw_hf", "adafactor"])
            }

            # Create unique output directory for this trial
            trial_id = trial.number
            output_dir = os.path.join(output_dir_base, f"trial_{trial_id}")
            os.makedirs(output_dir, exist_ok=True)

            self.logger.info(f"Starting trial {trial_id} for {self.model_family} model")
            self.logger.info(f"LORA config: {lora_config}")
            self.logger.info(f"Training args: {training_args}")

            # Initialize and train the model
            trainer = self.trainer_class(
                device=self.device,
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                lora_config=lora_config
            )
            
            trainer.setup()
            trainer.train(
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                output_dir=output_dir,
                training_args=training_args
            )
            
            # Evaluate and return perplexity
            perplexity = trainer.cal_perplexity()
            self.logger.info(f"Trial {trial_id} completed with perplexity: {perplexity:.2f}")
            
            return float(perplexity)
            
        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed: {str(e)}")
            raise optuna.TrialPruned() from e
        
    def optimize(self, train_dataset, eval_dataset, output_dir_base, n_trials=20, timeout=None):
        """
        Run hyperparameter optimization.
        """
        os.makedirs(output_dir_base, exist_ok=True)
        
        # Create study with appropriate pruner
        if self.model_family == "phi":
            pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=5, reduction_factor=3)
        else:  # TinyLlama
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(),
            pruner=pruner
        )
        
        objective_partial = partial(
            self.objective,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir_base=output_dir_base
        )
        
        study.optimize(objective_partial, n_trials=n_trials, timeout=timeout)
        
        # Save best parameters
        self.best_params = study.best_params
        trials_file = os.path.join(output_dir_base, "trials.json")
        with open(trials_file, "w") as f:
            json.dump([trial.params for trial in study.trials], f, indent=2)
            
        self.logger.info(f"Best trial:")
        self.logger.info(f"  Value (perplexity): {study.best_value:.2f}")
        self.logger.info("  Params: ")
        for key, value in study.best_params.items():
            self.logger.info(f"    {key}: {value}")
            
        return self.best_params
    
    def train_with_best_params(self, train_dataset, eval_dataset, output_dir):
        """Train final model with best parameters."""
        if not self.best_params:
            raise ValueError("No optimized parameters found. Run optimize() first.")
            
        final_trainer = self.trainer_class(
            device=self.device,
            model_name=self.model_name,
            tokenizer_name=self.tokenizer_name,
            lora_config={
                "r": self.best_params["lora_r"],
                "lora_alpha": self.best_params["lora_alpha"],
                "lora_dropout": self.best_params["lora_dropout"],
                "bias": self.best_params["lora_bias"],
                "target_modules": self.best_params["target_modules"]
            }
        )
        
        final_trainer.setup()
        final_trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=output_dir,
            training_args={
                "per_device_train_batch_size": self.best_params["batch_size"],
                "learning_rate": self.best_params["learning_rate"],
                "num_train_epochs": self.best_params["num_epochs"],
                "gradient_accumulation_steps": self.best_params["grad_accum"],
                "warmup_steps": self.best_params["warmup_steps"],
                "weight_decay": self.best_params["weight_decay"],
                "fp16": True if "cuda" in str(self.device).lower() else False,
                "optim": self.best_params.get("optimizer", "adamw_torch")
            }
        )
        
        return final_trainer