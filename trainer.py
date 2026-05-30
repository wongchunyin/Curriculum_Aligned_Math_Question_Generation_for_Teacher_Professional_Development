from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)
from datasets import Dataset
import torch as th
from typing import Dict, Optional
from custom_data_collator import CustomDataCollator
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig # Add TaskType for common LoRA configs


class LanguageModelTrainer:
    def __init__(
        self,
        device: int,
        model_name: str,
        tokenizer_name: Optional[str] = None,
        lora_config: Optional[Dict] = None, # Add lora_config as an argument
    ):
        """
        Initialize trainer with model and tokenizer
        Args:
            model_name: Pretrained model name/path
            tokenizer_name: Optional different tokenizer name/path
            device: Target device (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name or model_name
        self.lora_config_params = lora_config # Store LoRA config parameters


        # Initialize in setup()
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self._trainer = None


    def setup(self, trust_remote_code=False):
        """Initialize model, tokenizer and collator , and apply LoRA if configured."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name, trust_remote_code=trust_remote_code)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        ).to(self.device)


                
        if self.lora_config_params:
            print("Applying LoRA to the model...")
            # Default LoRA config parameters, can be overridden by self.lora_config_params
            default_lora_config = {
                "r": 8,                  # Rank of the update matrices
                "lora_alpha": 16,        # Scaling factor for LoRA
                "lora_dropout": 0.05,    # Dropout probability for LoRA layers
                "bias": "none",          # Bias type ('none', 'all', 'lora_only')
                "task_type": TaskType.CAUSAL_LM, # Define the task type
                # Add target_modules here based on your model architecture.
                # Common ones for causal LMs are often "q_proj", "v_proj", "k_proj", "o_proj"
                # You might need to inspect your model's architecture (e.g., print(base_model))
                # to find the correct module names.
                # Example for some models: "target_modules": ["q_proj", "v_proj"]
            }

            default_lora_config.update(self.lora_config_params)

            lora_config = LoraConfig(**default_lora_config)
                
            # Wrap the base model with PEFT
            self.model = get_peft_model(base_model, lora_config)
            print("LoRA model initialized. Trainable parameters:")
            self.model.print_trainable_parameters()

        else:
            self.model = base_model
            print("LoRA not applied. Full model loaded.")


        self.data_collator = CustomDataCollator(
            tokenizer=self.tokenizer
        )

    def tokenize_dataset(
        self,
        dataset: Dataset,
        text_field: str = "text",
        max_length: int = 512,
        truncation: bool = True
    ) -> Dataset:
        """
        Tokenize dataset for language modeling with 'input' and 'output' fields.
        Concatenates "<s>" + input + "\n" + output + "</s>".
        Labels are created as a copy of input_ids.
        Padding and final truncation to `max_length` are expected to be handled by the data collator.

        Args:
            dataset: HuggingFace Dataset with 'input' and 'output' fields.
            max_length: The maximum sequence length. This will be enforced by the
                        data collator during batching, as `truncation=False` is used here.
        Returns:
            Tokenized dataset with 'input_ids' and 'labels'.
        """
        def tokenize_fn(examples):

            texts = [
                "<s>" + inp + "\n" + out + "</s>"
                for inp, out in zip(examples["input"], examples["output"])
            ]

            encoding = self.tokenizer(
                texts,
                padding=False,         # No padding here, data collator handles it
                truncation=False,      # No truncation here, data collator handles it
                return_tensors=None,   # Return as lists/Python types
                add_special_tokens=True # Add model's special tokens (like BOS/EOS if not manually added)
            )
            
            # Labels are a copy of input_ids for causal language modeling
            # The data collator will typically mask out padding tokens in labels with -100
            encoding["labels"] = [ids[:] for ids in encoding["input_ids"]]

            return encoding
            
        return dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col not in ["input_ids", "attention_mask", "labels"]]
        )

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "./output",
        training_args: Optional[Dict] = None
    ):
        """
        Run training process
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            output_dir: Output directory for results
            training_args: Custom training arguments
        """
        default_args = {
            "output_dir": output_dir,
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 4,
            "num_train_epochs": 3,
            "logging_dir": f"{output_dir}/logs",
            "logging_strategy":"steps",          # <-- Important
            "logging_steps":100,                  # <-- Show log every N steps
            "learning_rate": 2e-4,
            "save_strategy": "epoch",
            "eval_steps": 100,
            "gradient_accumulation_steps": 2,
            "fp16": th.cuda.is_available(),
            "remove_unused_columns": False
        }


        
        # Merge with custom args
        if training_args:
            default_args.update(training_args)
            
        args = TrainingArguments(**default_args)
        
        self._trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer
        )

        self._trainer.train()
        
        # Save final model
        self._trainer.save_model(f"{output_dir}/{self.model_name}")
        self.tokenizer.save_pretrained(f"{output_dir}/{self.model_name}")
        print(f"Model adapters and tokenizer saved to {output_dir}/{self.model_name}")


    @classmethod
    def load_model(cls, path: str, device: str = "cpu"):
        """
        Load trained model from path
        Args:
            path: Path to saved model
            device: Target device
        Returns:
            New LanguageModelTrainer instance
        """
        device = device
        trainer = cls(model_name=path, device=device)

        # Load PEFT config to get base model name
        peft_config = PeftConfig.from_pretrained(path)
        base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)

        # Load adapter into base model
        trainer.model = PeftModel.from_pretrained(base_model, path).to(device)

        # Load tokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        # trainer.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True).to(device)
        # trainer.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        # trainer.setup()
        return trainer


    def cal_perplexity(self):
        if not hasattr(self, '_trainer') or self._trainer is None:
            raise ValueError("Training has not been run or trainer instance is not available.")
        if self._trainer.eval_dataset is None:
            raise ValueError("No evaluation dataset was provided during training to calculate perplexity.")


        if self.model is None:
            raise ValueError("Model not initialized. Call setup() first.")
        if self.tokenizer is None:
            raise ValueError("Tokenizer not initialized. Call setup() first.")
        if self.data_collator is None:
            raise ValueError("Data collator not initialized. Call setup() first.")



        results = self._trainer.evaluate()
        if "eval_loss" not in results:
            raise ValueError("Evaluation results do not contain 'eval_loss'. Ensure evaluation strategy is set.")

        perplexity = th.exp(th.tensor(results["eval_loss"]))
        print(f"Perplexity: {perplexity:.2f}")
        return perplexity
