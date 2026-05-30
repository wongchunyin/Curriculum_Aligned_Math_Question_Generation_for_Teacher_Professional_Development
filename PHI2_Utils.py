from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

class PHI2_Utils:
    def __init__(self):
        self.base_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.base_model(**inputs, output_hidden_states=True)
        return outputs.hidden_states[-1].mean(dim=1).detach().numpy()
    
    @staticmethod
    def save_embedded_obj(obj, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        print(f"Embedded object saved to {path}")

    @staticmethod
    def load_embedded_obj(path: str):
        import pickle
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def cosine_similarity(embedding1, embedding2):
        import numpy as np
        dot_product = np.dot(embedding1.flatten(), embedding2.flatten())
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
        
    

if __name__ == "__main__":
    phi2Utils = PHI2_Utils()
    import json
    with open("curriculum_Desc/desc.json", "r") as f:
        curriculum_data = json.load(f)

    descriptions = [item.get('scope', '') for item in curriculum_data]
    # Batch processing for efficiency
    batch_size = 16  # Adjust batch size as needed
    embeddings = []
    
    for i in range(0, len(descriptions), batch_size):
        batch = descriptions[i:i+batch_size]
        inputs = phi2Utils.tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        outputs = phi2Utils.base_model(**inputs, output_hidden_states=True)
        batch_embeddings = outputs.hidden_states[-1].mean(dim=1).detach().numpy()
        embeddings.extend(batch_embeddings)
    phi2Utils.save_embedded_obj(embeddings, "curriculum_Desc/desc_embeddings.pkl")
    print("Embeddings saved.")
    print(embeddings)
    print(len(embeddings))
