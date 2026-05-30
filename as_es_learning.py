from gsm8k_dataset import GSMDataset, get_examples, split_solution_into_subsentences, calculate_score_M, classify_gsm8k_solution_segments
from tqdm import tqdm 
from evaluation_matrix import compute_bleu
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np


class as_es_classification:
    """
    Class for AS/ES classification of GSM8K solutions.
    This class provides methods to generate labels for AS/ES classification based on the GSM8K dataset.
    """

    def __init__(self, model_name="google/flan-t5-base"):
        """Initialize the AS/ES classification model with a specified T5 model."""
        if model_name != "google/flan-t5-base":
            raise ValueError("Currently, only 'google/flan-t5-base' is supported for AS/ES classification.")
        
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)


    def data_validation(self, data):
        if type(data) != GSMDataset:
            return False
        
        if data is None or len(data) == 0:
            return False
        
        return True
        
    def label_generator(self, data):
        if not self.data_validation(data):
            raise ValueError("Data validation failed. Please check the dataset.")
    
        metrics = self.calculate_metrics(data)
        normalize_metrics = self.normalize_metrics_per_answer(metrics)
        labels = self.combining_segmentation(normalize_metrics)
        # labels = self.as_es_classification(normalize_metrics)

        return labels, metrics, normalize_metrics

    def calculate_metrics(self, data):
        results = {
            "entropy": [],
            "loss": [],
            "location": [],
            "bleu": []
        }

        with torch.no_grad():
            for idx in tqdm(range(0, len(data)), desc="Generating labels for AS/ES classification"): 
                question, answer, grade = data.__getitemInTxt__(idx)
                sub_sentences = split_solution_into_subsentences(answer)
                entropy_scores = []
                loss_scores = []
                location_scores = []

                # calculate BlEU by using corpus_bleu
                bleu_scores, _ = self.compute_bleu(reference=question, candidates=sub_sentences)
                
                # print(f"Question: {question}")
                for j, sentence in enumerate(sub_sentences):
                    # print(f"  {j+1}: {sentence}")

                    # calculate entropy and loss
                    entropy, loss = self.calculate_entropy_and_loss(query=question, sub_sentence=sentence)
                    entropy_scores.append(entropy)
                    loss_scores.append(loss)
                    # calculate the location score
                    location_scores.append(self.calculate_location(j))


                # end of the second loop
                results["entropy"].append(entropy_scores)
                results["loss"].append(loss_scores)
                results["bleu"].append(bleu_scores)
                results["location"].append(location_scores)

        # end of the first loop
        return results


    def calculate_entropy_and_loss(self, query, sub_sentence):    
        """Returns both loss and entropy for a sub-sentence given the query."""
        # Tokenize inputs
        encoder_inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        decoder_inputs = self.tokenizer(sub_sentence, return_tensors="pt", truncation=True, max_length=512)
        
        # Prepare labels (shifted for teacher forcing)
        labels = decoder_inputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss
        
        # with torch.no_grad():
        outputs = self.model(
            input_ids=encoder_inputs.input_ids,
            attention_mask=encoder_inputs.attention_mask,
            decoder_input_ids=decoder_inputs.input_ids,
            decoder_attention_mask=decoder_inputs.attention_mask,
            labels=labels,
            return_dict=True
        )
            
            # Get loss (cross-entropy)
        loss = outputs.loss.item()
            
        # Calculate entropy from logits
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean().item()
        
        return entropy, loss
        # return {"loss": loss, "entropy": entropy}

    
    def compute_bleu(self, candidates, reference, n_gram=4, smoothing=True):
        """
        Compute BLEU scores for multiple generated sentences compared to a single reference sentence.

        Args:
            generated_sentences (list of str): Sentences to evaluate.
            reference_sentence (str): The single reference sentence.
            n_gram (int): Maximum n-gram size (default BLEU-4).
            smoothing (bool): Whether to apply smoothing to avoid zero scores.

        Returns:
            list of float: BLEU scores for each generated sentence.
            float: Average BLEU score across all inputs.
        """
        ref_tokens = reference.split()
        weight = tuple([1.0 / n_gram] * n_gram) # Equal weights for n-grams e.g., (0.25, 0.25, 0.25, 0.25) for BLEU-4
        smoothie = SmoothingFunction().method4 if smoothing else None

        bleu_scores = []
        for gen in candidates:
            gen_tokens = gen.split()
            score = sentence_bleu([ref_tokens], gen_tokens, weights=weight, smoothing_function=smoothie)
            bleu_scores.append(score)

        average_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        return bleu_scores, average_score
    

    def calculate_location(self, location):
        """
        Classifies segments based on their position in the solution.
        Even-indexed segments are classified as 0, odd-indexed as 1.
        """
        if location % 2 == 0:
            return 0
        else:
            return 1


    def avg(self, scores):
        """
        Calculate the average of a list of scores.
        """
        if scores is None or len(scores) == 0:
            return 0.0

        total_score = 0
        no_of_scores = 0
        for score in scores:
            total_score += score
            no_of_scores += 1

        return total_score / no_of_scores


    def as_es_classification(self, metrics):
        """
        Classifies each segment of a GSM8K solution as AS or ES based on the chosen strategy.
        """

        results = {
            "entropy": [],
            "loss": [],
            "location": [],
            "bleu": []
        }
        
        for key in results.keys():
            for metric in metrics[key]:
                labels = []
                score_M = np.sum(metric)/len(metric)
                for score_i in metric:
                    if key != "location":
                        label = self.segmentation(
                            score_M=score_M, 
                            s_i=score_i # get the score of each sub-sentence by key
                            )
                    else:
                        label = self.interleaving_segmentation(score_i)

                    labels.append(label)
                    # end of the inner loop
                
                # end of the outer loop      
                results[key].append(labels)  
            # end of the first loop   
        return results

    def interleaving_segmentation(self, s_i):
        return "ES" if s_i % 2 == 0 else "AS"

    def segmentation(self, score_M, s_i, bate=1.0):
        if s_i > bate * score_M:
            return "AS"
        else:
            return "ES"
        
    def combining_segmentation(self, normalized_metrics, beta_threshold=0.5):
        weights = {
            'entropy': 0.3,
            'loss': 0.3,
            'bleu': 0.2,
            'location': 0.2
        }
                
        all_labels = []
        for i in range(len(normalized_metrics["entropy"])):
            entropy_list = normalized_metrics["entropy"][i]
            loss_list = normalized_metrics["loss"][i]
            bleu_list = normalized_metrics["bleu"][i]
            loc_list = normalized_metrics["location"][i]

            labels = []
            for s_i in range(len(entropy_list)):
                score = (
                    weights['entropy'] * entropy_list[s_i] +
                    weights['loss'] * loss_list[s_i] +
                    weights['bleu'] * bleu_list[s_i] +
                    weights['location'] * loc_list[s_i]
                )
                labels.append("AS" if score > beta_threshold else "ES")
            all_labels.append(labels)

        return all_labels

        

    def normalize_list(self, values):
        min_val = min(values)
        max_val = max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]


    def normalize_metrics_per_answer(self, dataset):
        normalized = {
            "entropy": [],
            "loss": [],
            "bleu": [],
            "location": dataset["location"]  # keep location as is (binary)
        }
        
        for i in range(len(dataset["entropy"])):
            entropy_row = self.normalize_list(dataset["entropy"][i])
            loss_row = self.normalize_list(dataset["loss"][i])
            inv_bleu = [1 - b for b in dataset["bleu"][i]]  # invert before normalize
            bleu_row = self.normalize_list(inv_bleu)

            normalized["entropy"].append(entropy_row)
            normalized["loss"].append(loss_row)
            normalized["bleu"].append(bleu_row)

        return normalized