import torch
import numpy as np
from typing import List
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer as NLITokenizer
)

class NLIClustering:
    """
    LLM-based clustering system following Enhanced SEP proposal methodology.
    Uses LLM classifier.
    """
    
    def __init__(self, config, logger):
        self.logger = logger
        self.config = config
        logger.info("=" * 60)
        logger.info("INITIALIZING NLI-BASED CLUSTERING")
        logger.info("=" * 60)
        
                # Load NLI model for semantic clustering
        logger.info(f"Loading NLI model: {config.nli_model_name}")
        self.nli_tokenizer = NLITokenizer.from_pretrained(config.nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model_name
        ).to(self.config.device)
            

    def check_bidirectional_entailment(self, text1: str, text2: str) -> bool:
        """Check if two texts entail each other bidirectionally"""
        if text1.strip() == text2.strip():
            return True
            
        # Check text1 -> text2
        entails_12 = self._check_entailment(text1, text2)
        # Check text2 -> text1  
        entails_21 = self._check_entailment(text2, text1)
        
        return entails_12 and entails_21
    
    def _check_entailment(self, premise: str, hypothesis: str) -> bool:
        """Check if premise entails hypothesis using NLI model"""
        inputs = self.nli_tokenizer(
            premise, hypothesis, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.config.max_length
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # For DeBERTa-MNLI: 0=entailment, 1=neutral, 2=contradiction
        # return false if contradiction is higer than 0.5
        entailment_prob = predictions[0][0].item()  # Probability of entailment
        neutral_prob = predictions[0][1].item()  # Probability of neutral
        contradiction_prob = predictions[0][2].item()  # Probability of 
        
        if contradiction_prob > entailment_prob + neutral_prob:
            return False
        return True
    
    def cluster_responses(self, responses: List[str]) -> List[List[int]]:
        """Cluster responses by semantic equivalence using greedy algorithm"""
        clusters = []
        response_to_cluster = {}
        
        for i, response in enumerate(responses):
            # Try to find existing cluster this response belongs to
            assigned = False
            for cluster_idx, cluster in enumerate(clusters):
                # Check against representative response in cluster
                representative_idx = cluster[0]
                representative_response = responses[representative_idx]
                
                if self.check_bidirectional_entailment(response, representative_response):
                    clusters[cluster_idx].append(i)
                    response_to_cluster[i] = cluster_idx
                    assigned = True
                    break
            
            # Create new cluster if no match found
            if not assigned:
                clusters.append([i])
                response_to_cluster[i] = len(clusters) - 1


        total_responses = len(responses)

        cluster_probs = []
        for cluster in clusters:
            prob = len(cluster) / total_responses
            cluster_probs.append(prob)

        # Calculate semantic entropy
        semantic_entropy_nli = -sum(p * np.log(p) for p in cluster_probs if p > 0)

        metadata_nli = {
            "clusters": clusters,
            "num_clusters": len(clusters),
            "semantic_entropy": semantic_entropy_nli
        }

        return metadata_nli
                