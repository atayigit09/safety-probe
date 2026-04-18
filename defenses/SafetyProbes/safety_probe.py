import torch
from typing import List, Dict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoModelForSequenceClassification, AutoTokenizer as NLITokenizer
)

from defenses.SafetyProbes.clustering.intent_based import IntentBasedClustering
from defenses.SafetyProbes.clustering.llm_clustering import LLMClustering
from defenses.SafetyProbes.clustering.bidirectional_entailment import NLIClustering
import logging
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


    
class SafetyProbe:
    """Implementation of semantic entropy calculation from the paper"""
    
    def __init__(self, config):
        self.model_config = config.model
        self.generation_config = config.generation
        self.safety_entropy_config = config.safety_entropy
        self.clustering_config = config.clustering
        self.data_config = config.data
        self.device = torch.device(self.model_config.device)

        if self.model_config.load_model:
            # Load main language model
            logger.info(f"Loading model: {self.model_config.model_id}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.model_id)
            attn_implementation = getattr(self.model_config, 'attn_implementation', 'eager')
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    # H100 optimizations
                    use_cache=self.model_config.use_cache,
                    low_cpu_mem_usage=self.model_config.low_cpu_mem_usage,
                    attn_implementation=attn_implementation,
                )
                logger.info(f"Model loaded with {attn_implementation} attention")
            except Exception as e:
                logger.warning(f"Failed to load model with Flash Attention 2: {e}")
                logger.info("Falling back to standard attention...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_config.model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    use_cache=self.model_config.use_cache,
                    low_cpu_mem_usage=self.model_config.low_cpu_mem_usage,
                )
            

            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                

            # Compile model for GPU optimization (PyTorch 2.0+)
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("Model compiled with torch.compile")
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
        else:
            logger.info("Model not loaded only clustering can be used")


        #initialize the clustering methods
        #self.intent_based_clustering = IntentBasedClustering(self.clustering_config.intent_based, logger)
        self.llm_clustering = LLMClustering(self.clustering_config.llm_based, logger)
        #self.nli_clustering = NLIClustering(self.clustering_config.nli_based, logger)


    def clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def generate_multiple_responses(self, prompt: str) -> List[str]:
        """Generate multiple responses for semantic entropy calculation"""

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        responses = []
        for _ in range(self.safety_entropy_config.num_generations):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.generation_config.max_length,
                    temperature=self.generation_config.temperature,
                    do_sample=True,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[-1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response.strip())
            
        return responses
    
    def generate_multiple_responses_batch(self, prompt: str) -> List[str]:
        """Generate multiple responses for semantic entropy calculation in batches"""
        batch_size = self.model_config.batch_size
        num_generations = self.safety_entropy_config.num_generations
    
        #make sure that num_generations is divisible by batch_size
        assert num_generations % batch_size == 0, "num_generations must be divisible by batch_size"
        
        # Tokenize the prompt once
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        responses = []
        num_batches = num_generations // batch_size
        
        for batch_idx in range(num_batches):
            # Create batch by repeating the same prompt
            batch_inputs = {
                'input_ids': inputs['input_ids'].repeat(batch_size, 1),
                'attention_mask': inputs['attention_mask'].repeat(batch_size, 1)
            }
            
            with torch.no_grad():
                # Generate responses for the entire batch
                batch_outputs = self.model.generate(
                    **batch_inputs,
                    max_new_tokens=self.generation_config.max_length,
                    temperature=self.generation_config.temperature,
                    do_sample=True,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Process each response in the batch
            for i in range(batch_size):
                # Extract only the generated part for each response
                generated_tokens = batch_outputs[i][inputs['input_ids'].shape[-1]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response.strip())
                

            # Clear GPU cache between batches to prevent memory buildup
            self.clear_gpu_cache()
        
        return responses

    def format_prompt(self, prompt: str) -> str:
        """
        Formats the input prompt using the tokenizer's chat template.
        This ensures we use the correct format for the instruction-tuned LLaMA model.
        For LLaMA 3.2 Instruct, this uses the proper chat format with special tokens.
        """
        # Use the tokenizer's built-in chat template
        messages = [
            {"role": "user", "content": prompt}
        ]
        #for qwen family models, set thinking=False
        if self.model_config.model_id.startswith("Qwen"):
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            
        else:
            return self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )

    def process_dataset(self, start_index = 0, dataset_type = "harmful"):
        """Process the dataset and update JSON file after each object for progress tracking"""
        if dataset_type == "harmful":
            dataset = json.load(open(self.data_config.harmful_path))
        elif dataset_type == "benign":
            dataset = json.load(open(self.data_config.benign_path))
        else:
            raise ValueError(f"Invalid dataset type: {dataset_type}")
       
        output_filename = f"{self.data_config.save_path}/{dataset_type}_processed.json"
        if not os.path.exists(self.data_config.save_path):
            os.makedirs(self.data_config.save_path)
        
        # Initialize or load existing processed data
        try:
            with open(output_filename, "r") as f:
                processed_objects = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            processed_objects = []
        
        for obj in dataset:
            idx = obj["id"]
            if idx < start_index:
                continue
                
            logger.info(f"Processing object {idx}")
            prompt = obj["prompt"]
            formatted_prompt = self.format_prompt(prompt)
            if self.model_config.batch_size > 1:
                responses = self.generate_multiple_responses_batch(formatted_prompt)
            else:
                responses = self.generate_multiple_responses(formatted_prompt)
            #update the object with the responses
            obj["llm_responses"] = responses
            #add the processed object to our list
            processed_objects.append(obj)
            
            # Save updated list to file after each iteration for progress tracking
            with open(output_filename, "w") as f:
                json.dump(processed_objects, f, indent=2)
            
            logger.info(f"Saved progress: {len(processed_objects)} objects processed")
        
        logger.info(f"Processing complete! Total objects processed: {len(processed_objects)}")
      

    def cluster_responses(self, load_from_file = True):
        output_filename = self.clustering_config.save_path + "/clustered_data.json"
        if not os.path.exists(self.clustering_config.save_path):
            os.makedirs(self.clustering_config.save_path)
        
        # Initialize or load existing processed data
        try:
            with open(output_filename, "r") as f:
                clustered_objects = json.load(f)
                last_id = clustered_objects[-1]["id"]
        except (FileNotFoundError, json.JSONDecodeError):
            clustered_objects = []
            last_id = -1


        if load_from_file:
            processed_file = self.clustering_config.processed_file
            with open(processed_file, "r") as f:
                processed_objects = json.load(f)
        else:
            logger.info("Not implemented yet...")
            return
        
        for obj in processed_objects:
            if obj["id"] <= last_id:
                continue
            
            responses = obj["llm_responses"]
            prompt = obj["prompt"]
            
            llm_data = self.llm_clustering.cluster_responses(responses, prompt)
            #nli_data = self.nli_clustering.cluster_responses(responses)
            #intent_data = self.intent_based_clustering.cluster_responses(responses, prompt)
            obj["llm_clustering"] = llm_data
            #obj["nli_clustering"] = nli_data
            #obj["intent_clustering"] = intent_data

            clustered_objects.append(obj)

            with open(output_filename, "w") as f:
                json.dump(clustered_objects, f, indent=2)
            
            logger.info(f"Saved progress: {len(clustered_objects)} objects clustered")

        
        logger.info(f"Processing complete! Total objects clustered: {len(clustered_objects)}")




