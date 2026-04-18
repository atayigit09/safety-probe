"""
Feature Generation and Caching Script for Safety Probes
------------------------------------------------------
This script extracts hidden states from Llama models at two token positions:
- TBG (Token Before Generation): Last token of the prompt
- SLT (Second Last Token): Second-to-last token of the full sequence (prompt + response)
Features and corresponding safety labels are cached to disk to avoid re-extraction.
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from tqdm import tqdm
import argparse

def extract_hidden_states_tbg(prompts, model, tokenizer, device, batch_size=8):
    all_tbg_states = []

    # Sequential processing when batch_size is 1
    if batch_size == 1:
        for i in tqdm(range(len(prompts)), desc="Extracting TBG hidden states (sequential)"):
            prompt = prompts[i]
            
            # === Extract TBG (Token Before Generation) ===
            formatted_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True
            )

            prompt_inputs = tokenizer(
                formatted_prompt,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**prompt_inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                attention_mask = prompt_inputs['attention_mask']

            # Extract last token for each layer
            last_token_position = attention_mask.sum(dim=1) - 1
            batch_tbg_states = []
            for layer_idx, layer_states in enumerate(hidden_states):
                layer_last_token = layer_states[0, last_token_position[0]].cpu()
                batch_tbg_states.append(layer_last_token.unsqueeze(0))  # Add batch dimension
            all_tbg_states.append(batch_tbg_states)
    else:
        # Original batch processing
        for i in tqdm(range(0, len(prompts), batch_size), desc="Extracting TBG hidden states"):
            batch_prompts = prompts[i:i + batch_size]

            # === Extract TBG (Token Before Generation) ===
            formatted_prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for prompt in batch_prompts
            ]

            prompt_inputs = tokenizer(
                formatted_prompts,
                return_tensors="pt"
            ).to(device)

            with torch.no_grad():
                outputs = model(**prompt_inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                attention_mask = prompt_inputs['attention_mask']

            batch_tbg_states = []
            for layer_idx, layer_states in enumerate(hidden_states):
                last_token_positions = attention_mask.sum(dim=1) - 1
                layer_last_tokens = []
                for seq_idx, last_pos in enumerate(last_token_positions):
                    layer_last_tokens.append(layer_states[seq_idx, last_pos].cpu())
                batch_tbg_states.append(torch.stack(layer_last_tokens))
            all_tbg_states.append(batch_tbg_states)

    # Concatenate all batches
    final_tbg_states = []
    num_layers = len(all_tbg_states[0])
    for layer_idx in range(num_layers):
        tbg_layer_states = torch.cat([batch[layer_idx] for batch in all_tbg_states], dim=0)
        final_tbg_states.append(tbg_layer_states)

    return final_tbg_states

def save_hidden_states(hidden_states_dict, cache_path, metadata=None):
    cache_data = {
        'tbg_states': hidden_states_dict['tbg'],
        'metadata': metadata or {},
        'num_layers': len(hidden_states_dict['tbg']),
        'hidden_dim': hidden_states_dict['tbg'][0].shape[-1],
        'num_samples_tbg': hidden_states_dict['tbg'][0].shape[0]
    }
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save(cache_data, cache_path)


def load_data(data_path):
    """
    Load clustered data with safety entropy and safety score.
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    # Filter out samples missing clustering info
    data_with_metrics = [
        item for item in data
        if 'llm_clustering' in item
        and 'safety_entropy' in item['llm_clustering']
        and 'safety_score' in item['llm_clustering']
    ]
    print(f"Filtered to {len(data_with_metrics)} samples with safety metrics")

    prompts = [item['prompt'] for item in data_with_metrics]
    responses = [item['llm_responses'][0] if item['llm_responses'] else "" for item in data_with_metrics]
    safety_entropies = np.array([item['llm_clustering']['safety_entropy'] for item in data_with_metrics], dtype=np.float32)
    safety_scores = np.array([item['llm_clustering']['safety_score'] for item in data_with_metrics], dtype=np.float32)
    joint_risk_targets = np.array([item['llm_clustering']['joint_risk_target'] for item in data_with_metrics], dtype=np.float32)

    print(f"\nData summary:")
    print(f" Total prompts: {len(prompts)}")
    print(f" Total responses: {len(responses)}")
    print(f" Safety entropy — min: {safety_entropies.min():.4f}, max: {safety_entropies.max():.4f}, mean: {safety_entropies.mean():.4f}")
    print(f" Safety score   — min: {safety_scores.min():.4f}, max: {safety_scores.max():.4f}, mean: {safety_scores.mean():.4f}")
    print(f" Joint risk target — min: {joint_risk_targets.min():.4f}, max: {joint_risk_targets.max():.4f}, mean: {joint_risk_targets.mean():.4f}")

    non_empty_responses = sum(1 for r in responses if r.strip())
    print(f" Samples with responses: {non_empty_responses}/{len(responses)}")

    return prompts, responses, safety_entropies, safety_scores, joint_risk_targets

def load_benign_data(data_path):
    """
    Load benign clustered data
    benign data has no safety metrics and responses
    only load the prompts and hardcode the safety entropy and safety score to 0
    """
    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} samples")

    prompts = [item['prompt'] for item in data]
    scores = [0.0 for item in data]

    return prompts, scores

# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description='Generate and cache hidden states for safety probe training')
    parser.add_argument('--harmful_data_path', type=str, default='data/clustering_results/meta-llama/Llama-3.2-3B-Instruct/harmful_test.json', help='Path to harmful clustered data JSON file')
    parser.add_argument('--benign_data_path', type=str, default='data/datasets/xTRam1_benign_test.json', help='Path to benign clustered data JSON file')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='HuggingFace model name')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for extraction')
    parser.add_argument('--cache_dir', type=str, default='feature_cache', help='Directory to save cached hidden states')
    parser.add_argument('--device', type=str, default='mps', choices=['cuda', 'mps', 'cpu'], help='Device to use for computation')
    args = parser.parse_args()

    cache_path = os.path.join(args.cache_dir, f"hidden_states.pt")

    device = torch.device(args.device)

    print("=" * 80)
    print("Feature Generation and Caching")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Device: {device}")
    print(f"Data path: {args.harmful_data_path}")
    print(f"Cache dir: {args.cache_dir}")
    print("=" * 80)

    # Load data
    print("\nLoading harmful data...")
    prompts, responses, safety_entropies, safety_scores, joint_risk_targets = load_data(args.harmful_data_path)




    combined_prompts, combined_scores = load_benign_data(args.benign_data_path)

    #concatenate the prompts and safety entropy and safety score
    combined_prompts = prompts + combined_prompts

    combined_safety_entropies = np.concatenate([safety_entropies, combined_scores])
    combined_safety_scores = np.concatenate([safety_scores, combined_scores])
    combined_safety_entropies = np.concatenate([safety_entropies, combined_scores])

    n_samples_tbg = len(combined_prompts)
    print(f"\nProcessing {n_samples_tbg} samples for TBG...")

    
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type in ['cuda', 'mps'] else torch.float32,
        device_map="auto",
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print(f"✓ Model loaded successfully")
    print(f" Number of layers: {len(model.model.layers)}")

    print("\n" + "=" * 80)
    print("Extracting hidden states...")
    print("=" * 80)

    tbg_hidden_states = extract_hidden_states_tbg(
        combined_prompts, model, tokenizer, device,
        batch_size=args.batch_size
    )


    print("\n" + "=" * 80)
    print("Saving to cache...")
    print("=" * 80)
    save_metadata = {
        'model_name': args.model_name,
        'n_samples_tbg': n_samples_tbg,
    }
    hidden_states_dict = {'tbg': tbg_hidden_states}
    save_hidden_states(hidden_states_dict, cache_path, metadata=save_metadata)


    # Save labels as separate files
  

    entropy_path_tbg = cache_path.replace('.pt', '_safety_entropy_labels_tbg.npy')
    score_path_tbg = cache_path.replace('.pt', '_safety_score_labels_tbg.npy')
    joint_risk_path_tbg = cache_path.replace('.pt', '_joint_risk_target_labels_tbg.npy')

    np.save(entropy_path_tbg, combined_safety_entropies)
    np.save(score_path_tbg, combined_safety_scores)
    np.save(joint_risk_path_tbg, joint_risk_targets)



    # Summary
    print("\n" + "=" * 80)
    print("Feature extraction complete!")
    print("=" * 80)
    print(f"TBG hidden states: {len(tbg_hidden_states)} layers, shape {tbg_hidden_states[0].shape}")
    print("=" * 80)
    print("\n✓ All done!")


if __name__ == '__main__':
    main()
