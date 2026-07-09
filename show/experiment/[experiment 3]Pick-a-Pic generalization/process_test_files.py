import json
import os
import requests
import time
import sys
from tqdm import tqdm

# DeepSeek API configuration
API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
API_URL = "https://api.deepseek.com/v1/chat/completions"

def call_deepseek(original_prompt):
    """Call DeepSeek API to generate 4 alternative prompts"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    system_prompt = """You are a helpful assistant that generates alternative phrasings of image generation prompts.
Given an image generation prompt, generate exactly 4 alternative prompts that have the same meaning but use different wording.
Keep the artistic style, technical specifications, and all key details.
Output only the 4 alternative prompts, one per line, without numbering or extra text."""

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Generate 4 alternative versions of this image generation prompt:\n\n{original_prompt}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        # Extract the response text and split into lines
        alternatives_text = result['choices'][0]['message']['content'].strip()
        alternatives = [line.strip() for line in alternatives_text.split('\n') if line.strip()]

        # Filter out numbered lines (like "1.", "2.", etc.)
        alternatives = [alt.lstrip('0123456789.-) ') for alt in alternatives]

        # Take exactly 4 alternatives
        alternatives = alternatives[:4]

        # If we got fewer than 4, pad with the original
        while len(alternatives) < 4:
            alternatives.append(original_prompt + " (variant)")

        return alternatives

    except Exception as e:
        print(f"Error calling DeepSeek API: {e}")
        # Return default alternatives in case of error
        return [
            original_prompt + " (variant 1)",
            original_prompt + " (variant 2)",
            original_prompt + " (variant 3)",
            original_prompt + " (variant 4)"
        ]

def process_file(input_path, output_path):
    """Process a test file with flat array structure"""
    print(f"\nProcessing {input_path}...")

    # Load the data
    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} entries")

    # Extract unique prompts
    unique_prompts = {}
    for item in data:
        prompt = item['prompt']
        if prompt not in unique_prompts:
            unique_prompts[prompt] = None

    print(f"Found {len(unique_prompts)} unique prompts")

    # Generate alternatives for each unique prompt
    for i, prompt in enumerate(tqdm(list(unique_prompts.keys()), desc="Generating alternatives")):
        alternatives = call_deepseek(prompt)
        unique_prompts[prompt] = alternatives

        # Rate limiting
        if (i + 1) % 10 == 0:
            time.sleep(1)
        else:
            time.sleep(0.2)

    # Add alternatives to all entries
    print("Adding alternatives to entries...")
    for item in data:
        prompt = item['prompt']
        item['alternative_prompts'] = unique_prompts[prompt]

    # Save results
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Done! Processed {len(data)} entries with {len(unique_prompts)} unique prompts")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python process_test_files.py <1|2>")
        sys.exit(1)

    file_num = sys.argv[1]
    input_path = f'/mnt/data/user/fengxian.ji/AUCTION/benchmark/ImageReward/test_{file_num}.json'
    output_path = f'/mnt/data/user/fengxian.ji/AUCTION/benchmark/ImageReward/test_{file_num}_with_alternatives.json'

    process_file(input_path, output_path)
