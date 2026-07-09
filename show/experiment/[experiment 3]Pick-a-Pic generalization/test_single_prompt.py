import json
import os
import requests

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

    response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()

    # Extract the response text and split into lines
    alternatives_text = result['choices'][0]['message']['content'].strip()
    alternatives = [line.strip() for line in alternatives_text.split('\n') if line.strip()]

    # Filter out numbered lines
    alternatives = [alt.lstrip('0123456789.-) ') for alt in alternatives]

    return alternatives[:4]

# Test with one example prompt
test_prompt = "deathly portal to the abyss, ultra detailed, warm interior light, cinematic shot, photorealistic, octane render, high definition, fine details, sinister tones, 8 k, mcbess mood,"

print("Original prompt:")
print(test_prompt)
print("\nGenerating alternatives...")

alternatives = call_deepseek(test_prompt)

print("\nAlternative prompts:")
for i, alt in enumerate(alternatives, 1):
    print(f"{i}. {alt}")
