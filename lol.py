from openai import OpenAI, RateLimitError
import os, json
client = OpenAI(api_key= "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")

try:
    r = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":"ping"}],
        max_tokens=1
    )
    print("✅ credit available:", json.dumps(r.usage.dict(), indent=2))
except RateLimitError as e:
    print("❌ still no quota:", e)
