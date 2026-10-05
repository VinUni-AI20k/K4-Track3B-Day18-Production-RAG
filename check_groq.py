"""Check the configured Groq key without printing it or response error bodies."""

from config import GROQ_API_KEY, GROQ_MODEL
from src.ai_client import create_client


def main():
    print(f"Groq model: {GROQ_MODEL}")
    if not GROQ_API_KEY:
        print("Missing GROQ_API_KEY in .env")
        return 1
    try:
        response = create_client().chat.completions.create(
            model=GROQ_MODEL, messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=2048)
        if not response.choices or not response.choices[0].message.content:
            print("Groq returned no text.")
            return 1
        print("Groq connection PASSED.")
        return 0
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        print(f"Groq connection failed: {type(exc).__name__}; HTTP status={status}")
        print("Check the key, model access, quota and internet connection.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
