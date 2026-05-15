import json
import httpx

OLLAMA_URL = "http://localhost:11434/api/chat"
model_to_use = "codegemma"

def get_completion(prompt: str, model: str) -> str:
    try:
        with httpx.Client() as client:
            with client.stream(
                "POST",
                OLLAMA_URL,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                },
            ) as response:
                response.raise_for_status()

                full_content = ""

                for line in response.iter_lines():
                    if not line:
                        continue

                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        full_content += content
                    except json.JSONDecodeError:
                        pass

                return full_content

    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while fetching the completion."

def main():
    prompt = "What is the capital of France?"
    output = get_completion(prompt, model_to_use)
    print(f"prompt: {prompt}")
    print(f"output: {output}")

if __name__ == "__main__":
    main()
