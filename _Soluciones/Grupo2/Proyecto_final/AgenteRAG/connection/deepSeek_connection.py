from openai import OpenAI

client = OpenAI(
    api_key="YOUR_DEEPSEEK_API_KEY",
    base_url="https://api.deepseek.com"
)

def get_completion(prompt: str) -> str:
    try:
        response = client.responses.create(
            model="deepseek-chat",
            input=prompt
        )
        return response.output_text
    except Exception as e:
        print("Error:", e)
        return "Error during completion"

def main():
    print(get_completion("What is the capital of France?"))

if __name__ == "__main__":
    main()
