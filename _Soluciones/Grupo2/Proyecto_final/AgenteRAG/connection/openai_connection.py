from openai import OpenAI

api_key = "sk-proj-"
model = "gpt-4.1-mini"   # o "gpt-4.1" / "gpt-5.1"

client = OpenAI(api_key=api_key)


def get_completion(prompt: str) -> str:
    try:
        response = client.responses.create(
            model=model,
            input=prompt
        )

        # La salida ahora está en response.output_text
        return response.output_text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "An error occurred while fetching the completion."


def main():
    prompt = "What is the capital of France?"
    completion = get_completion(prompt)
    print(completion)

if __name__ == "__main__":
    main()
