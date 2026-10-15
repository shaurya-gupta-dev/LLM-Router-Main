from openai import OpenAI

client = OpenAI(
    api_key = "sk-apx78b24212c93c96f5e0d3a72139b5bbff73e8309bad2d273",
    base_url = "https://api.apinex.bond/v1",
)

while True:
    user = input("You: ")
    response = client.chat.completions.create(
        model="free/gemini-3.8-flash",
        messages=[
            {"role": "system", "content": user}
        ]
    )

    print("Model: ", response.choices[0].message.content)