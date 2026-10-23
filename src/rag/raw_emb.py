import os
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# response = client.chat.completions.create(
#     model="nvidia/nemotron-3-ultra-550b-a55b:free",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "what is the capital of France?"},
#     ],
# )

# print(response.choices[0].message.content)


response = client.embeddings.create(
    input="Input string goes here", model="nvidia/nemotron-3-embed-1b:free"
)

# print(response)
print(type(response.data[0].embedding))
print(len(response.data[0].embedding))
