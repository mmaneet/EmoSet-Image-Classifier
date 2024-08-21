import base64
import json
import requests

# OpenAI API Key
api_key = "sk-proj-WIpQplHLRUX1npiRWK78T3BlbkFJxEgL8KY9XTMFT474BjJx"
# Path to your image
image_path = r"C:\Users\manee\EmoSet-118K-7\image\amusement\amusement_00000.jpg"


# Function to encode the image
def encode_image(image_path_p):
    with open(image_path_p, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def captioner(image_path_p):
    # Getting the base64 string
    base64_image = encode_image(image_path_p)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Generate a descriptive caption for this image that can help a model identify the emotion present."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    response_json = response.json()

    content_string = response_json["choices"][0]["message"]["content"]
    print(response.json())

    # Define the file path where you want to save the JSON
    json_file_path = "api_response.json"

    # Write the JSON response to a file
    with open(json_file_path, 'w') as json_file:
        json.dump(response_json, json_file, indent=4)

    print(f"API response saved to {json_file_path}")

    return content_string


sample = captioner(image_path)


def classifier(text):
    API_URL = "https://api-inference.huggingface.co/models/michellejieli/emotion_text_classifier"
    headers = {"Authorization": "Bearer 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"}

    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    output = query({
        "inputs": text,
    })

    print(output)

classifier(sample)
