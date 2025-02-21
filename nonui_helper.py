from groq import Groq
import os
from dotenv import load_dotenv
import base64
import openai
from openai import OpenAI
import requests

load_dotenv()

def analyze_image_local(frame_path: str) -> str:
    """
    Analyzes an image from a local file path to extract overlay text.

    Args:
        frame_path (str): Path to the local image file.

    Returns:
        str: Extracted overlay text or an error message.
    """
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"File not found: {frame_path}")
    api_key = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api_key)

    try:
        with open(frame_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        return f"Error encoding image: {str(e)}"

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract only the visually prominent overlay text superimposed on the image or video, excluding any text naturally part of the background or scene. Return NONE if no overlay text is present. Strictly adhere to not include any background text or context in returned string."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            model="llama-3.2-11b-vision-preview",
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()

        return "No text extracted from the image."

    except Exception as e:
        return f"Error during API call: {str(e)}"
    

def analyze_image_gpt(frame_path: str) -> str:
    """
    Analyzes an image from a local file path to extract overlay text using GPT-4o.

    Args:
        frame_path (str): Path to the local image file.

    Returns:
        str: Extracted overlay text or an error message.
    """
    if not os.path.exists(frame_path):
        raise FileNotFoundError(f"File not found: {frame_path}")
    
    api_key = os.getenv("OPENAI_API_KEY")  # Changed to OpenAI API key
    client = OpenAI(api_key=api_key)

    try:
        with open(frame_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        return f"Error encoding image: {str(e)}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o", 
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Extract only the visually prominent overlay text superimposed on the image or video, "
                                "excluding any text naturally part of the background or scene. "
                                "Return NONE if no overlay text is present. "
                                "Strictly adhere to not include any background text or context in returned string."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,  
            max_tokens=300
        )

        if response.choices and response.choices[0].message:
            return response.choices[0].message.content.strip()

        return "No text extracted from the image."

    except Exception as e:
        return f"Error during API call: {str(e)}"
    
