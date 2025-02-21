import os
import json
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import base64


load_dotenv()

def init_llm():
    return ChatOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o",
        temperature=0.1,
        max_tokens=500,
        model_kwargs={"response_format": {"type": "json_object"}}
    )


class TextAgent:
    def __init__(self, model):
        self.model = model
    def extract_all_text(self, image_path):
        # Encode image to base64
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = (
            "Extract ALL visible text from this image. Return valid string response. Include the word 'json' in your response. Image: {image}"
        )
        
        messages = [
            HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"  
                        }
                    }
                ]
            )
        ]

        response = self.model.invoke(messages).content
        print(response)
        return json.loads(response)
    

# Overlay Agent: Extracts only overlay text
class OverlayAgent:
    def __init__(self, model):
        self.model = model
    def extract_overlay_text(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        prompt = (
            "Extract ONLY overlay/deliberately added text. Understand that it is always horizontally oriented. Return JSON with {{\"overlay_text\": []}}. Use word 'json'. Image: {image}"
        )
        
        messages = [
            HumanMessage(content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"  
                        }
                    }
                ]
            )
        ]
        
        response = self.model.invoke(messages).content
        print(response)
        return json.loads(response)

class ValidationAgent:
    def __init__(self, model):
        self.model = model

    def validate_overlay_text(self, rf_output, text_agent_output, overlay_agent_output):
        prompt = PromptTemplate(
            input_variables=["rf_text", "text_agent_text", "overlay_agent_text"],
            template="""Analyze these results and return final overlay text in JSON format. For context, Random Forest Output and Overlay Agent outputs are referring to Overlay text while Text agent output refers to all text in the image.:
            Random Forest output: {rf_text}
            Text Agent output: {text_agent_text}
            Overlay Agent output: {overlay_agent_text}
            Output format: {{"overlay_text": "text"}}. Include 'json' in response."""
        )
        formatted_prompt = prompt.format(
            rf_text=rf_output,
            text_agent_text=json.dumps(text_agent_output),
            overlay_agent_text=json.dumps(overlay_agent_output)
        )
        
        # Explicit JSON instruction
        messages = [HumanMessage(content=formatted_prompt + "\nRespond with JSON containing overlay_text.")]
        
        raw_response = self.model.invoke(messages).content
        print(raw_response)
        return json.loads(raw_response)

'''
# Validation Agent: Validates RandomForest output, TextAgent, and OverlayAgent results
class ValidationAgent:
    def __init__(self, model):
        self.model = model

    def validate_overlay_text(self, rf_output, text_agent_output, overlay_agent_output):
        """
        Validate the overlay text with explicit JSON instruction
        """
        prompt = PromptTemplate(
            input_variables=["rf_text", "text_agent_text", "overlay_agent_text"],
            template=(
                "Analyze these text detection results and respond in JSON format:\n"
                "1. Random Forest: {rf_text}\n"
                "2. Text Agent: {text_agent_text}\n"
                "3. Overlay Agent: {overlay_agent_text}\n\n"
                "Return JSON with: {{\"overlay_text\": \"text\", \"confidence\": \"level\"}}\n"
                "Use 'NONE' if no valid overlay text. Always include the word 'json' in your response."
            )
        )
        try:
            formatted_prompt = prompt.format(
                rf_text=rf_output,
                text_agent_text=text_agent_output,
                overlay_agent_text=overlay_agent_output
            )
            
            # Ensure the message contains 'json' explicitly
            messages = [
                HumanMessage(content=formatted_prompt + "\nRemember to use JSON format.")
            ]
            
            raw_response = self.model.invoke(messages).content
            return json.loads(raw_response)

        except json.JSONDecodeError:
            return {"error": "Invalid JSON response"}
        except Exception as e:
            return {"error": str(e)}
'''

text_agent = TextAgent(init_llm())
overlay_agent = OverlayAgent(init_llm())
validation_agent = ValidationAgent(init_llm())
