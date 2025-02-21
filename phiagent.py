from pathlib import Path
from phi.agent import Agent
from phi.model.openai import OpenAIChat

# 1. OCR Agent
text_agent = Agent(
    name="Text Agent",
    role="Extract ALL visible text from this image and its metadata like contrast, font size, and style from images.",
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Extract text and provide detailed metadata."],
    show_tool_calls=True,
    markdown=True,
)

# 2. Overlay Agent
overlay_agent = Agent(
    name="Overlay Agent",
    role="Classify extracted text as overlay or background based on features and contextual meaning.",
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Use the metadata to classify text into overlay or background."],
    show_tool_calls=True,
    markdown=True,
)

# 3. Validation Agent
validation_agent = Agent(
    name="Validation Agent",
    role="From the outputs of Text Agent and Overlay Agent, Return the Overlay text and provide final output. Prioritize Overlay Agent.",
    model=OpenAIChat(id="gpt-4o"),
    instructions=["Validate the extracted overlay text from Overlay agent with Text Agent's output and provide the final output."],
    show_tool_calls=True,
    markdown=True,
)

# Combined Agent Team
agent_team = Agent(
    team=[text_agent, overlay_agent,validation_agent],
    instructions=["Omit any background text from the App/ Website interface", "Ensure output is clear and structured.","Return empty string if there is no Overlay text in the background."],
    show_tool_calls=True,
    markdown=True,
)

# Pipeline Function
def extract_overlay_text(frame_path):
    """
    Pipeline to extract overlay text from a TikTok video frame.
    Input: Path to the video frame image.
    Output: String containing overlay text.
    """
    # Ensure the file exists
    image_path = Path(frame_path)
    if not image_path.exists():
        raise FileNotFoundError(f"The file {frame_path} does not exist. Please provide a valid path.")

    agent_team.print_response(
        "Extract Overlay Text",
        images=[str(image_path)],
    )

