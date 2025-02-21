from fastapi import FastAPI, UploadFile
from utils import extract_frames
from nonui_helper import analyze_image_local, analyze_image_gpt
from randomforest import extract_text_features, classify_text
from multiagent import text_agent, overlay_agent, validation_agent
from phiagent import extract_overlay_text
from fastapi.responses import JSONResponse
from langchain_community.chat_models import ChatOpenAI
import numpy as np
import pandas as pd
from typing import List
import cv2
import shutil
import os
import json

app = FastAPI()

@app.post("/overlay-non-ui/")
async def process_overlay_non_ui(video: UploadFile):
    """
    Processes a video for non-UI overlay text extraction using the analyze_image function.

    Args:
        video (UploadFile): Uploaded video file.

    Returns:
        dict: Extracted overlay text for each frame in the video.
    """
    # Save video locally
    video_dir = "Data/raw/temp_videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, video.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # Extract frames (1 frame per second)
    frames_dir = "Data/frames/test2"
    os.makedirs(frames_dir, exist_ok=True)
    frames: List[str] = extract_frames(video_path, output_dir=frames_dir, fps=1)
    # Analyze each frame for overlay text
    overlay_texts = []
    for i, frame_path in enumerate(frames):
        overlay_text = analyze_image_gpt(frame_path)
        overlay_texts.append({"frame": i + 1, "text": overlay_text})

    return {"overlay_texts": overlay_texts}


@app.post("/overlay-ui-rf/")
async def randomforest_processing(video: UploadFile):
    """Process video using only RandomForest"""
    video_dir = "Data/raw/temp_videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, video.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    frames_dir = "Data/frames/rf_frames"
    os.makedirs(frames_dir, exist_ok=True)
    frames = extract_frames(video_path, output_dir=frames_dir, fps=1)
    
    rf_results = []
    model_path = r"models\random_forest_model.pkl"

    for i, frame_path in enumerate(frames):
        try:
            features_df = extract_text_features(frame_path)
            rf_output = classify_text(features_df, model_path)
            rf_results.append({"frame": i+1, "rf_text": rf_output})
        except Exception as e:
            rf_results.append({"frame": i+1, "error": str(e)})

    return {"randomforest_results": rf_results}

@app.post("/overlay-ui-ma/")
async def multiagent_processing(video: UploadFile):
    """Process video using MultiAgent system with debugging"""
    video_dir = "Data/raw/temp_videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, video.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    frames_dir = "Data/frames/ma_frames"
    os.makedirs(frames_dir, exist_ok=True)
    frames = extract_frames(video_path, output_dir=frames_dir, fps=1)
    
    ma_results = []

    for i, frame_path in enumerate(frames):
        try:
            # Debug: Log frame processing start
            print(f"\n=== Processing Frame {i+1} ===")
            print(f"Frame path: {frame_path}")

            # 1. Text Agent Debugging
            print("\n[Text Agent Input]")
            text_agent_output = text_agent.extract_all_text(frame_path)
            print("[Text Agent Raw Output]:", text_agent_output)

            # 2. Overlay Agent Debugging
            print("\n[Overlay Agent Input]")
            overlay_agent_output = overlay_agent.extract_overlay_text(frame_path)
            print("[Overlay Agent Raw Output]:", overlay_agent_output)

            # 3. Validation Agent Debugging
            print("\n[Validation Agent Inputs]")
            print("RF Output:", "")  # Empty as per previous setup
            print("Text Agent Output:", json.dumps(text_agent_output, indent=2))
            print("Overlay Agent Output:", json.dumps(overlay_agent_output, indent=2))

            # Modified validation call with explicit JSON instruction
            validated_output = validation_agent.validate_overlay_text(
                rf_output="",
                text_agent_output=json.dumps(text_agent_output),
                overlay_agent_output=json.dumps(overlay_agent_output)
            )
            print("[Validation Agent Raw Output]:", validated_output)

            # 4. Final Output Processing
            final_text = validated_output.get("overlay_text", "NONE")
            print("\n[Final Result]:", final_text)
            
            ma_results.append({"frame": i+1, "ma_text": final_text})

        except Exception as e:
            error_msg = f"Error processing frame {i+1}: {str(e)}"
            print(error_msg)
            ma_results.append({"frame": i+1, "error": error_msg})
            # Add error type identification
            if "400" in str(e):
                print("!! API Format Error Detected !!")
                print("Possible Fix: Ensure 'json' appears in system message")

    return {"multiagent_results": ma_results}


@app.post("/overlay-ui/")
async def process_overlay_ui(video: UploadFile):
    # Save video locally
    video_dir = "Data/raw/temp_videos"
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, video.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    frames_dir = "Data/frames/test4"
    os.makedirs(frames_dir, exist_ok=True)
    frames: List[str] = extract_frames(video_path, output_dir=frames_dir, fps=1)
    validated_texts = []

    for i, frame_path in enumerate(frames):
        try:
            # Extract text using Random Forest
            model_path = r"models\random_forest_model.pkl"
            features_df = extract_text_features(frame_path)
            rf_output = classify_text(features_df, model_path)

            # Extract text using TextAgent
            text_agent_output = text_agent.extract_all_text(frame_path)

            # Extract text using OverlayAgent
            overlay_agent_output = overlay_agent.extract_overlay_text(frame_path)

            # Validate and combine results
            validated_output = validation_agent.validate_overlay_text(
                rf_output=rf_output,
                text_agent_output=json.dumps(text_agent_output),
                overlay_agent_output=json.dumps(overlay_agent_output)
            )

            # Normalize response format
            final_text = validated_output.get("overlay_text", "NONE")
            validated_texts.append({"frame": i + 1, "validated_text": final_text})

        except Exception as e:
            validated_texts.append({"frame": i + 1, "validated_text": f"Error: {str(e)}"})

    return {"validated_texts": validated_texts}