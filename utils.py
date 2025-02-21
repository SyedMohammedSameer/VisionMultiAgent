import os
import subprocess

def extract_frames(video_path, output_dir, fps=1):
    os.makedirs(output_dir, exist_ok=True)

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    command = [
        "ffmpeg",
        "-i", video_path,       
        "-vf", f"fps={fps}",    
        "-q:v", "2",            
        output_pattern          
    ]

    try:

        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}")

    frame_paths = sorted([
        os.path.join(output_dir, frame)
        for frame in os.listdir(output_dir)
        if frame.startswith("frame_") and frame.endswith(".jpg")
    ])

    return frame_paths

if __name__ == "__main__":
    video_path = r"your_video_file_path_here.mp4"
    output_dir = r"intended_output_directory_here"
    fps = 1  
    frame_paths = extract_frames(video_path, output_dir, fps)

    for frame_path in frame_paths:
        print(frame_path)