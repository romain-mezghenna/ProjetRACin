import time
import faster_whisper
from faster_whisper import WhisperModel
import videoEditor
import os
import torch

# Function to transcribe the video locally on a CPU/GPU with the Whisper model
# The function takes the model name and path to the video file as input
# Optional : vad_filter parameter to enable/disable voice activity detection filter (default is True)
# Optional : language parameter to specify the language of the video (default is None) like "fr" for French, "en" for English, etc. see : https://github.com/openai/whisper
# The function writes the transcribed text to a file named "video_path-model_name.txt" in the directory ./transcripts/
# Model name can be "small", "medium", or "large-v3" or "distil-large-v3"
def transcribe_audio(model_name, video_path,vad_filter=True, language=None):
    # Test if the video file exists
    if not os.path.exists(video_path):
        print(f"Video file '{video_path}' does not exist.")
        return
    
    # Test if the audio file associated with the video exists in ./audios/ directory
    audio_path = f"audios/{video_path}.mp3"
    if not os.path.exists(audio_path):
        print(f"Audio file '{audio_path}' does not exist. Extracting audio from the video.")
        videoEditor.extract_audio(video_path)
        print(f"Audio extracted successfully.")

    # Test if the model name is valid
    if model_name not in faster_whisper.available_models():
        print(f"Model '{model_name}' is not available.")
        return
    
    # Test if the transcript file already exists
    if os.path.exists(f"transcripts/{os.path.basename(video_path)}-{model_name}.txt"):
        print(f"Transcript file 'transcripts/{os.path.basename(video_path)}-{model_name}.txt' already exists.")
        return
    
    # Load the Whisper model
    # Test if the user computer has a GPU with CUDA support
    if not torch.cuda.is_available():
        # Run on the CPU
        print("CUDA is not available on this machine. Running on CPU.")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    else:
        # Run on the GPU (CUDA - Nvidia GPU required)
        print("CUDA is available on this machine. Running on GPU.")
        model = WhisperModel(model_name, device="cuda", compute_type="float16")

    print(f"Model {model_name} loaded successfully.")

    # Get the start time
    start_time = time.time()

    # If the language is specified, set it in the model

    segments, info = model.transcribe(
        "./audios/" + video_path + ".mp3", beam_size=5, vad_filter=vad_filter,language=language
    )

    print(f"Transcription starting")

    print(
        "Detected language '%s' with probability %f"
        % (info.language, info.language_probability)
    )

    # Create a directory to store the transcripts
    os.makedirs("transcripts", exist_ok=True)

    # Write the transcribed text to a file
    with open(f"transcripts/{os.path.basename(video_path)}-{model_name}.txt", "a") as file:
        for segment in segments:
            print("[%.2fs -> %.2fs] : %s" % (segment.start, segment.end, segment.text))
            # Write a new line to the file
            file.write("[%.2fs -> %.2fs] : %s \n" % (segment.start, segment.end, segment.text))

    # Get the end time
    end_time = time.time()

    # Calculate the total time
    print(f"Transcription done in {end_time - start_time:.2f} seconds.")
    # Print the duration of the video
    print(f"Duration of the video: {info.duration:.2f} seconds.")
    # Print the duration with vad filter
    print(f"Duration of the video with vad filter: {info.duration_after_vad:.2f} seconds.")

