# library containts functions to edit video files

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor
import os

# Function to cut the video from start to end time
# Input : video_file : the path to the video file
#         start_time : the start time in seconds
#         end_time : the end time in seconds
# Output : None
def cut_video(video_file, start_time, end_time):
    # Extract the video from start to end time
    try :
        ffmpeg_extract_subclip(str(video_file), start_time, end_time, targetname=str(video_file) + "-cut_" + str(start_time) + "_" + str(end_time) + ".mp4")
        print(f"Video : {video_file} cut from {start_time} to {end_time} seconds has been succesfully saved.")
    except Exception as e:
        print(f"Error while cutting the video : {e}")


# Function to extract the audio from the video
# Input : video_file : the path to the video file
#         video_file_extension : the extension of the video file (default is mp4)
# Output : None
def extract_audio(video_file):
    # create directory ./audios/ if it does not exist
    os.makedirs("audios", exist_ok=True)
    # Extract the audio from the video
    if os.path.exists(video_file + ".mp3"):
        print(f"Audio file {video_file}.mp3 already exists.")
        return
    try :
        video = moviepy.editor.VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile("./audios/" + video_file + ".mp3")
        print(f"Audio file {video_file}.mp3 has been succesfully saved.")
    except Exception as e:
        print(f"Error while extracting the audio from the video : {e}")

# Function to extract an image from the video at a specific time
# Input : video_file : the path to the video file
#         time : the time in seconds
# Output : the path to the image file
def extract_image(video_file, time):
    # create directory ./images/ if it does not exist
    os.makedirs("images", exist_ok=True)
    # Extract the image from the video
    if os.path.exists(video_file + "-" + str(time) + ".png"):
        print(f"Image file {video_file}-{time}.png already exists.")
        return
    if not os.path.exists(video_file):
        print(f"Video file {video_file}.mp4 does not exist.")
        return
    try :
        video = moviepy.editor.VideoFileClip(video_file)
        video.save_frame("./images/" + video_file + "-" + str(time) + ".png", t=time)
        print(f"Image file {video_file}-{time}.png has been succesfully saved.")
        return "./images/" + video_file + "-" + str(time) + ".png"
    except Exception as e:
        print(f"Error while extracting the image from the video : {e}")


# Function to get the duration of a video
# Input : video_file : the path to the video file
# Output : the duration of the video in seconds
def get_video_duration(video_file):
    # Get the duration of the video
    try :
        video = moviepy.editor.VideoFileClip(video_file)
        duration = video.duration
        return duration
    except Exception as e:
        print(f"Error while getting the duration of the video : {e}")
        return 0