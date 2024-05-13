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
def extract_audio(video_file, video_file_extension="mp4"):
    # create directory ./audios/ if it does not exist
    os.makedirs("audios", exist_ok=True)
    # Extract the audio from the video
    if os.path.exists(video_file + ".mp3"):
        print(f"Audio file {video_file}.mp3 already exists.")
        return
    try :
        video = moviepy.editor.VideoFileClip(video_file + "." + video_file_extension)
        audio = video.audio
        audio.write_audiofile("./audios/" + video_file + ".mp3")
        print(f"Audio file {video_file}.mp3 has been succesfully saved.")
    except Exception as e:
        print(f"Error while extracting the audio from the video : {e}")
