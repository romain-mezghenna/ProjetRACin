import google.generativeai as genai
import base64
import transcripter

GOOGLE_API_KEY="AIzaSyB0V3zgp8frcOAf88jb4hY8MZwndGKjcl8"

genai.configure(api_key=GOOGLE_API_KEY)



# Load the model
model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

# # Get the video data in base64 format
# video_data = open("video.mp4", "rb").read()
# video_base64 = base64.b64encode(video_data).decode("utf-8")

# Generate the transcript with transcripteur
transcript = transcripter.transcribe_audio("medium", "video")

# Prompt
prompt = (
    "Analyze the video available at this link : https://drive.google.com/file/d/13ZuNFUDJAF26zs65uu8bkOhAmw_qwFhZ/view?usp=sharing and the transcript the describe as precisly as possible the moments where alcolhol is present at the screen (timecodes included). Here's the transcript of the audio of the video : "
    + transcript
)

# Generate the analysis with the Gemini 1.5 Pro model
response = model.generate_content(transcript)

# Print the response
print(response)
