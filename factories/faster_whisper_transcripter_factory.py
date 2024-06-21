import time
import os
import faster_whisper
from faster_whisper import WhisperModel
from models.transcript_model import TranscriptionResult, Segment
from utils.video_editor import extract_audio
from factories.transcripter_factory import TranscripterFactory, Transcripter


class FasterWhisperTranscripterFactory(TranscripterFactory):
    def __init__(self, model_name : str):
        super().__init__(model_name)
    
    def create_transcripter(self) -> Transcripter:
        class FasterWhisperTranscripter(Transcripter):

            def __init__(self, model_name : str):
                super().__init__(model_name)
                

            # Function to transcribe the video locally on a CPU/GPU with the Whisper model
            # The function takes the model name and path to the video file as input
            # Optional : vad_filter parameter to enable/disable voice activity detection filter (default is True)
            # Optional : language parameter to specify the language of the video (default is None) like "fr" for French, "en" for English, etc. see : https://github.com/openai/whisper
            # The function writes the transcribed text to a file named "video_path-self.model_name.txt" in the directory ./transcripts/
            # Model name can be "small", "medium", or "large-v3" or "distil-large-v3"
            def transcribe(self,video_path : str, vad_filter=True, language=None) -> TranscriptionResult:

                # Check if the transcript file already exists
                transcript_file = f"transcript_results/{os.path.basename(video_path)}-{self.model_name}.json"
                if os.path.exists(transcript_file):
                    print(f"Transcript file '{transcript_file}' already exists.")
                    # Load the existing transcript file
                    with open(transcript_file, 'r') as f:
                        # get all the data from the file
                        data = f.read()
                        return TranscriptionResult.model_validate_json(data)

                # Create a directory to store the transcripts
                os.makedirs("transcript_results", exist_ok=True)

                # Check if the video file exists
                if not os.path.exists(video_path):
                    print(f"Video file '{video_path}' does not exist.")
                    return
                
                # Check if the audio file associated with the video exists in ./audios/ directory
                audio_path = f"audios/{os.path.basename(video_path)}.mp3"
                if not os.path.exists(audio_path):
                    print(f"Audio file '{audio_path}' does not exist. Extracting audio from the video.")
                    extract_audio(video_path)
                    print(f"Audio extracted successfully.")

                # Check if the model name is valid
                if self.model_name not in faster_whisper.available_models():
                    print(f"Model '{self.model_name}' is not available.")
                    return
                
            
                # Load the Whisper model
                model = WhisperModel(self.model_name,device="cpu")

                print(f"Model {self.model_name} loaded successfully.")

                # Get the start time
                start_time = time.time()

                # Transcribe the audio
                segments, info = model.transcribe(
                    audio_path, beam_size=5, vad_filter=vad_filter, language=language
                )

                print(f"Transcription starting")
                print(f"Detected language '{info.language}' with probability {info.language_probability}")

                

                # Prepare results
                results = []
                for segment in segments:
                    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] : {segment.text}")
                    results.append(Segment(start_time=segment.start, end_time=segment.end, sentence=segment.text))

                # Get the end time
                end_time = time.time()
                duration = end_time - start_time

                # Create the transcription result
                transcription_result = TranscriptionResult(
                    video_file=video_path,
                    duration=duration,
                    model=self.model_name,
                    results=results,
                    language=info.language,
                    language_probability=info.language_probability
                )

                # Save the transcription result to a JSON file
                with open(transcript_file, "w",encoding="utf-8") as f:
                    transcription_result_json = transcription_result.model_dump_json(indent=2)
                    f.write(transcription_result_json)

                return transcription_result
        return FasterWhisperTranscripter(model_name=self.model_name)



