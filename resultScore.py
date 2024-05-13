import os
from jiwer import wer, cer


def load_transcript(file_path):
    """Load the transcript from the specified file path."""
    print(f"Loading transcript from {file_path}")
    transcript = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as file:
            # Read each line in the file
            for line in file:
                line = line.strip()  # Strip whitespace from the line

                # Skip empty lines
                if not line:
                    continue

                # Split the line into timestamp and sentence using " : "
                parts = line.split(" : ", maxsplit=1)
                if len(parts) == 2:
                    # If there is a sentence part, add it to the list
                    sentence = parts[1].strip()
                    transcript.append(sentence)

            # Combine sentences into a single string
            full_transcript = " ".join(transcript)

            print(f"Loaded transcript length: {len(full_transcript)} characters")
            return full_transcript

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def evaluate_transcriptions(human_transcript, ml_transcript):
    """Evaluate ML transcription against human transcription."""
    if not human_transcript or not ml_transcript:
        print("Skipping evaluation due to empty transcript.")
        return None

    # Calculate WER and CER
    wer_score = wer(human_transcript, ml_transcript)
    cer_score = cer(human_transcript, ml_transcript)

    # Return results as a dictionary
    return {"wer": wer_score, "cer": cer_score}


def main():
    # Directory containing the transcripts
    transcripts_dir = "./transcripts/"

    # Initialize a list to store evaluation results
    evaluation_results = []

    # Iterate through files in the transcripts directory
    for file_name in os.listdir(transcripts_dir):
        if file_name.endswith("-human.txt"):
            # Extract audio file name without extension
            audio_file_name = file_name[:-10]  # Remove "-human.txt"

            # Load human transcript
            human_transcript_path = os.path.join(transcripts_dir, file_name)
            human_transcript = load_transcript(human_transcript_path)

            if human_transcript:
                print(f"Evaluating transcripts for audio file: {audio_file_name}")

                # Iterate through ML model transcripts
                for model in ["tiny", "small", "medium", "base", "large"]:
                    ml_file_name = f"{audio_file_name}-{model}.txt"
                    ml_transcript_path = os.path.join(transcripts_dir, ml_file_name)

                    # Check if the ML transcript file exists
                    if os.path.exists(ml_transcript_path):
                        # Load ML transcript
                        ml_transcript = load_transcript(ml_transcript_path)

                        if ml_transcript:
                            # Evaluate the ML transcript against the human transcript
                            metrics = evaluate_transcriptions(
                                human_transcript, ml_transcript
                            )

                            if metrics:
                                print(f"Model: {model}")
                                print(f"WER: {metrics['wer']:.2f}")
                                print(f"CER: {metrics['cer']:.2f}")

                                # Append results to the list
                                evaluation_results.append(
                                    {
                                        "audio_file": audio_file_name,
                                        "model": model,
                                        "wer": metrics["wer"],
                                        "cer": metrics["cer"],
                                    }
                                )
                        else:
                            print(f"Failed to load ML transcript for model: {model}")
                    else:
                        print(f"No ML transcript found for model: {model}")
            else:
                print(f"Failed to load human transcript for file: {file_name}")
        else:
            print(
                f"Skipping file: {file_name} as it doesn't match '-human.txt' pattern."
            )

    # Save evaluation results to a file
    results_file = "evaluation_results.txt"
    with open(results_file, "w") as f:
        for result in evaluation_results:
            f.write(f"Audio File: {result['audio_file']}\n")
            f.write(f"Model: {result['model']}\n")
            f.write(f"WER: {result['wer']:.2f}\n")
            f.write(f"CER: {result['cer']:.2f}\n")
            f.write("\n")
    print(f"Results have been saved to {results_file}")


if __name__ == "__main__":
    main()
