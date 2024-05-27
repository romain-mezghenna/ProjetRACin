# ProjetRACin: Alcohol Representation Detection in Films

## Table of Contents

1. [Introduction](#introduction)
2. [Technologies Used](#technologies-used)
    - [Prerequisites](#prerequisites)
    - [Libraries Used](#libraries-used)
3. [Software Architecture](#software-architecture)
    - [Overall Architecture](#overall-architecture)
4. [Getting Started](#getting-started)
    - [Downloading the Prototype](#downloading-the-prototype)
    - [Installing Dependencies](#installing-dependencies)
    - [Running the Program](#running-the-program)
5. [Results Storage](#results-storage)
6. [Contributing](#contributing)
7. [License](#license)
8. [Acknowledgments](#acknowledgments)

## Introduction

This repository hosts the prototype for detecting alcohol representations in films, developed as part of the RACin project (Representations of Alcohol Consumption in French Cinema since the 1960s). The project aims to use AI models to identify instances of alcohol depiction in movies.

## Technologies Used

### Prerequisites

- **Ollama:** Required for running local machine learning models. [Download Ollama](https://ollama.com)
- **Python:** Version 3.12 or higher. [Download Python](https://python.org)

### Libraries Used

- **MoviePy:** For video editing and manipulation.
- **Faster Whisper:** For video transcription, optimized for performance. [More info](https://github.com/SYSTRAN/faster-whisper)
- **Imageio-ffmpeg and Imageio:** For video reading and writing functionalities.
- **Ultralytics:** For object and scene detection using the YOLOv8 model. [More info](https://docs.ultralytics.com/models/yolov8)
- **Ollama:** For interacting with the Ollama program, including downloading and customizing models. [More info](https://github.com/ollama/ollama-python)
- **Langchain and Langchain-Community:** For integrating AI tools like Ollama, OpenAI, and Gemini.

## Software Architecture

### Overall Architecture

The prototype uses a class-based architecture designed for flexibility and ease of integration with various model providers. The abstract factory design pattern allows switching between different model providers without modifying the existing codebase.

For a detailed view of the class diagram, refer to `class_diagram.jpeg` in the `docs/` repository.

For a detailed view of an example of utilisation, refer to `sequence_diagram.jpeg` in the `docs/` repository.

For an example of analysis please consul `main.py`.
## Getting Started

### Downloading the Prototype

Clone the repository from GitHub:

```bash
git clone https://github.com/yourusername/ProjetRACin.git
cd ProjetRACin
```

### Installing Dependencies

Install the required Python libraries using the command to install from the `requirements.txt` file.

**Note for Windows users:** If you encounter errors, install Visual Studio Build Tools. [Download here](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

Select : 
- [x] MSM de C++ 2022 Redistribuable
- [x] MSVC v143 - VS 2022 C++ x64/x86 Build Tool (latest version)
- [x] Kit de développement logiciel (SDK) Windows 10/11 (same as your version of Windows)

 

### Running the Program

To run the prototype, use the command to execute `main.py` with the video path as an argument.

Replace `<video_path>` with the path to the video you want to analyze. For example, if your video file is named `video.mp4`, use the appropriate command to run the program.

Most video formats are supported. If you encounter errors with a specific video file, please open an issue on the GitHub project page.

## Results Storage

- **Transcription Results:** Stored in `transcript_results` with the format `video_path-model_name.json`.
- **Object Detection Results:** Stored in `object_detection_results` with the format `alcohol_detections_video_path.json`. Images for analysis are stored in `object_detection_results/images/`.
- **Overall Analysis Results:** Stored in the current directory with the format `video_path.json`.

Storing intermediate results helps avoid redundant computations, saving processing power and time.

## Contributing

We welcome contributions from the community. To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push to the branch.
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Authors

Romain Mezghenna : **romain24.mezghenna01@gmail.com**

Alexis Chartier : **alexischartier30130pse@gmail.com**

## Acknowledgments

- Erwan Pointeau Lagadec (Requester)
- Christophe Fiorio (Supervisor)

Special thanks to all the contributors and the open-source community for their valuable tools and libraries.