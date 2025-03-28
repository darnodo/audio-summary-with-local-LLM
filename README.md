# Audio Summary with Local LLM

This tool is designed to provide a quick and concise summary of audio and video files. It supports summarizing content either from a local file or directly from YouTube. The tool uses Whisper for transcription and a local version of Llama3 (via Ollama) for generating summaries.

> [!TIP]  
> It is possible to change the model you wish to use.
> To do this, change the `OLLAMA_MODEL` variable, and download the associated model via [ollama](https://github.com/ollama/ollama)

## Features

- **YouTube Integration**: Download and summarize content directly from YouTube.
- **Local File Support**: Summarize audio/video files available on your local disk.
- **Transcription**: Converts audio content to text using Whisper.
- **Summarization**: Generates a concise summary using Llama3 (Ollama).
- **Transcript Only Option**: Option to only transcribe the audio content without generating a summary.
- **Device Optimization**: Automatically uses the best available hardware (MPS for Mac, CUDA for NVIDIA GPUs, or CPU).

## Prerequisites

Before you start using this tool, you need to install the following dependencies:

- Python 3.12 and lower than 3.13
- [Ollama](https://ollama.com) for LLM model management
- `ffmpeg` (required for audio processing)
- [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management

## Installation

### Using uv

Clone the repository and install the required Python packages using [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/damienarnodo/audio-summary-with-local-LLM.git
cd audio-summary-with-local-LLM

# Create and activate a virtual environment with uv
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### LLM Requirement

[Download and install](https://ollama.com) Ollama to carry out LLM Management. More details about LLM models supported can be found on the Ollama [GitHub](https://github.com/ollama/ollama).

Download and use the Llama3 model:

```bash
ollama pull llama3

## Test the access:
ollama run llama3 "tell me a joke"
```

## Usage

The tool can be executed with the following command line options:

- `--from-youtube`: To download and summarize a video from YouTube.
- `--from-local`: To load and summarize an audio or video file from the local disk.
- `--output`: Specify the output file path (default: ./summary.md)
- `--transcript-only`: To only transcribe the audio content without generating a summary.
- `--language`: Select the language to be used for the transcription (default: en)

### Examples

1. **Summarizing a YouTube video:**

   ```bash
   uv run python src/summary.py --from-youtube <YouTube-Video-URL>
   ```

2. **Summarizing a local audio file:**

   ```bash
   uv run python src/summary.py --from-local <path-to-audio-file>
   ```

3. **Transcribing a YouTube video without summarizing:**

   ```bash
   uv run python src/summary.py --from-youtube <YouTube-Video-URL> --transcript-only
   ```

4. **Transcribing a local audio file without summarizing:**

   ```bash
   uv run python src/summary.py --from-local <path-to-audio-file> --transcript-only
   ```

5. **Specifying a custom output file:**

   ```bash
   uv run python src/summary.py --from-youtube <YouTube-Video-URL> --output my_summary.md
   ```

The output summary will be saved in a markdown file in the specified output directory, while the transcript will be saved in the temporary directory.

## Output

The summarized content is saved as a markdown file (default: `summary.md`) in the current working directory. This file includes a title and a concise summary of the content. The transcript is saved in the `tmp/transcript.txt` file.

## Hardware Acceleration

The tool automatically detects and uses the best available hardware:

- MPS (Metal Performance Shaders) for Apple Silicon Macs
- CUDA for NVIDIA GPUs
- Falls back to CPU when neither is available

### Handling Longer Audio Files

This tool can process audio files of any length. For files longer than 30 seconds, the script automatically:

1. Chunks the audio into manageable segments
2. Processes each chunk separately
3. Combines the results into a single transcript

## Sources

- [YouTube Video Summarizer with OpenAI Whisper and GPT](https://github.com/mirabdullahyaser/Summarizing-Youtube-Videos-with-OpenAI-Whisper-and-GPT-3/tree/master)
- [Ollama GitHub Repository](https://github.com/ollama/ollama)
- [Transformers by Hugging Face](https://huggingface.co/docs/transformers/index)
- [yt-dlp Documentation](https://github.com/yt-dlp/yt-dlp)

## Troubleshooting

### ffmpeg not found

If you encounter this error::

```bash
yt_dlp.utils.DownloadError: ERROR: Postprocessing: ffprobe and ffmpeg not found. Please install or provide the path using --ffmpeg-location
```

Please refer to [this post](https://www.reddit.com/r/StacherIO/wiki/ffmpeg/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)

### Audio Format Issues

If you encounter this error:

```bash
ValueError: Soundfile is either not in the correct format or is malformed. Ensure that the soundfile has a valid audio file extension (e.g. wav, flac or mp3) and is not corrupted.
```

Try converting your file with ffmpeg:

```bash
ffmpeg -i my_file.mp4 -movflags faststart my_file_fixed.mp4
```

### Memory Issues on CPU

If you're running on CPU and encounter memory issues during transcription, consider:

1. Using a smaller Whisper model
2. Processing shorter audio segments
3. Ensuring you have sufficient RAM available

### Slow Transcription

Transcription can be slow on CPU. For best performance:

1. Use a machine with GPU or Apple Silicon (MPS)
2. Keep audio files under 10 minutes when possible
3. Close other resource-intensive applications

### Update the Whisper or LLM Model

You can easily change the models used for transcription and summarization by modifying the variables at the top of the script:

```python
# Default models
OLLAMA_MODEL = "llama3"
WHISPER_MODEL = "openai/whisper-large-v2"
```

#### Changing the Whisper Model

To use a different Whisper model for transcription:

1. Update the `WHISPER_MODEL` variable with one of these options:
   - `"openai/whisper-tiny"` (fastest, least accurate)
   - `"openai/whisper-base"` (faster, less accurate)
   - `"openai/whisper-small"` (balanced)
   - `"openai/whisper-medium"` (slower, more accurate)
   - `"openai/whisper-large-v2"` (slowest, most accurate)

2. Example:

   ```python
   WHISPER_MODEL = "openai/whisper-medium"  # A good balance between speed and accuracy
   ```

For CPU-only systems, using a smaller model like `whisper-base` is recommended for better performance.

#### Changing the LLM Model

To use a different model for summarization:

1. First, pull the desired model with Ollama:

   ```bash
   ollama pull mistral  # or any other supported model
   ```

2. Then update the `OLLAMA_MODEL` variable:

   ```python
   OLLAMA_MODEL = "mistral"  # or any other model you've pulled
   ```

3. Popular alternatives include:
   - `"llama3"` (default)
   - `"mistral"`
   - `"llama2"`
   - `"gemma:7b"`
   - `"phi"`

For a complete list of available models, visit the [Ollama model library](https://ollama.com/library).
