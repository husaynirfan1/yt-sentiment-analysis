# yt-sentiment-analysis

A YouTube comment sentiment analysis application leveraging Large Language Models (LLMs). This project is specifically designed to analyze sentiment in YouTube comments on videos related to Malaysian politics, aiming to provide insights into public opinion regarding political entities and topics.

The project consists of two main components:

*   `gui_app.py`: A user-friendly graphical interface for managing inputs (such as YouTube video URLs), initiating the sentiment analysis process, and viewing the summarized results.
*   `scraper_v2.py`: A powerful backend script responsible for fetching YouTube comment data, utilizing AI models for filtering relevant comments and performing sentiment analysis, and generating detailed output reports.


# Demo

https://github.com/user-attachments/assets/3f6d7fd2-c67b-44eb-8538-7f8f6732b394

### Features:
*   **User-Friendly Interface:** A GUI (`gui_app.py`) for easy interaction, input management, and results visualization.
*   **Targeted Video Fetching:** Retrieves videos from a specified YouTube channel.
*   **AI-Powered Political Video Filtering:** Utilizes LLMs to identify and select videos relevant to Malaysian politics based on titles and descriptions.
*   **Automated Audio Processing:** Downloads audio from selected videos.
*   **Speech-to-Text Transcription:** Transcribes video audio using AI (Fireworks Whisper).
*   **Content Summarization:** Generates concise contextual summaries of video content using LLMs.
*   **In-Depth Comment Analysis:**
    *   Fetches comments from YouTube videos.
    *   Performs targeted sentiment analysis on individual comments using LLMs.
    *   Identifies sarcasm and provides reasoning.
    *   Determines the primary political entity targeted by the comment.
    *   Assigns a sentiment score (positive, negative, neutral) and a numerical score (-1.0 to 1.0).
    *   Provides reasoning for sentiment and entity identification.
    *   Estimates the confidence level of the analysis.
*   **Comprehensive Reporting:**
    *   Generates an overall sentiment summary across multiple videos and comments.
    *   Creates a visual plot of sentiment distribution.
    *   Saves detailed results for each video in a structured folder format.
*   **Flexible API Key Management:** Supports API key input via GUI or a `.env` file.
*   **Customizable Analysis Parameters:** Allows users to define the maximum number of videos to scan and comments to analyze.
*   **CLI Availability:** Offers a command-line interface (`scraper_v2.py`) for backend operations and scripting.

### Tech Stack

*   **Programming Language:**
    *   Python (specifically Python 3.9)
*   **Graphical User Interface (GUI):**
    *   Tkinter
    *   `customtkinter` (for modern theming)
*   **Data Acquisition & Web Scraping:**
    *   `google-api-python-client` (for YouTube Data API v3)
    *   `yt-dlp` (for downloading YouTube video audio)
*   **Artificial Intelligence & Machine Learning (LLM):**
    *   `langchain` (for orchestrating LLM interactions)
    *   Fireworks AI API (for accessing LLM models)
        *   Models used: `accounts/fireworks/models/qwen3-235b-a22b` (for filtering, summarization, sentiment analysis), `whisper-v3` (for audio transcription via Fireworks' OpenAI-compatible endpoint)
    *   `openai` (Python client, used for Fireworks AI's Whisper-compatible API)
*   **Data Handling & Manipulation:**
    *   Standard Python libraries (JSON, OS, re, etc.)
*   **Data Visualization:**
    *   `matplotlib` (for generating sentiment plots)
*   **Environment & Configuration:**
    *   `python-dotenv` (for managing API keys from `.env` files)
*   **Utilities:**
    *   `Pillow` (PIL - Python Imaging Library, used by `customtkinter` and potentially `matplotlib`)

### Setup and Installation

#### Prerequisites

*   **Python:** Version 3.9 (the application has been tested with this version).
*   **pip:** Python package installer (usually comes with Python).
*   **Git:** For cloning the repository.
*   **(Optional but Recommended)** A bash-like terminal (e.g., Git Bash on Windows, or the default terminal on macOS/Linux) for easier command execution.

#### Installation Steps

1.  **Clone the Repository:**
    Open your terminal and run the following commands:
    ```bash
    git clone https://github.com/user/yt-sentiment-analysis.git # Replace with the actual repository URL
    cd yt-sentiment-analysis # Replace with the actual repository directory name
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    A virtual environment helps manage project-specific dependencies and avoids conflicts with other Python projects.
    ```bash
    python3 -m venv venv
    ```
    Activate the virtual environment:
    *   On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        venv\Scripts\activate
        ```
    Your terminal prompt should change to indicate that the virtual environment is active.

3.  **Install Dependencies:**
    A `requirements.txt` file is provided, listing all necessary Python packages. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    To use the application's full capabilities, you'll need API keys for:
    *   **YouTube Data API v3:** For fetching video information and comments.
    *   **Fireworks AI API:** For accessing LLMs for analysis and transcription.

    An example file `.env.example` is provided as a template.
    *   First, copy this file to create your own `.env` file:
        ```bash
        cp .env.example .env
        ```
    *   Next, open the `.env` file with a text editor and enter your API keys:
        ```
        YOUTUBE_API_KEY="YOUR_YOUTUBE_API_KEY_HERE"
        FIREWORKS_API_KEY="YOUR_FIREWORKS_API_KEY_HERE"
        ```
    *   **Note:** While API keys can also be entered directly into the GUI, using the `.env` file is more secure and convenient for repeated use, as the application will automatically load them.

5.  **(Optional) FFmpeg for `yt-dlp`:**
    The `yt-dlp` tool, used for downloading audio from YouTube videos, may require FFmpeg for certain audio conversion tasks (e.g., ensuring compatibility or converting to formats like MP3).
    *   It's recommended to install FFmpeg on your system and ensure it's added to your system's PATH.
    *   You can download FFmpeg from its official website: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

### Usage Guide (GUI)

This section explains how to run and use the GUI application (`gui_app.py`).

#### 1. Launching the Application

Ensure you have completed the "Setup and Installation" steps, specifically:
*   The virtual environment is activated (`source venv/bin/activate` or `venv\Scripts\activate`).
*   All dependencies are installed (`pip install -r requirements.txt`).

To launch the GUI application, run the following command in your terminal from the project's root directory:
```bash
python gui_app.py
```

#### 2. Main Interface Overview

The GUI provides a tabbed interface for managing the analysis process and viewing results.

*   **API Key and Channel ID Inputs (Top Section):**
    *   **`YouTube Data API Key`**: Your API key for accessing the YouTube Data API. This is required to fetch video details and comments. It will be pre-filled if you have set it in the `.env` file.
    *   **`Fireworks API Key`**: Your API key for Fireworks AI. This is needed for LLM-based filtering, transcription, summarization, and sentiment analysis. It will be pre-filled if you have set it in the `.env` file.
    *   **`YouTube Channel ID`**: The unique ID of the YouTube channel from which you want to fetch videos (e.g., `UCXXXXXXX`).

#### 3. Configuration Settings (Top Section)

*   **`Max Videos to Scan`**: This number limits how many of the most recent videos from the specified channel will be initially fetched and scanned by the LLM to identify politically relevant content. For example, setting this to `50` means the application will look at the latest 50 videos.
*   **`Comments to Analyze per Video`**: This defines the maximum number of comments to retrieve and analyze for each selected video. You can enter a specific number (e.g., `100`) or type `all` to attempt to fetch and analyze all available comments (be mindful of API quotas and processing time).

#### 4. Workflow

*   **Fetching Videos:**
    1.  Enter the required API keys (if not pre-filled from `.env`), the YouTube Channel ID, and adjust the configuration settings.
    2.  Click the **"Fetch Videos"** button.
    3.  This action initiates the `scraper_v2.py` script in the background. The script will:
        *   Fetch the latest videos from the channel (up to `Max Videos to Scan`).
        *   Use an LLM to filter these videos based on their titles and descriptions to identify those relevant to Malaysian politics.
        *   Log messages and progress will appear in the **"Log"** tab.

*   **Selecting Videos for Analysis:**
    1.  Once the fetching and filtering process is complete, the **"Filtered Political Videos"** list (on the left side of the "Video Selection & Analysis" tab) will be populated with videos deemed politically relevant.
    2.  Review this list and use the checkboxes next to each video title to select the ones you want to analyze in detail.

*   **Analyzing Selected Videos:**
    1.  After selecting the desired videos, click the **"Analyze Selected Videos"** button.
    2.  This will again trigger `scraper_v2.py` to perform the full analysis pipeline on *only* the videos you selected. This includes:
        *   Downloading audio for each selected video.
        *   Transcribing the audio to text.
        *   Summarizing the video content.
        *   Fetching comments for each video (up to `Comments to Analyze per Video`).
        *   Performing detailed sentiment analysis on each comment.
    3.  Progress and any errors will be updated in the **"Log"** tab. Results will be saved in the `output` directory, organized by video.

#### 5. Understanding the Output Tabs

The GUI features several tabs to display information:

*   **Log Tab:**
    *   This tab shows real-time (or near real-time) messages, progress updates, and error notifications generated by the backend scripts (`scraper_v2.py`) during the fetching and analysis processes. Check here first if something seems stuck or isn't working as expected.

*   **Overall Summary Tab:**
    *   This tab is designed to give a high-level overview of the sentiment across all analyzed videos.
    *   It attempts to load and display the aggregated sentiment data from the `output/overall_sentiment_summary.json` file (if available).
    *   It also displays the sentiment visualization plot (`output/overall_sentiment_plot.png`) if it has been generated.

*   **Individual Video Results Tab:**
    *   This tab provides a list of all videos that were selected and processed for detailed analysis.
    *   For each video in the list, it indicates the status or existence of key output files that should have been generated in that video's specific output folder (e.g., `output/<video_id>/video_meta.json`, `output/<video_id>/transcription.txt`, `output/<video_id>/targeted_sentiment_analysis.json`).
    *   A crucial feature here is the **"Open Video Folder"** button associated with each video. Clicking this button will open the specific directory for that video (e.g., `output/<video_id>/`) in your system's file explorer, allowing you to directly access all generated reports, transcripts, and data files for that video.

### Usage Guide (CLI - `scraper_v2.py`)

The `scraper_v2.py` script can be used directly from the command line for more advanced or automated workflows, or for running backend processes without the GUI.

#### 1. General Usage

*   **Basic command structure:** `python scraper_v2.py [arguments]`
*   **Virtual Environment:** Remember to activate your virtual environment (`source venv/bin/activate` or `venv\Scripts\activate`) before running the script.
*   **Use Cases:** Ideal for scripting, batch processing, or integrating the analysis capabilities into other systems.

#### 2. Actions

The `--action` argument is mandatory and specifies the operation to perform.

*   **`--action fetch_videos`**
    *   **Purpose:** Fetches videos from a specified YouTube channel and filters them using an LLM to identify politically relevant content.
    *   **Required Argument:**
        *   `--channel-id <YOUTUBE_CHANNEL_ID>`: The ID of the YouTube channel.
    *   **Optional Arguments:**
        *   `--youtube-api-key <YOUR_KEY>`: Overrides the YouTube API key in the `.env` file.
        *   `--fireworks-api-key <YOUR_KEY>`: Overrides the Fireworks API key in the `.env` file.
        *   `--max-videos-scan <NUMBER>`: Maximum number of recent videos to scan (default: 20).
    *   **Output:** Prints a JSON formatted list of filtered video objects to standard output. This output can be redirected to a file (e.g., `> political_videos.json`) for use with the `analyze_videos` action.

*   **`--action analyze_videos`**
    *   **Purpose:** Performs a full analysis (audio download, transcription, summarization, comment fetching, and sentiment analysis) on a list of videos provided in a JSON file.
    *   **Required Argument:**
        *   `--input-file <PATH_TO_JSON_FILE>`: Path to a JSON file containing the list of video objects to analyze (typically the output from the `fetch_videos` action).
    *   **Optional Arguments:**
        *   `--output-dir <DIRECTORY_PATH>`: Directory where analysis results will be saved (default: `analysis_results_scraper`). Each video will have its own subfolder.
        *   `--comments-to-analyze <NUMBER_OR_ALL>`: Number of comments to analyze per video (e.g., `100`, `500`, or `all`) (default: "100").
        *   `--youtube-api-key <YOUR_KEY>`: Overrides the YouTube API key in the `.env` file.
        *   `--fireworks-api-key <YOUR_KEY>`: Overrides the Fireworks API key in the `.env` file.
    *   **Output:** Saves detailed analysis results (metadata, transcript, summary, comment analysis, etc.) into structured folders within the specified output directory. An overall summary and plot will also be generated in the root of the output directory.

#### 3. API Keys with CLI

*   API keys for YouTube and Fireworks AI can be provided directly as command-line arguments (e.g., `--youtube-api-key YOUR_ACTUAL_KEY`).
*   If provided via command-line, these arguments will override any keys found in your `.env` file for that specific execution.
*   If not provided as arguments, the script will attempt to load them from the `.env` file as described in the "Setup and Installation" section.

#### 4. Example Commands

*   **Fetching videos from a channel:**
    ```bash
    python scraper_v2.py --action fetch_videos --channel-id "UCXXXXXXX" --max-videos-scan 50 > political_videos.json
    ```
    This command fetches the latest 50 videos from the channel `UCXXXXXXX`, filters them for political content, and saves the resulting list of video details into a file named `political_videos.json`.

*   **Analyzing videos from a file:**
    ```bash
    python scraper_v2.py --action analyze_videos --input-file political_videos.json --output-dir my_analysis_results --comments-to-analyze 50
    ```
    This command takes the `political_videos.json` file (created in the previous example), analyzes each video listed in it, processes up to 50 comments for each, and saves all the output files and reports into the `my_analysis_results` directory.

### Output Structure

The application generates a structured set of output files and directories to store the results of the analysis.

#### 1. Main Output Directory

*   The analysis results are saved in a main output directory. The default name depends on how the analysis was initiated:
    *   `analysis_results_gui`: When the analysis is run from the GUI (`gui_app.py`).
    *   `analysis_results_scraper`: When the `analyze_videos` action is run from the CLI (`scraper_v2.py`). This can be overridden using the `--output-dir` argument in the CLI.
*   This main directory will contain:
    *   Subdirectories for each individual video that was analyzed.
    *   Overall summary files that aggregate data from all analyzed videos.

#### 2. Individual Video Output Subdirectory

*   Each video selected and processed for analysis will have its own subdirectory within the main output directory.
*   **Structure:** The subdirectory name is typically formatted as `<VIDEO_ID>_<SANITIZED_VIDEO_TITLE_PART>`.
    *   Example: `dQw4w9WgXcQ_Rick_Astley_Never_Gonna_Give_You_Up`
*   **Key files within each video's subdirectory:**
    *   `video_meta.json`: Contains metadata about the video, such as its title, full description, YouTube URL, and unique video ID.
    *   `original_comments.json`: A JSON file listing the raw comments fetched from the video, up to the limit specified in the configuration (e.g., "Comments to Analyze per Video"). Each comment entry usually includes the comment text, author, and other details.
    *   `audio_track.mp3`: The audio track downloaded from the YouTube video. This file is generated if audio processing (download and conversion) was successful.
    *   `transcription.txt`: A plain text file containing the transcription of the video's audio content, generated by the speech-to-text model.
    *   `contextual_summary.txt`: An AI-generated summary that provides a concise overview of the video's content. This summary is typically derived from the video's title, description, and the generated transcription.
    *   `targeted_sentiment_analysis.json`: This crucial JSON file stores the detailed sentiment analysis results for each processed comment from the video. For each comment, it may include:
        *   The original comment text.
        *   Sarcasm detection (e.g., true/false) and reasoning.
        *   The primary political entity targeted by the comment.
        *   The assigned sentiment (e.g., "positive", "negative", "neutral").
        *   A numerical sentiment score (e.g., ranging from -1.0 to 1.0).
        *   Reasoning for the sentiment and entity identification.
        *   Confidence level of the analysis.

#### 3. Overall Summary Files (in the main output directory)

Located directly within the main output directory (e.g., `analysis_results_gui/` or `analysis_results_scraper/`), these files provide an aggregated view of the analysis:

*   `overall_sentiment_summary.json`: A JSON file that aggregates sentiment data from all analyzed comments across all processed videos. Its content typically includes:
    *   Overall statistics, such as the total number of comments analyzed and the count of comments identified as sarcastic.
    *   A breakdown of sentiment counts (positive, negative, neutral) for each political entity identified across all comments.
    *   Average sentiment scores calculated for each targeted political entity.
*   `overall_sentiment_plot.png`: An image file (PNG format) displaying a bar chart. This chart visually represents the sentiment distribution (number of positive, negative, and neutral comments) for each major political entity targeted in the comments.

### Code Overview

This section provides a high-level look at the main Python scripts in the project and their roles.

#### 1. `gui_app.py` (Graphical User Interface)

*   **Purpose:** Provides an interactive graphical interface for the user to manage and run the sentiment analysis pipeline.
*   **Key Features & Design:**
    *   Built using `customtkinter` for a modern and themeable user interface.
    *   Manages user inputs such as API keys (YouTube, Fireworks AI), YouTube Channel ID, and analysis parameters (e.g., max videos to scan, comments per video).
    *   Initiates backend processes by calling `scraper_v2.py` as a subprocess, allowing for modular execution.
    *   Displays lists of fetched and filtered videos, allowing users to select specific videos for detailed analysis.
    *   Presents logs, overall sentiment summaries, and individual video results in a tabbed interface for organized viewing.
        *   **Log Tab:** Shows real-time output from the backend script.
        *   **Overall Summary Tab:** Displays aggregated sentiment data and plots.
        *   **Individual Video Results Tab:** Lists analyzed videos and provides access to their specific output folders.
    *   Employs threading to run the backend `scraper_v2.py` script. This ensures the GUI remains responsive and does not freeze during long-running operations like video fetching or analysis.

#### 2. `scraper_v2.py` (Backend Scraper & Analyzer)

*   **Purpose:** Acts as the core engine of the application, handling all data fetching, processing, AI-driven analysis, and report generation.
*   **Execution:**
    *   Can be run independently from the command line (CLI) for automated or scripted tasks.
    *   Is also executed as a subprocess by `gui_app.py` to perform actions initiated from the user interface.
*   **Key Functional Stages :**
    *   **`agent0_fetch_and_filter_videos`:**
        *   Fetches a list of recent videos from the specified YouTube channel using the YouTube Data API.
        *   Utilizes a Large Language Model (LLM) via Fireworks AI to filter these videos, identifying those relevant to Malaysian politics based on their titles and descriptions.
    *   **Audio Processing:**
        *   Downloads the audio track from selected YouTube videos using `yt-dlp`.
        *   Transcribes the downloaded audio to text using the Fireworks AI Whisper model (via their OpenAI-compatible API).
    *   **Content Summarization (`generate_contextual_summary_langchain`):**
        *   Uses an LLM (via Langchain and Fireworks AI) to generate a concise summary of each video's content, based on its metadata (title, description) and transcription.
    *   **Comment Fetching:**
        *   Retrieves comments for the selected videos using the YouTube Data API, respecting the user-defined limit on the number of comments.
    *   **Sentiment Analysis (`analyze_comment_sentiment_analysis_langchain`):**
        *   Performs detailed, targeted sentiment analysis on individual comments. This is a multi-faceted LLM task (via Langchain and Fireworks AI) that aims to:
            *   Detect sarcasm in the comment.
            *   Identify the primary political entity or topic being discussed.
            *   Assign a sentiment label (e.g., "positive," "negative," "neutral").
            *   Provide a numerical sentiment score.
            *   Generate a textual reasoning for the assigned sentiment and identified entity.
            *   Estimate the confidence of the analysis.
    *   **Report Generation:**
        *   Aggregates sentiment data from all analyzed comments and videos into a structured `overall_sentiment_summary.json` file.
        *   Creates a visual bar chart (`overall_sentiment_plot.png`) showing the distribution of sentiments towards different political entities.
*   **Technical Highlights:**
    *   Leverages Pydantic models (e.g., `TargetedCommentSentiment`, `PoliticalVideoCheck`) for data validation and, crucially, for parsing structured output from LLM responses. This ensures that the complex data extracted by AI models is reliable and easy to work with.
    *   Manages all file system operations for creating and organizing the output directory structure, saving individual video data, transcriptions, summaries, and aggregated reports.
    *   Handles API key management by prioritizing keys provided as CLI arguments over those defined in the `.env` file, offering flexibility for different execution contexts.

### Troubleshooting

This section lists common problems users might encounter and suggests solutions.

#### 1. API Key Errors

*   **Symptom:** Errors in the log related to "API key invalid," "quota exceeded," or authentication failures.
*   **Solutions:**
    *   Double-check that the API keys in your `.env` file or entered in the GUI are correct and have the necessary permissions (e.g., YouTube Data API v3 enabled, Fireworks AI account active).
    *   Ensure there are no extra spaces or characters around the keys.
    *   Check API usage dashboards (Google Cloud Console for YouTube, Fireworks AI platform) for quota limits.
    *   If using `.env`, ensure the file is correctly named (`.env`) and located in the project's root directory.

#### 2. Python Version Issues

*   **Symptom:** Syntax errors, `ModuleNotFoundError` that might relate to features in newer Python versions, or `customtkinter` not working as expected.
*   **Solutions:**
    *   Ensure you are using Python 3.9, as the application has been tested with this version. Use `python --version` or `python3 --version` to check.
    *   It's highly recommended to use a virtual environment (as described in the "Setup and Installation" section) to manage project-specific dependencies and Python versions.

#### 3. `requirements.txt` Installation Problems

*   **Symptom:** Errors during the `pip install -r requirements.txt` command.
*   **Solutions:**
    *   Ensure `pip` is up-to-date: `pip install --upgrade pip`.
    *   Check your internet connection.
    *   If a specific package fails to install, try installing it individually (e.g., `pip install packagename`) to see more detailed error messages. Some packages might have system-level dependencies that need to be installed separately (though this is less common for the packages in this project).

#### 4. FFmpeg Not Found (for `yt-dlp` audio extraction)

*   **Symptom:** Logs indicate failure during audio download or conversion stages. Messages might specifically mention that FFmpeg is missing or not found. The application might still fetch comments and perform text-based analysis, but audio-related features (transcription, audio file saving) will likely fail.
*   **Solutions:**
    *   Install FFmpeg from its official website: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html).
    *   Ensure that the directory containing the `ffmpeg` executable (and `ffprobe`, which is usually included) is added to your system's PATH environment variable.
    *   Verify the installation by opening a new terminal window and typing `ffmpeg -version`. If it's installed correctly, you should see version information.

#### 5. GUI Doesn't Launch or Looks Incorrect

*   **Symptom:** An error message appears when trying to run `python gui_app.py`, or the GUI window opens but elements are missing, distorted, or not functioning.
*   **Solutions:**
    *   Ensure all dependencies from `requirements.txt` are installed correctly within your active virtual environment. Pay special attention to `customtkinter` and `Pillow`.
    *   If you are on a Linux system, ensure you have Tkinter support installed. This is often provided via a system package (e.g., `sudo apt-get install python3-tk` on Debian/Ubuntu-based systems).
    *   Check for any error messages printed in the terminal when you attempt to launch the application. These messages can provide clues about the cause of the issue.

#### 6. LLM Output Issues (e.g., empty JSON, incorrect formatting)

*   **Symptom:** Errors appear in the logs related to JSON parsing from LLM (Large Language Model) output. Analysis results in the output files might be incomplete, nonsensical, or missing expected data fields.
*   **Solutions:**
    *   **Intermittent LLM Issues:** This could be due to temporary issues with the LLM provider (Fireworks AI). Consider trying the operation again after a short wait.
    *   **API Key & Credits:** Verify that your Fireworks AI API key is active and has sufficient credits or usage quota if applicable.
    *   **Prompt Adherence:** The prompts used in `scraper_v2.py` are designed to instruct the LLM to return responses in a specific JSON format. If the LLM significantly deviates from this format, parsing errors can occur. While the script includes some cleaning functions, fundamental changes in LLM behavior might require adjustments to the prompts or parsing logic in the code.
    *   **Check Logs:** Examine the logs generated by `scraper_v2.py` (visible in the GUI's "Log" tab or in the console if running CLI) for any specific error messages returned during the LLM calls. These can sometimes indicate problems with the input data, prompt structure, or the LLM itself.
