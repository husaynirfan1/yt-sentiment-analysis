import os
import re
import json
import time # For potential rate limiting if ever needed
import yt_dlp
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Langchain and Fireworks AI specific imports
from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# For Fireworks AI Whisper transcription via OpenAI-compatible API
import openai

# Load environment variables from .env file
load_dotenv()

# --- API KEYS ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")

# --- FFmpeg Location (Optional) ---
FFMPEG_LOCATION = None

# --- Malaysian Political Entities for Analysis ---
MALAYSIAN_POLITICAL_ENTITIES = [
    "Pakatan Harapan (PH)", "Perikatan Nasional (PN)", "Barisan Nasional (BN)",
    "Gabungan Parti Sarawak (GPS)", "Gabungan Rakyat Sabah (GRS)",
    "Malaysian United Democratic Alliance (MUDA)", "Parti Sosialis Malaysia (PSM)",
    "Other (Specify if clearly identifiable and not listed)",
    "Neutral/No clear leniency", "Unclear due to insufficient information"
]
TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES = [
    "Pakatan Harapan (PH)", "Perikatan Nasional (PN)", "Barisan Nasional (BN)",
    "Gabungan Parti Sarawak (GPS)", "Gabungan Rakyat Sabah (GRS)",
    "Malaysian United Democratic Alliance (MUDA)", "Parti Sosialis Malaysia (PSM)",
    # We'll add a special value for the LLM to use when no specific listed entity is targeted.
    "General Comment/No Specific Listed Entity"
]
class TargetedCommentSentiment(BaseModel):
    comment_text: str = Field(description="The original comment text.")
    is_sarcastic: bool = Field(description="True if the comment is sarcastic, False otherwise.")
    sarcasm_reasoning: str = Field(description="Brief explanation if sarcasm is detected, or 'N/A' if not sarcastic.")

    primary_target_entity: str = Field(
        description=(
            "The Malaysian political entity primarily targeted or discussed in the comment. "
            f"Must be one of: {', '.join(TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES)}. "
            "If the comment's sentiment is general or targets an unlisted individual/group not clearly representing a listed entity, "
            "use 'General Comment/No Specific Listed Entity'."
        )
    )
    entity_identification_reasoning: str = Field(
        description="Brief reasoning for identifying the primary_target_entity, or for choosing 'General Comment/No Specific Listed Entity'."
    )

    sentiment_expressed: str = Field(
        description=(
            "The sentiment expressed towards the 'primary_target_entity'. If 'primary_target_entity' is "
            "'General Comment/No Specific Listed Entity', this is the overall sentiment of the comment. "
            "Must be one of: 'positive', 'negative', 'neutral'."
        )
    )
    sentiment_score: float = Field(
        description=(
            "A score from -1.0 (very negative) to 1.0 (very positive) for the 'sentiment_expressed', "
            "with 0.0 representing neutral. This score should reflect the intensity of the sentiment. "
            "Sarcasm should be factored in (e.g., if sarcasm inverts an apparently positive statement to express a negative sentiment towards the target)."
        )
    )
    overall_reasoning: str = Field(
        description=(
            "A brief explanation for the sentiment classification and score, specifically regarding the 'primary_target_entity' "
            "(or the overall comment if 'General Comment/No Specific Listed Entity'). "
            "Explain how sarcasm influenced this judgment."
        )
    )

# --- Pydantic Models for LLM Outputs ---
class PoliticalVideoCheck(BaseModel):
    is_political: bool = Field(description="True if the video is primarily about Malaysian politics, False otherwise.")
    reasoning: str = Field(description="A brief explanation for the decision.")
    video_title: str = Field(description="The original video title.")
    video_id: str = Field(description="The YouTube video ID.")

class CommentLeniency(BaseModel):
    comment_text: str = Field(description="The original comment text.")
    is_sarcastic: bool = Field(description="True if the comment is sarcastic, False otherwise.")
    sarcasm_reasoning: str = Field(description="Brief explanation if sarcasm is detected, or 'N/A' if not sarcastic.")
    political_entity_lean: str = Field(description=f"The identified political entity from the list: {', '.join(MALAYSIAN_POLITICAL_ENTITIES)}. Consider sarcasm's impact.")
    leniency_score_percent: float = Field(description="A score from 0 to 100 indicating confidence/strength of leaning (0 if Neutral/Unclear). Adjust based on sarcasm if it negates apparent leaning.")
    reasoning: str = Field(description="A brief explanation for your political leniency judgment (1-2 sentences), factoring in any detected sarcasm.")

class CommentSentimentAnalysis(BaseModel):
    comment_text: str = Field(description="The original comment text.")
    is_sarcastic: bool = Field(description="True if the comment is sarcastic, False otherwise.")
    sarcasm_reasoning: str = Field(description="Brief explanation if sarcasm is detected, or 'N/A' if not sarcastic.")
    overall_sentiment: str = Field(description="The overall sentiment of the comment. Must be one of: 'positive', 'negative', 'neutral'.")
    sentiment_score: float = Field(description="A score from -1.0 (very negative) to 1.0 (very positive), with 0.0 representing neutral. This score should reflect the intensity of the sentiment. Sarcasm should be factored in (e.g., if sarcasm inverts an apparently positive statement to express a negative sentiment).")
    sentiment_reasoning: str = Field(description="A brief explanation for the sentiment classification and score, explicitly mentioning how any detected sarcasm influenced the judgment.")
# --- Helper Functions ---
def extract_video_id(url):
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]{11})",
        r"(?:https?:\/\/)?youtu\.be\/([a-zA-Z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match: return match.group(1)
    print(f"Warning: Could not extract video ID from URL: {url}")
    return None

def save_data_to_file(data_content, output_folder_path, filename):
    if data_content is None: return
    file_path = os.path.join(output_folder_path, filename)
    try:
        mode = 'w'
        encoding = 'utf-8'
        with open(file_path, mode, encoding=encoding) as f:
            if isinstance(data_content, str):
                f.write(data_content)
            else:
                json.dump(data_content, f, ensure_ascii=False, indent=4)
        print(f"Saved {filename} to: {os.path.abspath(file_path)}")
    except Exception as e:
        print(f"Error saving {filename} to {file_path}: {e}")

def clean_llm_json_output(raw_text: str) -> str:
    """
    Cleans potential markdown formatting around JSON output from LLMs.
    """
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    # Sometimes LLMs might add a preamble like "Here's the JSON output:"
    # This is a basic attempt to find the start of the JSON.
    # More robust parsing might be needed if issues persist.
    json_start_index = text.find('{')
    json_end_index = text.rfind('}')
    if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
        text = text[json_start_index : json_end_index + 1]
    return text.strip()

# --- YouTube API Functions ---
def get_channel_uploads_playlist_id(youtube, channel_id):
    try:
        request = youtube.channels().list(part="contentDetails", id=channel_id)
        response = request.execute()
        if response.get("items"):
            return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as e:
        print(f"API error getting uploads playlist ID for channel {channel_id}: {e}")
    return None

def fetch_videos_from_playlist(youtube, playlist_id, max_results=25):
    videos = []
    next_page_token = None
    try:
        while True:
            request = youtube.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=min(max_results - len(videos), 50), # Fetch up to 50 at a time
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                video_id = snippet.get("resourceId", {}).get("videoId")
                title = snippet.get("title")
                description = snippet.get("description")
                if video_id and title: # Ensure essential info is present
                    videos.append({
                        "video_id": video_id,
                        "title": title,
                        "description": description,
                        "url": f"https://www.youtube.com/watch?v={video_id}" # Corrected URL
                    })
                if len(videos) >= max_results: break
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(videos) >= max_results: break
        print(f"Fetched {len(videos)} video snippets from channel's uploads playlist.")
    except HttpError as e:
        print(f"API error fetching videos from playlist {playlist_id}: {e}")
    return videos

def fetch_youtube_video_details_and_comments_api(video_url, api_key):
    video_details = {'title': None, 'description': None, 'comments_data': []}
    video_id = extract_video_id(video_url)
    if not video_id: return video_details

    if not api_key:
        print("Warning: YouTube API Key not set. Cannot fetch video details or comments.")
        return video_details
    try:
        youtube = build('youtube', 'v3', developerKey=api_key)
        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_response = video_request.execute()
        if video_response.get("items"):
            snippet = video_response["items"][0]["snippet"]
            video_details['title'] = snippet.get("title", "No title (API)")
            video_details['description'] = snippet.get("description", "No description (API)")
        else:
            print(f"Could not fetch video details (API) for ID: {video_id}")

        next_page_token = None
        while True: # Limiting comment fetch for now to avoid excessive API calls during testing
            comment_request = youtube.commentThreads().list(
                part='snippet,replies', videoId=video_id, maxResults=100, # Max 100 per page
                pageToken=next_page_token, textFormat='plainText', order='relevance') # Added order
            comment_response = comment_request.execute()
            for item in comment_response.get('items', []):
                top_comment_snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                if top_comment_snippet.get('textDisplay'): # Ensure comment text exists
                    video_details['comments_data'].append({
                        'author': top_comment_snippet.get('authorDisplayName', 'Unknown Author'),
                        'comment_text': top_comment_snippet.get('textDisplay')
                    })
                if item.get('replies'):
                    for reply in item['replies']['comments']:
                        reply_snippet = reply.get('snippet', {})
                        if reply_snippet.get('textDisplay'): # Ensure reply text exists
                            video_details['comments_data'].append({
                                'author': f"    (Reply to {top_comment_snippet.get('authorDisplayName', 'Original Author')}) {reply_snippet.get('authorDisplayName', 'Unknown Author')}",
                                'comment_text': reply_snippet.get('textDisplay')
                            })
            next_page_token = comment_response.get('nextPageToken')
            # Removed break for single page to fetch more comments if available
            if not next_page_token or len(video_details['comments_data']) >= 200: # Cap at 200 comments for now
                 break
        if video_details['comments_data']:
            print(f"Extracted {len(video_details['comments_data'])} comments (API).")
        else:
            print(f"No comments found or extracted for video ID: {video_id}")
    except HttpError as e:
        error_content = e.content.decode('utf-8', 'ignore') if e.content else "No error content"
        print(f"YouTube API HTTP error {e.resp.status if e.resp else 'N/A'}: {error_content}")
    except Exception as e:
        print(f"Error during YouTube API operations: {e}")
    return video_details


# --- yt-dlp Audio Download ---
def download_audio_mp3_yt_dlp(video_url, output_path='.', filename_template='audio.%(ext)s'):
    # Ensure the base filename is 'audio' so we can predict 'audio.mp3'
    base_filename = os.path.splitext(filename_template)[0]
    actual_filename_mp3 = os.path.join(output_path, f"{base_filename}.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': os.path.join(output_path, base_filename), # yt-dlp adds .mp3
        'quiet': False, 'noplaylist': True, 'nocheckcertificate': True,
        'ignoreerrors': True, # Continue on download errors for individual videos if in a playlist context (though noplaylist is True)
    }
    if FFMPEG_LOCATION: ydl_opts['ffmpeg_location'] = FFMPEG_LOCATION

    try:
        print(f"\nDownloading audio for: {video_url} to {os.path.abspath(output_path)}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if os.path.exists(actual_filename_mp3):
            print(f"Audio downloaded and converted to MP3: {actual_filename_mp3}")
            return actual_filename_mp3
        else:
            # Fallback: Check if any .mp3 file was created in the directory if the exact name isn't found
            # This can happen if the title had special characters that yt-dlp sanitized differently.
            # However, with 'outtmpl' set to a fixed name like 'audio', this should be less of an issue.
            found_mp3s = [f for f in os.listdir(output_path) if f.lower().endswith(".mp3") and f.lower().startswith(base_filename.lower())]
            if found_mp3s:
                found_path = os.path.join(output_path, found_mp3s[0])
                # Optionally rename to the expected actual_filename_mp3
                # os.rename(found_path, actual_filename_mp3)
                print(f"Audio downloaded. Found MP3 (possibly sanitized name): {found_path}")
                return found_path # or actual_filename_mp3 if renamed
            print(f"Error: MP3 file not found at {actual_filename_mp3} or matching pattern in {output_path}.")
            return None
    except Exception as e:
        print(f"Error during audio download for {video_url}: {e}")
    return None

# --- AI Agents (Langchain & Fireworks AI) ---
# --- Leniency Calculation Functions ---
def calculate_targeted_sentiment_summary(analyses_list, title="Targeted Sentiment Summary"):
    """
    Calculates and prints targeted sentiment distribution from a list of analyses.
    Each item in analyses_list is expected to conform to TargetedCommentSentiment model.
    """
    if not analyses_list:
        print(f"\n--- {title} ---")
        print("No targeted sentiment analyses provided to summarize.")
        return {
            "by_entity": {entity: {"positive": 0, "negative": 0, "neutral": 0, "sarcastic_count": 0,
                                   "sum_sentiment_score": 0.0, "valid_score_count": 0,
                                   "comment_count": 0, "average_sentiment_score": 0.0}
                          for entity in TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES},
            "overall_stats": {
                "total_analyzed_items": 0,
                "valid_analyses": 0, # Analyses that are dicts
                "total_sarcastic_comments_overall": 0,
            }
        }

    sentiment_by_entity = {
        entity: {
            "positive": 0, "negative": 0, "neutral": 0,
            "sarcastic_count": 0,
            "sum_sentiment_score": 0.0, "valid_score_count": 0,
            "comment_count": 0, # Total comments primarily targeting this entity
            "average_sentiment_score": 0.0
        } for entity in TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES
    }

    overall_stats = {
        "total_analyzed_items": 0,
        "valid_analyses": 0,
        "total_sarcastic_comments_overall": 0,
    }

    for analysis in analyses_list:
        overall_stats["total_analyzed_items"] += 1
        if not isinstance(analysis, dict):
            print(f"Skipping invalid analysis item (not a dict): {analysis}")
            continue

        overall_stats["valid_analyses"] += 1
        target_entity = analysis.get('primary_target_entity')
        sentiment = analysis.get('sentiment_expressed', 'neutral').lower()
        score = analysis.get('sentiment_score')
        is_sarcastic = analysis.get('is_sarcastic', False)

        # Validate and normalize target_entity
        if target_entity not in TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES:
            original_target = target_entity # Store for logging
            target_entity = "General Comment/No Specific Listed Entity" # Default to general
            print(f"Warning: Unrecognized 'primary_target_entity' ('{original_target}') found. "
                  f"Classifying under '{target_entity}'. Comment: '{analysis.get('comment_text', 'N/A')[:50]}...'")
        
        entity_data = sentiment_by_entity[target_entity]
        entity_data["comment_count"] += 1

        if sentiment in ["positive", "negative", "neutral"]:
            entity_data[sentiment] += 1
        else:
            print(f"Warning: Unknown 'sentiment_expressed' ('{sentiment}') for target '{target_entity}'. "
                  f"Defaulting to neutral. Comment: '{analysis.get('comment_text', 'N/A')[:50]}...'")
            entity_data["neutral"] += 1 # Default to neutral

        if isinstance(score, (float, int)):
            entity_data["sum_sentiment_score"] += float(score)
            entity_data["valid_score_count"] += 1
        
        if is_sarcastic:
            entity_data["sarcastic_count"] += 1
            overall_stats["total_sarcastic_comments_overall"] += 1

    # Calculate averages for each entity
    for entity, data in sentiment_by_entity.items():
        if data["valid_score_count"] > 0:
            data["average_sentiment_score"] = data["sum_sentiment_score"] / data["valid_score_count"]
        else:
            data["average_sentiment_score"] = 0.0 # Or None, or handle in printing

    summary_output = {
        "by_entity": sentiment_by_entity,
        "overall_stats": overall_stats
    }

    # Print textual summary
    print(f"\n--- {title} ---")
    print(f"Overall Items Processed: {overall_stats['total_analyzed_items']}")
    print(f"Overall Valid Analyses: {overall_stats['valid_analyses']}")
    print(f"Overall Sarcastic Comments: {overall_stats['total_sarcastic_comments_overall']}")
    print("\nSentiment Breakdown by Primary Target Entity:")

    for entity_name, data in summary_output["by_entity"].items():
        if data["comment_count"] > 0: # Only print entities that had associated comments
            print(f"\n  Entity: {entity_name} ({data['comment_count']} comment(s) primarily targeted)")
            print(f"    Sentiments: Positive: {data['positive']}, Negative: {data['negative']}, Neutral: {data['neutral']}")
            print(f"    Sarcastic comments for this entity: {data['sarcastic_count']}")
            avg_score_display = f"{data['average_sentiment_score']:.2f}" if data['valid_score_count'] > 0 else "N/A"
            print(f"    Average Sentiment Score: {avg_score_display} (from {data['valid_score_count']} scored comments)")
            
    return summary_output

# Agent 0: Filter Channel Videos for Political Content
# Agent 0: Filter Channel Videos for Political Content
def filter_video_for_politics_llm(video_info, fireworks_api_key):
    if not fireworks_api_key:
        print("Fireworks API Key missing for LLM filtering.")
        return None
    llm = ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", fireworks_api_key=fireworks_api_key, temperature=0.0, max_tokens=300)

    # --- FIX APPLIED HERE for Pydantic V2 ---
    # Generate the schema dictionary
    political_video_check_schema_dict = PoliticalVideoCheck.model_json_schema()
    # Convert the schema dictionary to a JSON string
    political_video_check_schema_str = json.dumps(political_video_check_schema_dict, indent=2)
    # Escape the curly braces within the schema string
    escaped_political_video_check_schema = political_video_check_schema_str.replace("{", "{{").replace("}", "}}")
    # --- END FIX ---

    system_prompt = (
        "You are an AI assistant acting as a preliminary filter for a **Malaysian political sentiment analysis project.** "
        "Your task is to determine if a YouTube video, based *only* on its title and description, is **PRIMARILY and SUBSTANTIVELY focused on Malaysian politics**, making it a good candidate for subsequent in-depth sentiment analysis of its content and comments."
        "\n\n"
        "Consider a video relevant if it directly discusses:"
        "  - Malaysian elections (federal or state)."
        "  - Policies, actions, or significant statements of Malaysian political parties (e.g., Pakatan Harapan (PH), Perikatan Nasional (PN), Barisan Nasional (BN), Gabungan Parti Sarawak (GPS), Gabungan Rakyat Sabah (GRS), MUDA, PSM, and other clearly identifiable Malaysian political entities)."
        "  - Malaysian government policies, their debates, and societal impact."
        "  - Speeches, rallies, or significant public appearances by key Malaysian political figures discussing political matters."
        "  - Political news analysis, op-eds, or detailed commentary specifically centered on Malaysian political events or discourse."
        "\n\n"
        "**Crucially, the video should likely contain or elicit discussions, opinions, or sentiments related to these Malaysian political topics.** "
        "Filter out videos that are:"
        "  - Only tangentially related to Malaysian politics (e.g., a lifestyle vlog that briefly mentions a politician)."
        "  - Purely international news with no specific Malaysian angle or impact."
        "  - Clickbait or sensationalized titles that don't reflect substantive political content in the description."
        "  - Primarily about non-political topics (e.g., sports, entertainment, general Malaysian culture) unless there's a very strong, explicit political dimension described."
        "\n\n"
        "The language of the title or description may be English or Malay."
        "Your response MUST be a single, valid JSON object, and NOTHING ELSE. Do not add any explanatory text before or after the JSON. "
        f"The JSON object must strictly adhere to the following Pydantic schema: {escaped_political_video_check_schema}" # Ensure this variable is correctly defined in your scope
    )

    human_prompt = (
        "Video Title: {video_title}\n"
        "Video Description: {video_description}\n\n"
        "Based on the title and description, and the detailed criteria in the system prompt, is this video primarily related to Malaysian politics and suitable for sentiment analysis? "
        "Provide your answer as a JSON object matching the schema. Ensure all fields ('is_political', 'reasoning', 'video_title', 'video_id') are present."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain_llm_part = prompt_template | llm | StrOutputParser()
    json_parser = JsonOutputParser(pydantic_object=PoliticalVideoCheck)

    raw_llm_output = ""
    try:
        raw_llm_output = chain_llm_part.invoke({
            "video_title": video_info.get("title", ""),
            "video_description": video_info.get("description", "")[:1000]
        })
        cleaned_output = clean_llm_json_output(raw_llm_output)
        result = json_parser.parse(cleaned_output)
        result_dict = result if isinstance(result, dict) else result.dict()
        result_dict['video_title'] = video_info.get("title")
        result_dict['video_id'] = video_info.get("video_id")
        return result_dict
    except Exception as e:
        print(f"LLM filtering error for video '{video_info.get('title')}'. \nError: {e}")
        print(f"Raw LLM output that caused parsing error:\n---\n{raw_llm_output}\n---")
        return {"is_political": False,
                "reasoning": f"Error in LLM analysis: {e}. Raw output: {raw_llm_output[:200]}...",
                "video_title": video_info.get("title"),
                "video_id": video_info.get("video_id")}



def agent0_fetch_and_filter_videos(channel_id, youtube_api_key, fireworks_api_key, max_videos_to_scan=25):
    print("\n--- Agent 0: Fetching and Filtering Channel Videos ---")
    if not youtube_api_key:
        print("YouTube API Key missing. Cannot fetch channel videos.")
        return []
    if not fireworks_api_key:
        print("Fireworks API Key missing. Cannot filter videos using LLM.")
        # Optionally, one could return all videos if LLM filtering is not possible,
        # but for this script's purpose, filtering is key.
        return []

    youtube = build('youtube', 'v3', developerKey=youtube_api_key)
    uploads_playlist_id = get_channel_uploads_playlist_id(youtube, channel_id)
    if not uploads_playlist_id:
        print(f"Could not find uploads playlist for channel ID: {channel_id}")
        return []

    print(f"Fetching latest up to {max_videos_to_scan} videos from channel {channel_id} (Playlist: {uploads_playlist_id})...")
    channel_videos = fetch_videos_from_playlist(youtube, uploads_playlist_id, max_results=max_videos_to_scan)

    political_videos = []
    if not channel_videos:
        print("No videos found in the channel's uploads playlist or error fetching.")
        return []

    print(f"\nFiltering {len(channel_videos)} videos for political content using LLM...")
    for i, video_info in enumerate(channel_videos):
        print(f"  Filtering video {i+1}/{len(channel_videos)}: {video_info.get('title', 'N/A')[:70]}...")
        # Add a small delay to avoid hitting rate limits if any are very strict
        # time.sleep(0.5) # Increased delay slightly
        filter_result = filter_video_for_politics_llm(video_info, fireworks_api_key)

        if filter_result and isinstance(filter_result, dict) and filter_result.get('is_political'):
            political_videos.append({
                "title": filter_result.get('video_title'),
                "video_id": filter_result.get('video_id'),
                "url": f"https://www.youtube.com/watch?v={filter_result.get('video_id')}", # Use standard URL
                "reasoning_for_political": filter_result.get('reasoning')
            })
        elif filter_result and isinstance(filter_result, dict):
            print(f"    Video '{filter_result.get('video_title')}' not primarily political. Reason: {filter_result.get('reasoning')}")
        else:
            print(f"    Could not determine political nature for video '{video_info.get('title', 'N/A')}'.")
        time.sleep(1) # Rate limiting for LLM calls

    print(f"\nAgent 0 found {len(political_videos)} politics-related videos out of {len(channel_videos)} scanned.")
    return political_videos

# Agent 1: Transcription
def transcribe_audio_fireworks(mp3_file_path, api_key):
    if not api_key:
        print("Fireworks API Key missing for transcription.")
        return None
    if not os.path.exists(mp3_file_path):
        print(f"MP3 file not found for transcription: {mp3_file_path}")
        return None

    print(f"Starting transcription for: {mp3_file_path} (Fireworks Whisper)...")
    # Ensure you have the `openai` package installed and configured for Fireworks
    client = openai.OpenAI(api_key=api_key, base_url="https://api.fireworks.ai/inference/v1")
    try:
        with open(mp3_file_path, "rb") as audio_file:
            # Using whisper-large-v3 as it's generally more accurate
            # The model name might be just "whisper-large-v3" or "accounts/fireworks/models/whisper-large-v3"
            # Check Fireworks documentation for the exact model identifier.
            transcript_response = client.audio.transcriptions.create(
                model="whisper-v3", # Using a more specific model name
                file=audio_file,
                response_format="text" # Requesting plain text output
            )
        # The response for "text" format is directly the string
        transcribed_text = transcript_response if isinstance(transcript_response, str) else str(transcript_response)

        print("Transcription successful.")
        return transcribed_text.strip()
    except Exception as e:
        print(f"Fireworks Whisper transcription error: {e}")
        # Attempt to get more details from the error if it's an APIError
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"API Error Details: {error_details}")
            except json.JSONDecodeError:
                print(f"API Error Content (not JSON): {e.response.text}")
        elif hasattr(e, 'message'):
             print(f"Error Message: {e.message}")
    return None

# Agent 2: Contextual Summary
def generate_contextual_summary_langchain(transcribed_text, video_title, video_description, api_key):
    if not api_key:
        print("Fireworks API Key missing for summary generation.")
        return None
    if not transcribed_text or transcribed_text.strip() == "":
        print("Transcription text is empty or missing. Cannot generate summary.")
        return "Contextual summary could not be generated due to missing transcription."

    print("Generating contextual summary (Langchain + Fireworks AI)...")
    # Consider models fine-tuned for summarization or instruct models
    llm = ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", fireworks_api_key=api_key, max_tokens=1024, temperature=0.3)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an AI assistant tasked with creating a concise contextual summary of a YouTube video. "
            "The video is related to Malaysian topics, potentially political. "
            "Use the provided video title, description, and the full transcription. "
            "Your summary should highlight the main topics discussed, the overall sentiment or tone of the video (if discernible), and the apparent purpose or key message. "
            "The summary should be neutral and objective, even if the content is opinionated. Keep it to 2-4 sentences. "
            "The original content might be in English or Malay, or a mix. Produce the summary in English."
        )),
        ("human", (
            "Video Title: {video_title}\n"
            "Video Description: {video_description}\n\n"
            "Full Video Transcription:\n\"\"\"\n{transcribed_text}\n\"\"\"\n\n"
            "Based on all the above information, please provide a concise contextual summary (2-4 sentences in English):"
        ))
    ])
    chain = prompt_template | llm | StrOutputParser()
    try:
        summary = chain.invoke({
            "video_title": video_title or "N/A",
            "video_description": (video_description or "N/A")[:1500], # Limit description length
            "transcribed_text": transcribed_text
        })
        print("Contextual summary generated.")
        return summary.strip()
    except Exception as e:
        print(f"Summary generation error: {e}")
    return None

# Agent 3: Political Leniency & Sarcasm Analysis
# TODO: Consider renaming this function later to reflect its new purpose (sentiment analysis).
# Its name is now misleading as it performs sentiment analysis.
def analyze_comment_sentiment_analysis_langchain(comment, contextual_summary, video_title, video_description, api_key):
    # This function now performs TARGETED SENTIMENT ANALYSIS.
    if not api_key:
        print("Fireworks API Key missing for targeted comment sentiment analysis.")
        return { # Return a structure matching TargetedCommentSentiment
            "comment_text": comment.get('comment_text', "Error: API key missing"),
            "is_sarcastic": False, "sarcasm_reasoning": "N/A",
            "primary_target_entity": "General Comment/No Specific Listed Entity",
            "entity_identification_reasoning": "N/A - API Key Missing",
            "sentiment_expressed": "neutral", "sentiment_score": 0.0,
            "overall_reasoning": "Error: Fireworks API Key missing."
        }

    llm = ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", fireworks_api_key=api_key, temperature=0.1, max_tokens=8124) # Increased max_tokens slightly for more complex output

    # Generate the schema dictionary for TargetedCommentSentiment
    targeted_sentiment_schema_dict = TargetedCommentSentiment.model_json_schema()
    targeted_sentiment_schema_str = json.dumps(targeted_sentiment_schema_dict, indent=2)
    escaped_targeted_sentiment_schema = targeted_sentiment_schema_str.replace("{", "{{").replace("}", "}}")

    system_prompt = (
        "You are an expert AI assistant specializing in Targeted Sentiment Analysis of YouTube comments, particularly those related to Malaysian politics. "
        "You will be given a YouTube comment, along with the video's title, description, and a contextual summary. Your goal is to identify the primary political entity targeted by the comment (if any) and the sentiment expressed towards it. "
        "Follow these steps carefully for the provided comment: "
        "1. Identify Sarcasm: Determine if the comment ('comment_text') is sarcastic ('is_sarcastic': boolean). If true, briefly explain in 'sarcasm_reasoning'. "
        "2. Identify Primary Target Entity: Determine the 'primary_target_entity' of the comment's sentiment. "
        f"   - This entity MUST be chosen from the following exact list: {', '.join(TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES)}. "
        "   - If the comment expresses a general sentiment not directed at any single listed entity, or if it targets an individual or group not clearly representing one of the listed entities, you MUST use 'General Comment/No Specific Listed Entity'. "
        "   - Provide your reasoning for this choice in 'entity_identification_reasoning'. "
        "3. Determine Sentiment Towards Target: "
        "   - Classify the 'sentiment_expressed' towards the identified 'primary_target_entity'. This sentiment MUST be one of: 'positive', 'negative', or 'neutral'. "
        "   - Provide a 'sentiment_score' from -1.0 (intensely negative) to 1.0 (intensely positive), with 0.0 being neutral. This score should reflect the intensity of the 'sentiment_expressed'. "
        "   - IMPORTANT: If sarcasm is detected, the 'sentiment_expressed' and 'sentiment_score' MUST reflect the true, underlying sarcastic meaning towards the target. "
        "4. Explain Overall Judgment: Provide a brief 'overall_reasoning' (1-2 sentences) for your sentiment classification ('sentiment_expressed') and 'sentiment_score', specifically in relation to the 'primary_target_entity'. Explicitly state how any detected sarcasm influenced your final judgment. "
        "Your response MUST be a single, valid JSON object, and NOTHING ELSE. Do not add any explanatory text before or after the JSON. "
        f"The JSON object must strictly adhere to this Pydantic schema: {escaped_targeted_sentiment_schema}"
    )

    human_prompt = (
        "Video Title: {video_title}\n"
        "Video Description: {video_description}\n"
        "Contextual Summary of Video: {contextual_summary}\n\n"
        "Analyze the following YouTube Comment for targeted sentiment based on the instructions and schema provided in the system prompt.\n"
        "YouTube Comment:\n\"\"\"\n{comment_text}\n\"\"\"\n\n"
        "Targeted Sentiment Analysis (JSON Object):"
    )

    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    chain_llm_part = prompt_template | llm | StrOutputParser()
    json_parser = JsonOutputParser(pydantic_object=TargetedCommentSentiment) # Use the NEW Pydantic model

    comment_text_to_analyze = comment.get('comment_text', "")
    raw_llm_output = ""

    default_error_output = {
        "comment_text": comment_text_to_analyze, "is_sarcastic": False, "sarcasm_reasoning": "N/A - Analysis error",
        "primary_target_entity": "General Comment/No Specific Listed Entity",
        "entity_identification_reasoning": "N/A - Error during analysis.",
        "sentiment_expressed": "neutral", "sentiment_score": 0.0,
        "overall_reasoning": "Could not perform targeted sentiment analysis due to an error."
    }

    if not comment_text_to_analyze.strip():
        empty_comment_output = default_error_output.copy()
        empty_comment_output["comment_text"] = ""
        empty_comment_output["sarcasm_reasoning"] = "N/A - Comment was empty."
        empty_comment_output["entity_identification_reasoning"] = "Comment was empty."
        empty_comment_output["overall_reasoning"] = "Comment was empty or whitespace only."
        return empty_comment_output

    try:
        raw_llm_output = chain_llm_part.invoke({
            "video_title": video_title or "N/A",
            "video_description": (video_description or "N/A")[:1000],
            "contextual_summary": contextual_summary or "N/A",
            "comment_text": comment_text_to_analyze
        })
        cleaned_output = clean_llm_json_output(raw_llm_output)
        result = json_parser.parse(cleaned_output)
        result_dict = result if isinstance(result, dict) else result.dict()
        result_dict['comment_text'] = comment_text_to_analyze
        return result_dict
    except Exception as e:
        print(f"Targeted sentiment analysis error for comment '{comment_text_to_analyze[:50]}...'. \nError: {e}")
        print(f"Raw LLM output that caused parsing error:\n---\n{raw_llm_output}\n---")
        error_output = default_error_output.copy()
        error_output["sarcasm_reasoning"] = f"LLM Analysis Error. Raw: {raw_llm_output[:100]}"
        error_output["overall_reasoning"] = f"LLM error during targeted sentiment analysis: {str(e)[:100]}"
        return error_output



# --- New Function for ASCII Leniency Summary ---
def generate_and_display_targeted_sentiment_visual(summary_data, max_bar_width=30):
    """
    Generates and displays ASCII bar charts for targeted sentiment distribution per entity.
    Takes summary_data from calculate_targeted_sentiment_summary.
    """
    print("\n--- Targeted Sentiment Distribution Visual ---")

    if not summary_data or not summary_data.get("by_entity"):
        print("No targeted sentiment data available to generate a visual summary.")
        return

    data_by_entity = summary_data["by_entity"]
    overall_stats = summary_data.get("overall_stats", {})

    entities_with_comments = {
        entity: data for entity, data in data_by_entity.items() if data.get("comment_count", 0) > 0
    }

    if not entities_with_comments:
        print("No comments were found primarily targeting any specific entities for visualization.")
        print(f"(Overall Valid Analyses: {overall_stats.get('valid_analyses',0)})")
        return

    print(f"Overall Valid Analyses: {overall_stats.get('valid_analyses',0)}, "
          f"Total Sarcastic Comments: {overall_stats.get('total_sarcastic_comments_overall',0)}")

    sentiment_labels = ["Positive", "Negative", "Neutral"]
    max_sentiment_label_len = max(len(label) for label in sentiment_labels)

    for entity_name, entity_data in entities_with_comments.items():
        avg_score_display = f"{entity_data['average_sentiment_score']:.2f}" if entity_data['valid_score_count'] > 0 else "N/A"
        print(f"\n{entity_name}: "
              f"({entity_data['comment_count']} comments, "
              f"{entity_data['sarcastic_count']} sarcastic, "
              f"Avg Score: {avg_score_display})")

        sentiments_to_display = {
            "Positive": entity_data.get("positive", 0),
            "Negative": entity_data.get("negative", 0),
            "Neutral":  entity_data.get("neutral", 0)
        }
        
        total_classified_for_entity = sum(sentiments_to_display.values())

        if total_classified_for_entity == 0:
            print("    No classified sentiments (positive/negative/neutral) for this entity.")
            continue

        # Determine the maximum count among the sentiment categories for this entity for scaling bars
        max_count_for_entity_bar_scaling = 0
        counts_for_entity = [c for c in sentiments_to_display.values() if isinstance(c, int) and c > 0]
        if counts_for_entity:
            max_count_for_entity_bar_scaling = max(counts_for_entity)
        
        # Define line length for the chart for this entity
        line_len = max_sentiment_label_len + 3 + max_bar_width + 3 + 5 + 10 
        print("  " + "-" * line_len)

        for sentiment_label, count_val in sentiments_to_display.items():
            percentage = (count_val / total_classified_for_entity * 100) if total_classified_for_entity > 0 else 0.0
            
            if max_count_for_entity_bar_scaling > 0: # Avoid division by zero
                bar_length = int((count_val / max_count_for_entity_bar_scaling) * max_bar_width)
            else: # If max_count is 0, all counts for this entity are 0 for pos/neg/neu
                bar_length = 0
                
            bar_char = '█' # U+2588 FULL BLOCK
            bar = bar_char * bar_length
            padding = ' ' * (max_bar_width - bar_length)
            
            # Indent sentiment bars under the entity name
            print(f"  {sentiment_label:<{max_sentiment_label_len}} [{bar}{padding}] {count_val:>4} ({percentage:>5.1f}%)")
        print("  " + "-" * line_len)
        
    print("\n--- End of Visual Summary ---")

# --- Main Orchestration ---
if __name__ == '__main__':
    # --- Configuration ---
    INPUT_CHANNEL_ID = ""
    # Prompt for Channel ID if not hardcoded
    while not INPUT_CHANNEL_ID.startswith("UC") or len(INPUT_CHANNEL_ID) < 10: # Basic validation
        INPUT_CHANNEL_ID = input("Please enter the YouTube Channel ID (starts with UC, e.g., UCXXXX...): ").strip()
        if not INPUT_CHANNEL_ID.startswith("UC"):
            print("Invalid Channel ID format. It must start with 'UC'.")
        elif len(INPUT_CHANNEL_ID) < 10: # Arbitrary minimum length, typical IDs are longer
             print("Channel ID seems too short. Please re-enter.")


    MAX_VIDEOS_TO_SCAN_FROM_CHANNEL = 20 # Reduced for quicker testing, adjust as needed
    COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT = 100 # Or 'all'

    # --- Initial Checks ---
    if not YOUTUBE_API_KEY: print("Error: YOUTUBE_API_KEY not found in environment variables or .env file.")
    if not FIREWORKS_API_KEY: print("Error: FIREWORKS_API_KEY not found. AI functions will be skipped or fail.")

    if not YOUTUBE_API_KEY: # Fireworks key is checked within functions
        print("Exiting due to missing YouTube API key.")
        exit()

    # --- Agent 0: Fetch & Filter Politics-Related Videos ---
    politics_related_videos = agent0_fetch_and_filter_videos(
        INPUT_CHANNEL_ID, YOUTUBE_API_KEY, FIREWORKS_API_KEY, MAX_VIDEOS_TO_SCAN_FROM_CHANNEL
    )

    if not politics_related_videos:
        print("No politics-related videos found or an error occurred with Agent 0. Exiting.")
        exit()

    print("\n--- Politics-Related Videos Found by Agent 0 ---")
    for i, video in enumerate(politics_related_videos):
        print(f"{i+1}. {video.get('title')} (ID: {video.get('video_id')})")
        print(f"   Reason: {video.get('reasoning_for_political', 'N/A')}")
        print(f"   URL: {video.get('url')}")


    # --- User Selection of Videos to Process ---
    videos_to_process_fully = []
    if not politics_related_videos: # Should have exited above, but defensive check
        print("No videos to select from. Exiting.")
        exit()

    while True:
        user_choice_str = input(f"\nEnter video number (1-{len(politics_related_videos)}) to process, "
                                f"'all' for all {len(politics_related_videos)} listed, or 'quit': ").strip().lower()
        if user_choice_str == 'quit':
            print("Exiting.")
            exit()
        elif user_choice_str == 'all':
            videos_to_process_fully = politics_related_videos
            print(f"Selected all {len(videos_to_process_fully)} videos for full analysis.")
            break
        else:
            try:
                choice_index = int(user_choice_str) - 1
                if 0 <= choice_index < len(politics_related_videos):
                    selected_video = politics_related_videos[choice_index]
                    videos_to_process_fully.append(selected_video)
                    print(f"Added video: {selected_video.get('title')} to processing queue.")
                    # Ask if user wants to add more, or process now
                    add_more = input("Add another video? (yes/no/all/quit): ").strip().lower()
                    if add_more == 'no' or add_more == 'quit':
                        break
                    elif add_more == 'all':
                        videos_to_process_fully = politics_related_videos
                        print(f"Selected all {len(videos_to_process_fully)} videos for full analysis.")
                        break
                    # If 'yes' or anything else, the loop continues to ask for another number
                else:
                    print(f"Invalid video number. Please enter a number between 1 and {len(politics_related_videos)}.")
            except ValueError:
                print("Invalid input. Please enter a number, 'all', or 'quit'.")

    if not videos_to_process_fully:
        print("No videos selected for processing. Exiting.")
        exit()

    print(f"\n--- Starting full analysis for {len(videos_to_process_fully)} selected video(s) ---")


    # --- Main Processing Loop for Selected Videos ---
    grand_total_analyzed_comments_data = []

    for video_idx, video_info_from_agent0 in enumerate(videos_to_process_fully):
        current_video_url = video_info_from_agent0.get("url")
        current_video_id = video_info_from_agent0.get("video_id") # Get ID for folder name
        current_video_title_from_agent0 = video_info_from_agent0.get("title", f"Unknown_Video_{current_video_id or 'NoID'}")

        print(f"\n\n=== Processing Video {video_idx + 1}/{len(videos_to_process_fully)}: {current_video_title_from_agent0} ===")
        print(f"URL: {current_video_url}")

        # Fetch details again for consistency, though Agent0 provides some
        video_details_api = fetch_youtube_video_details_and_comments_api(current_video_url, YOUTUBE_API_KEY)
        # Prefer API title if available and different from Agent0's (which might be truncated or slightly off)
        video_title_for_processing = video_details_api.get('title') if video_details_api.get('title') else current_video_title_from_agent0
        video_description_for_processing = video_details_api.get('description')
        original_comments_from_api = video_details_api.get('comments_data', [])

        # --- Folder Setup ---
        base_download_parent_folder = "analysis_data" # Changed folder name
        # Sanitize title for folder name, include video ID for uniqueness
        sanitized_title_part = re.sub(r'[^\w\s-]', '', video_title_for_processing).strip().replace(' ', '_')
        sanitized_title_part = sanitized_title_part[:60] if sanitized_title_part else "Video" # Shorter part
        
        # Use video ID for folder name to ensure uniqueness and avoid issues with long/special char titles
        video_id_for_folder = current_video_id if current_video_id else "UnknownID"
        unique_video_folder_name = f"{video_id_for_folder}_{sanitized_title_part}"

        output_video_data_folder = os.path.join(base_download_parent_folder, unique_video_folder_name)
        try:
            os.makedirs(output_video_data_folder, exist_ok=True)
            print(f"Data for this video will be saved in: {os.path.abspath(output_video_data_folder)}")
        except OSError as e:
            print(f"Error creating output directory '{output_video_data_folder}': {e}. Skipping this video.")
            continue

        # Save initial video info
        save_data_to_file({"title": video_title_for_processing, "description": video_description_for_processing, "url": current_video_url, "id": current_video_id},
                          output_video_data_folder, "video_meta.json")
        if original_comments_from_api:
            save_data_to_file(original_comments_from_api, output_video_data_folder, "original_comments.json")
        else:
            print("No comments found via API for this video.")


        transcribed_text = None
        contextual_summary = None
        # mp3_file_path = None # Declared later

        if FIREWORKS_API_KEY: # Only proceed if key exists
            print(f"\n--- Downloading & Transcribing Audio for '{video_title_for_processing}' ---")
            # Use a fixed name for the audio file within its video-specific folder
            mp3_file_path = download_audio_mp3_yt_dlp(current_video_url, output_video_data_folder, filename_template='audio_track.%(ext)s')
            if mp3_file_path and os.path.exists(mp3_file_path):
                transcribed_text = transcribe_audio_fireworks(mp3_file_path, FIREWORKS_API_KEY)
                if transcribed_text:
                    save_data_to_file(transcribed_text, output_video_data_folder, "transcription.txt")
                else:
                    print("Transcription failed or returned empty.")
            else:
                print(f"Audio download failed for {current_video_url}, skipping transcription.")
        else:
            print("Skipping audio download and transcription as FIREWORKS_API_KEY is not set.")

        if transcribed_text and FIREWORKS_API_KEY:
            print("\n--- Generating Contextual Summary (Agent 2) ---")
            contextual_summary = generate_contextual_summary_langchain(
                transcribed_text, video_title_for_processing, video_description_for_processing, FIREWORKS_API_KEY
            )
            if contextual_summary:
                save_data_to_file(contextual_summary, output_video_data_folder, "contextual_summary.txt")
            else:
                print("Contextual summary generation failed or returned empty.")
        elif FIREWORKS_API_KEY: # If key exists but no transcription
             print("Skipping contextual summary as transcription is unavailable.")


        if original_comments_from_api and FIREWORKS_API_KEY:
            print("\n--- Analyzing Comments for Political Leniency & Sarcasm (Agent 3) ---")
            analyzed_comments_for_this_video = []
            
            comments_to_process_for_video = []
            if isinstance(COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT, str) and COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT.lower() == 'all':
                comments_to_process_for_video = original_comments_from_api
            elif isinstance(COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT, int) and COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT > 0:
                comments_to_process_for_video = original_comments_from_api[:COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT]
            
            if comments_to_process_for_video:
                print(f"Analyzing {len(comments_to_process_for_video)} comments for this video...")
                for i, comment_obj in enumerate(comments_to_process_for_video):
                    if not comment_obj or not isinstance(comment_obj.get('comment_text'), str) or not comment_obj.get('comment_text').strip():
                        print(f"  Skipping invalid or empty comment object at index {i}.")
                        analysis_error_obj = {"comment_text": str(comment_obj.get('comment_text', 'INVALID_COMMENT_DATA')), "is_sarcastic": False, "sarcasm_reasoning": "N/A - Invalid/Empty Comment Data",
                                            "political_entity_lean": "Error", "leniency_score_percent": 0.0,
                                            "reasoning": "Invalid or empty comment data provided to analysis."}
                        analyzed_comments_for_this_video.append(analysis_error_obj)
                        # grand_total_analyzed_comments_data.append(analysis_error_obj) # Add error to grand total
                        continue

                    print(f"  Analyzing comment {i+1}/{len(comments_to_process_for_video)}: \"{comment_obj.get('comment_text', '')[:60].replace(os.linesep, ' ')}...\"")
                    # time.sleep(0.5) # Rate limit before LLM call
                    analysis_result = analyze_comment_sentiment_analysis_langchain(
                        comment_obj, contextual_summary, video_title_for_processing, video_description_for_processing, FIREWORKS_API_KEY
                    )
                    analyzed_comments_for_this_video.append(analysis_result)
                    # grand_total_analyzed_comments_data.append(analysis_result) # Add to grand total immediately
                    time.sleep(1) # Rate limiting for LLM calls

                if analyzed_comments_for_this_video:
                    save_data_to_file(analyzed_comments_for_this_video, output_video_data_folder, "political_leniency_analysis.json")
                    print(f"  Finished analysis for {len(analyzed_comments_for_this_video)} comments for this video.")
                    grand_total_analyzed_comments_data.extend(analyzed_comments_for_this_video) # Add all results for this video to grand total
            else:
                print(f"No comments selected for analysis for this video (config: {COMMENTS_TO_ANALYZE_PER_VIDEO_COUNT}).")
        elif FIREWORKS_API_KEY: # Key exists but no comments
            print("Skipping comment analysis as no comments were fetched or FIREWORKS_API_KEY is missing.")
        
        print(f"\n=== Finished processing video: {video_title_for_processing} ===")
        if video_idx < len(videos_to_process_fully) - 1:
            print("--- Waiting briefly before next video ---")
            time.sleep(5) # Brief pause between processing different videos

    # --- Display Overall Leniency Visual & Summary (after all selected videos are processed) ---
# --- Display Overall Sentiment Summary & Visual (after all selected videos are processed) ---
    if grand_total_analyzed_comments_data: # Ensure this list holds results from the targeted sentiment analysis function
        print("\n\n========================================================")
        print("       OVERALL COMBINED TARGETED SENTIMENT ANALYSIS RESULTS")
        print("========================================================")
        
        # Calculate and print the detailed targeted sentiment summary
        targeted_sentiment_summary_results = calculate_targeted_sentiment_summary(
            grand_total_analyzed_comments_data, 
            title="Grand Total (All Selected Videos) Targeted Sentiment"
        )
        
        # Generate and display the visual representation of the targeted sentiment distribution
        # Check if there's valid data in the summary before trying to visualize
        if (targeted_sentiment_summary_results and 
            targeted_sentiment_summary_results.get("overall_stats", {}).get("valid_analyses", 0) > 0):
            generate_and_display_targeted_sentiment_visual(targeted_sentiment_summary_results)
        else:
            print("\nNo valid targeted sentiment data was processed to display a visual summary.")
            
    else:
        print("\nNo comments were analyzed across all selected videos, so no overall targeted sentiment summary to display.")

    print("\nAll selected videos processed. Script finished.")
    # Ensure base_download_parent_folder is defined if you use this print statement
    # print(f"All data saved in subfolders within: {os.path.abspath(base_download_parent_folder)}")
    print(f"All data saved in subfolders within: {os.path.abspath(base_download_parent_folder)}")

