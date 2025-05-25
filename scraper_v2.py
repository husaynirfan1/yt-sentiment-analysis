#!/usr/bin/env python3
import os
import re
import json
import time
import yt_dlp
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import argparse # Added
import sys # Added

# Langchain and Fireworks AI specific imports
from langchain_fireworks import ChatFireworks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# For Fireworks AI Whisper transcription via OpenAI-compatible API
import openai

# Load environment variables from .env file
load_dotenv()

# --- API KEYS (Defaults from .env, will be overridden by args if provided) ---
_YOUTUBE_API_KEY_FROM_ENV = os.environ.get("YOUTUBE_API_KEY")
_FIREWORKS_API_KEY_FROM_ENV = os.environ.get("FIREWORKS_API_KEY")

# These will hold the effective API keys used by the script
EFFECTIVE_YOUTUBE_API_KEY = None
EFFECTIVE_FIREWORKS_API_KEY = None

# --- FFmpeg Location (Optional) ---
FFMPEG_LOCATION = None # You can set this directly if needed, or add as an arg

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
    "General Comment/No Specific Listed Entity"
]

# --- Pydantic Models (TargetedCommentSentiment, PoliticalVideoCheck, etc. - unchanged) ---
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

class PoliticalVideoCheck(BaseModel):
    is_political: bool = Field(description="True if the video is primarily about Malaysian politics, False otherwise.")
    reasoning: str = Field(description="A brief explanation for the decision.")
    video_title: str = Field(description="The original video title.")
    video_id: str = Field(description="The YouTube video ID.")

# ... (CommentLeniency and CommentSentimentAnalysis models remain the same)

# --- Helper Functions (extract_video_id, save_data_to_file, clean_llm_json_output - largely unchanged) ---
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
    print(f"Warning: Could not extract video ID from URL: {url}", file=sys.stderr) # Print warnings/errors to stderr
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
        print(f"Saved {filename} to: {os.path.abspath(file_path)}", file=sys.stderr)
    except Exception as e:
        print(f"Error saving {filename} to {file_path}: {e}", file=sys.stderr)

def clean_llm_json_output(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):]
    if text.endswith("```"):
        text = text[:-len("```")]
    json_start_index = text.find('{')
    json_end_index = text.rfind('}')
    if json_start_index != -1 and json_end_index != -1 and json_end_index > json_start_index:
        text = text[json_start_index : json_end_index + 1]
    return text.strip()

# --- YouTube API Functions (Modified to use passed API key) ---
def get_channel_uploads_playlist_id(youtube_service_object, channel_id): # Accepts youtube object
    try:
        request = youtube_service_object.channels().list(part="contentDetails", id=channel_id)
        response = request.execute()
        if response.get("items"):
            return response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    except HttpError as e:
        print(f"API error getting uploads playlist ID for channel {channel_id}: {e}", file=sys.stderr)
    return None

def fetch_videos_from_playlist(youtube_service_object, playlist_id, max_results=25): # Accepts youtube object
    videos = []
    next_page_token = None
    try:
        while True:
            request = youtube_service_object.playlistItems().list(
                part="snippet",
                playlistId=playlist_id,
                maxResults=min(max_results - len(videos), 50),
                pageToken=next_page_token
            )
            response = request.execute()
            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                video_id = snippet.get("resourceId", {}).get("videoId")
                title = snippet.get("title")
                description = snippet.get("description")
                if video_id and title:
                    videos.append({
                        "video_id": video_id,
                        "title": title,
                        "description": description,
                        "url": f"https://www.youtube.com/watch?v={video_id}" # Standard URL
                    })
                if len(videos) >= max_results: break
            next_page_token = response.get("nextPageToken")
            if not next_page_token or len(videos) >= max_results: break
        print(f"Fetched {len(videos)} video snippets from channel's uploads playlist.", file=sys.stderr)
    except HttpError as e:
        print(f"API error fetching videos from playlist {playlist_id}: {e}", file=sys.stderr)
    return videos

def fetch_youtube_video_details_and_comments_api(video_url, youtube_api_key_to_use, comments_to_fetch_count_str="100"):
    video_details = {'title': None, 'description': None, 'comments_data': []}
    video_id = extract_video_id(video_url)
    if not video_id: return video_details

    if not youtube_api_key_to_use:
        print("Warning: YouTube API Key not set. Cannot fetch video details or comments.", file=sys.stderr)
        return video_details

    max_comments_to_fetch = 0
    if comments_to_fetch_count_str.lower() == 'all':
        max_comments_to_fetch = float('inf') # Effectively all, YouTube API limits will apply per page
    else:
        try:
            max_comments_to_fetch = int(comments_to_fetch_count_str)
            if max_comments_to_fetch < 0: max_comments_to_fetch = 0 # No negative fetching
        except ValueError:
            print(f"Warning: Invalid comment count '{comments_to_fetch_count_str}'. Defaulting to 0 comments.", file=sys.stderr)
            max_comments_to_fetch = 0 # Default to 0 if invalid

    if max_comments_to_fetch == 0: # If explicitly 0, don't fetch comments
        print(f"Comment fetching is set to 0 for video ID: {video_id}", file=sys.stderr)
        # Still try to get video title and description
        try:
            youtube = build('youtube', 'v3', developerKey=youtube_api_key_to_use)
            video_request = youtube.videos().list(part="snippet", id=video_id)
            video_response = video_request.execute()
            if video_response.get("items"):
                snippet = video_response["items"][0]["snippet"]
                video_details['title'] = snippet.get("title", "No title (API)")
                video_details['description'] = snippet.get("description", "No description (API)")
            else:
                print(f"Could not fetch video details (API) for ID: {video_id}", file=sys.stderr)
        except HttpError as e:
            error_content = e.content.decode('utf-8', 'ignore') if e.content else "No error content"
            print(f"YouTube API HTTP error (details only) {e.resp.status if e.resp else 'N/A'}: {error_content}", file=sys.stderr)
        except Exception as e:
            print(f"Error during YouTube API video detail operations: {e}", file=sys.stderr)
        return video_details


    try:
        youtube = build('youtube', 'v3', developerKey=youtube_api_key_to_use)
        video_request = youtube.videos().list(part="snippet", id=video_id)
        video_response = video_request.execute()
        if video_response.get("items"):
            snippet = video_response["items"][0]["snippet"]
            video_details['title'] = snippet.get("title", "No title (API)")
            video_details['description'] = snippet.get("description", "No description (API)")
        else:
            print(f"Could not fetch video details (API) for ID: {video_id}", file=sys.stderr)

        if max_comments_to_fetch > 0: # Only proceed if we need to fetch comments
            next_page_token = None
            comments_fetched_count = 0
            while True:
                # Determine maxResults for this page
                remaining_to_fetch = max_comments_to_fetch - comments_fetched_count
                results_this_page = min(100, remaining_to_fetch if max_comments_to_fetch != float('inf') else 100)
                if results_this_page <= 0 and max_comments_to_fetch != float('inf'): break


                comment_request = youtube.commentThreads().list(
                    part='snippet,replies', videoId=video_id, maxResults=results_this_page,
                    pageToken=next_page_token, textFormat='plainText', order='relevance')
                comment_response = comment_request.execute()

                for item in comment_response.get('items', []):
                    if comments_fetched_count >= max_comments_to_fetch: break
                    top_comment_snippet = item.get('snippet', {}).get('topLevelComment', {}).get('snippet', {})
                    if top_comment_snippet.get('textDisplay'):
                        video_details['comments_data'].append({
                            'author': top_comment_snippet.get('authorDisplayName', 'Unknown Author'),
                            'comment_text': top_comment_snippet.get('textDisplay')
                        })
                        comments_fetched_count += 1
                    
                    if comments_fetched_count >= max_comments_to_fetch: break

                    if item.get('replies'):
                        for reply in item['replies']['comments']:
                            if comments_fetched_count >= max_comments_to_fetch: break
                            reply_snippet = reply.get('snippet', {})
                            if reply_snippet.get('textDisplay'):
                                video_details['comments_data'].append({
                                    'author': f"    (Reply to {top_comment_snippet.get('authorDisplayName', 'Original Author')}) {reply_snippet.get('authorDisplayName', 'Unknown Author')}",
                                    'comment_text': reply_snippet.get('textDisplay')
                                })
                                comments_fetched_count += 1
                
                next_page_token = comment_response.get('nextPageToken')
                if not next_page_token or comments_fetched_count >= max_comments_to_fetch:
                    break
            
            if video_details['comments_data']:
                print(f"Extracted {len(video_details['comments_data'])} comments (API) for {video_id}.", file=sys.stderr)
            else:
                print(f"No comments found or extracted for video ID: {video_id}", file=sys.stderr)
    except HttpError as e:
        error_content = e.content.decode('utf-8', 'ignore') if e.content else "No error content"
        print(f"YouTube API HTTP error {e.resp.status if e.resp else 'N/A'}: {error_content}", file=sys.stderr)
    except Exception as e:
        print(f"Error during YouTube API operations: {e}", file=sys.stderr)
    return video_details


# --- yt-dlp Audio Download (largely unchanged, ensure FFMPEG_LOCATION if needed) ---
def download_audio_mp3_yt_dlp(video_url, output_path='.', filename_template='audio.%(ext)s'):
    base_filename = os.path.splitext(filename_template)[0]
    actual_filename_mp3 = os.path.join(output_path, f"{base_filename}.mp3")

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
        'outtmpl': os.path.join(output_path, base_filename),
        'quiet': True, 'noplaylist': True, 'nocheckcertificate': True, # quiet True
        'ignoreerrors': True,
    }
    if FFMPEG_LOCATION: ydl_opts['ffmpeg_location'] = FFMPEG_LOCATION

    try:
        print(f"\nDownloading audio for: {video_url} to {os.path.abspath(output_path)}", file=sys.stderr)
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        if os.path.exists(actual_filename_mp3):
            print(f"Audio downloaded and converted to MP3: {actual_filename_mp3}", file=sys.stderr)
            return actual_filename_mp3
        else:
            # Fallback for sanitized names (less likely with fixed outtmpl base)
            found_mp3s = [f for f in os.listdir(output_path) if f.lower().endswith(".mp3") and f.lower().startswith(base_filename.lower())]
            if found_mp3s:
                found_path = os.path.join(output_path, found_mp3s[0])
                print(f"Audio downloaded. Found MP3 (possibly sanitized name): {found_path}", file=sys.stderr)
                return found_path
            print(f"Error: MP3 file not found at {actual_filename_mp3} or matching pattern in {output_path}.", file=sys.stderr)
            return None
    except Exception as e:
        print(f"Error during audio download for {video_url}: {e}", file=sys.stderr)
    return None

# --- AI Agents (Modified to accept fireworks_api_key) ---
def calculate_targeted_sentiment_summary(analyses_list, title="Targeted Sentiment Summary"):
    # (Function content largely unchanged, but ensure it prints to stderr or returns data for main to handle)
    # For simplicity, this function will continue to print to stderr for its verbose output.
    # The JSON data it returns will be used for file saving by the caller.
    if not analyses_list:
        print(f"\n--- {title} ---", file=sys.stderr)
        print("No targeted sentiment analyses provided to summarize.", file=sys.stderr)
        return {
            "by_entity": {entity: {"positive": 0, "negative": 0, "neutral": 0, "sarcastic_count": 0,
                                   "sum_sentiment_score": 0.0, "valid_score_count": 0,
                                   "comment_count": 0, "average_sentiment_score": 0.0}
                          for entity in TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES},
            "overall_stats": {
                "total_analyzed_items": 0,
                "valid_analyses": 0, 
                "total_sarcastic_comments_overall": 0,
            }
        }

    sentiment_by_entity = {
        entity: {
            "positive": 0, "negative": 0, "neutral": 0,
            "sarcastic_count": 0,
            "sum_sentiment_score": 0.0, "valid_score_count": 0,
            "comment_count": 0, 
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
            print(f"Skipping invalid analysis item (not a dict): {analysis}", file=sys.stderr)
            continue

        overall_stats["valid_analyses"] += 1
        target_entity = analysis.get('primary_target_entity')
        sentiment = analysis.get('sentiment_expressed', 'neutral').lower()
        score = analysis.get('sentiment_score')
        is_sarcastic = analysis.get('is_sarcastic', False)

        if target_entity not in TARGETABLE_MALAYSIAN_POLITICAL_ENTITIES:
            original_target = target_entity 
            target_entity = "General Comment/No Specific Listed Entity" 
            print(f"Warning: Unrecognized 'primary_target_entity' ('{original_target}') found. "
                  f"Classifying under '{target_entity}'. Comment: '{analysis.get('comment_text', 'N/A')[:50]}...'", file=sys.stderr)
        
        entity_data = sentiment_by_entity[target_entity]
        entity_data["comment_count"] += 1

        if sentiment in ["positive", "negative", "neutral"]:
            entity_data[sentiment] += 1
        else:
            print(f"Warning: Unknown 'sentiment_expressed' ('{sentiment}') for target '{target_entity}'. "
                  f"Defaulting to neutral. Comment: '{analysis.get('comment_text', 'N/A')[:50]}...'", file=sys.stderr)
            entity_data["neutral"] += 1

        if isinstance(score, (float, int)):
            entity_data["sum_sentiment_score"] += float(score)
            entity_data["valid_score_count"] += 1
        
        if is_sarcastic:
            entity_data["sarcastic_count"] += 1
            overall_stats["total_sarcastic_comments_overall"] += 1

    for entity, data in sentiment_by_entity.items():
        if data["valid_score_count"] > 0:
            data["average_sentiment_score"] = data["sum_sentiment_score"] / data["valid_score_count"]
        else:
            data["average_sentiment_score"] = 0.0

    summary_output_dict = { # This dictionary will be returned
        "by_entity": sentiment_by_entity,
        "overall_stats": overall_stats
    }

    # Print textual summary to stderr
    print(f"\n--- {title} (Printed to Stderr) ---", file=sys.stderr)
    print(f"Overall Items Processed: {overall_stats['total_analyzed_items']}", file=sys.stderr)
    print(f"Overall Valid Analyses: {overall_stats['valid_analyses']}", file=sys.stderr)
    print(f"Overall Sarcastic Comments: {overall_stats['total_sarcastic_comments_overall']}", file=sys.stderr)
    print("\nSentiment Breakdown by Primary Target Entity:", file=sys.stderr)

    for entity_name, data in summary_output_dict["by_entity"].items():
        if data["comment_count"] > 0:
            print(f"\n  Entity: {entity_name} ({data['comment_count']} comment(s) primarily targeted)", file=sys.stderr)
            print(f"    Sentiments: Positive: {data['positive']}, Negative: {data['negative']}, Neutral: {data['neutral']}", file=sys.stderr)
            print(f"    Sarcastic comments for this entity: {data['sarcastic_count']}", file=sys.stderr)
            avg_score_display = f"{data['average_sentiment_score']:.2f}" if data['valid_score_count'] > 0 else "N/A"
            print(f"    Average Sentiment Score: {avg_score_display} (from {data['valid_score_count']} scored comments)", file=sys.stderr)
            
    return summary_output_dict


def filter_video_for_politics_llm(video_info, fireworks_api_key_to_use): # Param name updated
    if not fireworks_api_key_to_use:
        print("Fireworks API Key missing for LLM filtering.", file=sys.stderr)
        # Return a structure that indicates failure but includes original info
        return {"is_political": False,
                "reasoning": "Error: Fireworks API Key missing for filtering.",
                "video_title": video_info.get("title"),
                "video_id": video_info.get("video_id")}

    llm = ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", fireworks_api_key=fireworks_api_key_to_use, temperature=0.0, max_tokens=300) # Updated model
    political_video_check_schema_dict = PoliticalVideoCheck.model_json_schema()
    political_video_check_schema_str = json.dumps(political_video_check_schema_dict, indent=2)
    escaped_political_video_check_schema = political_video_check_schema_str.replace("{", "{{").replace("}", "}}")
    
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
        f"The JSON object must strictly adhere to the following Pydantic schema: {escaped_political_video_check_schema}"
    )
    human_prompt = (
        "Video Title: {video_title}\n"
        "Video Description: {video_description}\n\n"
        "Based on the title and description, and the detailed criteria in the system prompt, is this video primarily related to Malaysian politics and suitable for sentiment analysis? "
        "Provide your answer as a JSON object matching the schema. Ensure all fields ('is_political', 'reasoning', 'video_title', 'video_id') are present."
    )
    prompt_template = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", human_prompt)])
    # Use JsonOutputParser directly with the chain for Pydantic v2
    chain_llm_part = prompt_template | llm | JsonOutputParser(pydantic_object=PoliticalVideoCheck)

    raw_llm_output_for_debug = "" # For debugging if parsing fails
    try:
        # parsed_output is what JsonOutputParser returns
        parsed_output = chain_llm_part.invoke({
            "video_title": video_info.get("title", ""),
            "video_description": video_info.get("description", "")[:1000] # Limit description length
        })

        result_dict = None # Initialize

        if isinstance(parsed_output, dict):
            result_dict = parsed_output # It's already a dict
        elif isinstance(parsed_output, PoliticalVideoCheck): # PoliticalVideoCheck is your Pydantic model
            # It's a Pydantic model instance, convert to dict
            if hasattr(parsed_output, 'model_dump'): # Pydantic V2
                result_dict = parsed_output.model_dump()
            elif hasattr(parsed_output, 'dict'): # Pydantic V1
                result_dict = parsed_output.dict()
            else:
                print(f"Error: Pydantic model {type(parsed_output)} has no model_dump/dict method.", file=sys.stderr)
                # Fallback: create a dict with error info
                result_dict = {"is_political": False, "reasoning": "Pydantic model conversion error"}
        else:
            print(f"Error: LLM JsonOutputParser returned unexpected type: {type(parsed_output)}", file=sys.stderr)
            result_dict = {"is_political": False, "reasoning": f"LLM output parsing error, unexpected type: {type(parsed_output)}"}

        # Ensure the final dictionary has the required structure, including defaults if parsing failed partially
        final_result = {
            "is_political": result_dict.get('is_political', False),
            "reasoning": result_dict.get('reasoning', "Reasoning missing or error during parsing."),
            "video_title": video_info.get("title"), # Ensure these are always present
            "video_id": video_info.get("video_id")
        }
        return final_result

    except Exception as e:
        # This block catches other errors during the invoke() or if the above logic fails unexpectedly
        str_parser_for_debug = StrOutputParser()
        try:
            # Attempt to get raw output for debugging if the chain invocation itself failed earlier
            raw_llm_output_for_debug = (prompt_template | llm | str_parser_for_debug).invoke({
                "video_title": video_info.get("title", ""),
                "video_description": video_info.get("description", "")[:1000]
            })
        except Exception as dbg_e: # Catch errors during this debug attempt
            raw_llm_output_for_debug = f"Could not retrieve raw LLM output for debugging. Debug error: {dbg_e}"

        print(f"LLM filtering error for video '{video_info.get('title')}'. \nOriginal Error: {e}", file=sys.stderr)
        print(f"Raw LLM output (if retrievable) that might have caused error:\n---\n{raw_llm_output_for_debug}\n---", file=sys.stderr)
        # Return the standard error structure
        return {"is_political": False,
                "reasoning": f"Error in LLM analysis: {e}. Raw output sample: {raw_llm_output_for_debug[:200]}...",
                "video_title": video_info.get("title"),
                "video_id": video_info.get("video_id")}


def agent0_fetch_and_filter_videos(channel_id, youtube_api_key_to_use, fireworks_api_key_to_use, max_videos_to_scan=25):
    print("\n--- Agent 0: Fetching and Filtering Channel Videos ---", file=sys.stderr)
    if not youtube_api_key_to_use:
        print("YouTube API Key missing. Cannot fetch channel videos.", file=sys.stderr)
        return []
    # Fireworks API key check is handled by filter_video_for_politics_llm

    youtube_service = build('youtube', 'v3', developerKey=youtube_api_key_to_use)
    uploads_playlist_id = get_channel_uploads_playlist_id(youtube_service, channel_id)
    if not uploads_playlist_id:
        print(f"Could not find uploads playlist for channel ID: {channel_id}", file=sys.stderr)
        return []

    print(f"Fetching latest up to {max_videos_to_scan} videos from channel {channel_id} (Playlist: {uploads_playlist_id})...", file=sys.stderr)
    channel_videos = fetch_videos_from_playlist(youtube_service, uploads_playlist_id, max_results=max_videos_to_scan)

    political_videos = []
    if not channel_videos:
        print("No videos found in the channel's uploads playlist or error fetching.", file=sys.stderr)
        return []

    print(f"\nFiltering {len(channel_videos)} videos for political content using LLM...", file=sys.stderr)
    for i, video_info in enumerate(channel_videos):
        print(f"  Filtering video {i+1}/{len(channel_videos)}: {video_info.get('title', 'N/A')[:70]}...", file=sys.stderr)
        # Pass the fireworks_api_key_to_use to the filtering function
        filter_result = filter_video_for_politics_llm(video_info, fireworks_api_key_to_use)

        if filter_result and isinstance(filter_result, dict) and filter_result.get('is_political'):
            political_videos.append({
                "title": filter_result.get('video_title'), # Already set by filter_video_for_politics_llm
                "video_id": filter_result.get('video_id'), # Already set
                "url": f"https://www.youtube.com/watch?v={filter_result.get('video_id')}",
                "reasoning_for_political": filter_result.get('reasoning'),
                "description": video_info.get("description", "") # Keep original description if needed later
            })
        elif filter_result and isinstance(filter_result, dict):
            print(f"    Video '{filter_result.get('video_title')}' not primarily political. Reason: {filter_result.get('reasoning')}", file=sys.stderr)
        else:
            print(f"    Could not determine political nature for video '{video_info.get('title', 'N/A')}'.", file=sys.stderr)
        time.sleep(1) # Rate limiting for LLM calls (per video)

    print(f"\nAgent 0 found {len(political_videos)} politics-related videos out of {len(channel_videos)} scanned.", file=sys.stderr)
    return political_videos


def transcribe_audio_fireworks(mp3_file_path, fireworks_api_key_to_use): # Param updated
    if not fireworks_api_key_to_use:
        print("Fireworks API Key missing for transcription.", file=sys.stderr)
        return None
    if not os.path.exists(mp3_file_path):
        print(f"MP3 file not found for transcription: {mp3_file_path}", file=sys.stderr)
        return None

    print(f"Starting transcription for: {mp3_file_path} (Fireworks Whisper)...", file=sys.stderr)
    client = openai.OpenAI(api_key=fireworks_api_key_to_use, base_url="https://api.fireworks.ai/inference/v1")
    try:
        with open(mp3_file_path, "rb") as audio_file:
            # Model updated based on Fireworks common model naming
            transcript_response = client.audio.transcriptions.create(
                model="whisper-v3", # More specific model name
                file=audio_file,
                response_format="text"
            )
        transcribed_text = transcript_response if isinstance(transcript_response, str) else str(transcript_response)
        print("Transcription successful.", file=sys.stderr)
        return transcribed_text.strip()
    except Exception as e:
        print(f"Fireworks Whisper transcription error: {e}", file=sys.stderr)
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_details = e.response.json()
                print(f"API Error Details: {error_details}", file=sys.stderr)
            except json.JSONDecodeError:
                print(f"API Error Content (not JSON): {e.response.text}", file=sys.stderr)
        elif hasattr(e, 'message'):
             print(f"Error Message: {e.message}", file=sys.stderr)
    return None


def generate_contextual_summary_langchain(transcribed_text, video_title, video_description, fireworks_api_key_to_use): # Param updated
    if not fireworks_api_key_to_use:
        print("Fireworks API Key missing for summary generation.", file=sys.stderr)
        return None
    if not transcribed_text or transcribed_text.strip() == "":
        print("Transcription text is empty or missing. Cannot generate summary.", file=sys.stderr)
        return "Contextual summary could not be generated due to missing transcription."

    print("Generating contextual summary (Langchain + Fireworks AI)...", file=sys.stderr)
    llm = ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", fireworks_api_key=fireworks_api_key_to_use, max_tokens=1024, temperature=0.3) # Updated model
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
            "video_description": (video_description or "N/A")[:1500],
            "transcribed_text": transcribed_text
        })
        print("Contextual summary generated.", file=sys.stderr)
        return summary.strip()
    except Exception as e:
        print(f"Summary generation error: {e}", file=sys.stderr)
    return None

def analyze_comment_sentiment_analysis_langchain(comment, contextual_summary, video_title, video_description, fireworks_api_key_to_use): # Param updated
    if not fireworks_api_key_to_use:
        print("Fireworks API Key missing for targeted comment sentiment analysis.", file=sys.stderr)
        return {
            "comment_text": comment.get('comment_text', "Error: API key missing"),
            "is_sarcastic": False, "sarcasm_reasoning": "N/A",
            "primary_target_entity": "General Comment/No Specific Listed Entity",
            "entity_identification_reasoning": "N/A - API Key Missing",
            "sentiment_expressed": "neutral", "sentiment_score": 0.0,
            "overall_reasoning": "Error: Fireworks API Key missing."
        }

    llm = ChatFireworks(model="accounts/fireworks/models/llama4-maverick-instruct-basic", fireworks_api_key=fireworks_api_key_to_use, temperature=0.1, max_tokens=1024) # Updated model, max_tokens adjusted
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
    chain_llm_part = prompt_template | llm | JsonOutputParser(pydantic_object=TargetedCommentSentiment)

    comment_text_to_analyze = comment.get('comment_text', "") # Ensure this is defined
    # Define default_error_output specific to this function's Pydantic model
    default_error_output = {
        "comment_text": comment_text_to_analyze, "is_sarcastic": False, "sarcasm_reasoning": "N/A - Analysis error",
        "primary_target_entity": "General Comment/No Specific Listed Entity",
        "entity_identification_reasoning": "N/A - Error during analysis.",
        "sentiment_expressed": "neutral", "sentiment_score": 0.0,
        "overall_reasoning": "Could not perform targeted sentiment analysis due to an error."
    }
    if not comment_text_to_analyze.strip(): # Handle empty comments earlier
        empty_comment_output = default_error_output.copy()
        empty_comment_output["comment_text"] = "" # Override if it was None
        empty_comment_output["sarcasm_reasoning"] = "N/A - Comment was empty."
        empty_comment_output["entity_identification_reasoning"] = "Comment was empty."
        empty_comment_output["overall_reasoning"] = "Comment was empty or whitespace only."
        return empty_comment_output

    raw_llm_output_for_debug = ""
    try:
        parsed_output = chain_llm_part.invoke({
            "video_title": video_title or "N/A",
            "video_description": (video_description or "N/A")[:1000],
            "contextual_summary": contextual_summary or "N/A",
            "comment_text": comment_text_to_analyze
        })

        result_dict = None # Initialize

        if isinstance(parsed_output, dict):
            result_dict = parsed_output # It's already a dict
        elif isinstance(parsed_output, TargetedCommentSentiment): # Your Pydantic model
            if hasattr(parsed_output, 'model_dump'): # Pydantic V2
                result_dict = parsed_output.model_dump()
            elif hasattr(parsed_output, 'dict'): # Pydantic V1
                result_dict = parsed_output.dict()
            else:
                print(f"Error: Pydantic model {type(parsed_output)} has no model_dump/dict method.", file=sys.stderr)
                result_dict = default_error_output.copy()
                result_dict["overall_reasoning"] = "Pydantic model conversion error"
        else:
            print(f"Error: LLM JsonOutputParser returned unexpected type: {type(parsed_output)}", file=sys.stderr)
            result_dict = default_error_output.copy()
            result_dict["overall_reasoning"] = f"LLM output parsing error, unexpected type: {type(parsed_output)}"
        
        # Ensure the original comment_text is in the final dict
        result_dict['comment_text'] = comment_text_to_analyze
        return result_dict

    except Exception as e:
        str_parser_for_debug = StrOutputParser()
        try:
            raw_llm_output_for_debug = (prompt_template | llm | str_parser_for_debug).invoke({
                "video_title": video_title or "N/A",
                "video_description": (video_description or "N/A")[:1000],
                "contextual_summary": contextual_summary or "N/A",
                "comment_text": comment_text_to_analyze
            })
        except Exception as dbg_e:
            raw_llm_output_for_debug = f"Could not retrieve raw LLM output for debugging. Debug error: {dbg_e}"

        print(f"Targeted sentiment analysis error for comment '{comment_text_to_analyze[:50]}...'. \nOriginal Error: {e}", file=sys.stderr)
        print(f"Raw LLM output (if retrievable) that might have caused error:\n---\n{raw_llm_output_for_debug}\n---", file=sys.stderr)
        
        error_output_on_exception = default_error_output.copy()
        error_output_on_exception["sarcasm_reasoning"] = f"LLM Analysis Exception. Raw: {raw_llm_output_for_debug[:100]}"
        error_output_on_exception["overall_reasoning"] = f"LLM exception during targeted sentiment analysis: {str(e)[:100]}"
        return error_output_on_exception

# --- Modified Function for ASCII Leniency Summary (to return string) ---
def generate_targeted_sentiment_visual_str(summary_data, max_bar_width=30):
    output_lines = []
    output_lines.append("--- Targeted Sentiment Distribution Visual ---")

    if not summary_data or not summary_data.get("by_entity"):
        output_lines.append("No targeted sentiment data available to generate a visual summary.")
        return "\n".join(output_lines)

    data_by_entity = summary_data["by_entity"]
    overall_stats = summary_data.get("overall_stats", {})
    entities_with_comments = {
        entity: data for entity, data in data_by_entity.items() if data.get("comment_count", 0) > 0
    }

    if not entities_with_comments:
        output_lines.append("No comments were found primarily targeting any specific entities for visualization.")
        output_lines.append(f"(Overall Valid Analyses: {overall_stats.get('valid_analyses',0)})")
        return "\n".join(output_lines)

    output_lines.append(f"Overall Valid Analyses: {overall_stats.get('valid_analyses',0)}, "
                        f"Total Sarcastic Comments: {overall_stats.get('total_sarcastic_comments_overall',0)}")

    sentiment_labels = ["Positive", "Negative", "Neutral"]
    max_sentiment_label_len = max(len(label) for label in sentiment_labels)

    for entity_name, entity_data in entities_with_comments.items():
        avg_score_display = f"{entity_data['average_sentiment_score']:.2f}" if entity_data['valid_score_count'] > 0 else "N/A"
        output_lines.append(f"\n{entity_name}: "
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
            output_lines.append("    No classified sentiments (positive/negative/neutral) for this entity.")
            continue

        max_count_for_entity_bar_scaling = max(c for c in sentiments_to_display.values() if isinstance(c, int) and c > 0) if \
                                           any(c > 0 for c in sentiments_to_display.values() if isinstance(c,int)) else 1 # Avoid div by zero
        
        line_len = max_sentiment_label_len + 3 + max_bar_width + 3 + 5 + 10 
        output_lines.append("  " + "-" * line_len)

        for sentiment_label, count_val in sentiments_to_display.items():
            percentage = (count_val / total_classified_for_entity * 100) if total_classified_for_entity > 0 else 0.0
            bar_length = int((count_val / max_count_for_entity_bar_scaling) * max_bar_width) if max_count_for_entity_bar_scaling > 0 else 0
            bar_char = '█'
            bar = bar_char * bar_length
            padding = ' ' * (max_bar_width - bar_length)
            output_lines.append(f"  {sentiment_label:<{max_sentiment_label_len}} [{bar}{padding}] {count_val:>4} ({percentage:>5.1f}%)")
        output_lines.append("  " + "-" * line_len)
        
    output_lines.append("\n--- End of Visual Summary ---")
    return "\n".join(output_lines)


# --- Main Orchestration Logic ---
def main():
    global EFFECTIVE_YOUTUBE_API_KEY, EFFECTIVE_FIREWORKS_API_KEY # Allow modification of globals

    parser = argparse.ArgumentParser(description="YouTube Video Scraper and Analyzer for Political Sentiment.")
    parser.add_argument("--action", choices=["fetch_videos", "analyze_videos"], required=True,
                        help="Action to perform: fetch video list or analyze videos.")
    parser.add_argument("--channel-id", help="YouTube Channel ID (for fetch_videos).")
    parser.add_argument("--youtube-api-key", help="YouTube API Key (overrides .env).")
    parser.add_argument("--fireworks-api-key", help="Fireworks API Key (overrides .env).")
    parser.add_argument("--max-videos-scan", type=int, default=20,
                        help="Maximum number of recent videos to scan from the channel (for fetch_videos).")
    parser.add_argument("--input-file", help="Path to JSON file containing video list (for analyze_videos).")
    parser.add_argument("--output-dir", default="analysis_results_scraper",
                        help="Directory to save analysis results (for analyze_videos).")
    parser.add_argument("--comments-to-analyze", default="100", # String, can be "all" or number
                        help="Number of comments to analyze per video, or 'all' (for analyze_videos).")
    
    args = parser.parse_args()

    # Set effective API keys
    EFFECTIVE_YOUTUBE_API_KEY = args.youtube_api_key if args.youtube_api_key else _YOUTUBE_API_KEY_FROM_ENV
    EFFECTIVE_FIREWORKS_API_KEY = args.fireworks_api_key if args.fireworks_api_key else _FIREWORKS_API_KEY_FROM_ENV

    if not EFFECTIVE_YOUTUBE_API_KEY:
        print("Error: YouTube API Key is required, either via --youtube-api-key or .env file (YOUTUBE_API_KEY).", file=sys.stderr)
        # For fetch_videos, it's critical. For analyze_videos, it's also critical for fetching comments.
        sys.exit(1)
    # Fireworks API key is optional; features will be skipped if not provided.
    if not EFFECTIVE_FIREWORKS_API_KEY:
        print("Warning: Fireworks API Key is not set (via --fireworks-api-key or .env FIREWORKS_API_KEY). AI-based features (filtering, transcription, summary, sentiment) will be skipped or may fail.", file=sys.stderr)


    if args.action == "fetch_videos":
        if not args.channel_id:
            print("Error: --channel-id is required for fetch_videos action.", file=sys.stderr)
            sys.exit(1)
        
        political_videos_data = agent0_fetch_and_filter_videos(
            args.channel_id, 
            EFFECTIVE_YOUTUBE_API_KEY, 
            EFFECTIVE_FIREWORKS_API_KEY, 
            args.max_videos_scan
        )
        # Output JSON to stdout for gui_app.py
        print(json.dumps(political_videos_data, indent=4))

    elif args.action == "analyze_videos":
        if not args.input_file:
            print("Error: --input-file is required for analyze_videos action.", file=sys.stderr)
            sys.exit(1)
        
        try:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                videos_to_process_fully = json.load(f)
        except Exception as e:
            print(f"Error reading input file {args.input_file}: {e}", file=sys.stderr)
            sys.exit(1)

        if not videos_to_process_fully:
            print("No videos found in the input file to analyze.", file=sys.stderr)
            sys.exit(0)

        os.makedirs(args.output_dir, exist_ok=True)
        grand_total_analyzed_comments_data = []

        comments_to_analyze_val_str = args.comments_to_analyze.strip()
        # fetch_youtube_video_details_and_comments_api will handle parsing this string

        for video_idx, video_info_from_input in enumerate(videos_to_process_fully):
            current_video_url = video_info_from_input.get("url")
            current_video_id = video_info_from_input.get("video_id")
            current_video_title_from_input = video_info_from_input.get("title", f"Unknown_Video_{current_video_id or 'NoID'}")

            print(f"\n\n=== Processing Video {video_idx + 1}/{len(videos_to_process_fully)}: {current_video_title_from_input} (ID: {current_video_id}) ===", file=sys.stderr)
            print(f"URL: {current_video_url}", file=sys.stderr)

            # Fetch details & comments (pass EFFECTIVE_YOUTUBE_API_KEY and comments_to_analyze_val_str)
            video_details_api = fetch_youtube_video_details_and_comments_api(
                current_video_url, 
                EFFECTIVE_YOUTUBE_API_KEY,
                comments_to_analyze_val_str # Pass the count/ "all" string here
            )
            video_title_for_processing = video_details_api.get('title') if video_details_api.get('title') else current_video_title_from_input
            video_description_for_processing = video_details_api.get('description')
            original_comments_from_api = video_details_api.get('comments_data', [])
            
            # --- Folder Setup ---
            sanitized_title_part = re.sub(r'[^\w\s-]', '', video_title_for_processing).strip().replace(' ', '_')
            sanitized_title_part = sanitized_title_part[:60] if sanitized_title_part else "Video"
            video_id_for_folder = current_video_id if current_video_id else "UnknownID"
            unique_video_folder_name = f"{video_id_for_folder}_{sanitized_title_part}"
            output_video_data_folder = os.path.join(args.output_dir, unique_video_folder_name)
            
            try:
                os.makedirs(output_video_data_folder, exist_ok=True)
                print(f"Data for this video will be saved in: {os.path.abspath(output_video_data_folder)}", file=sys.stderr)
            except OSError as e:
                print(f"Error creating output directory '{output_video_data_folder}': {e}. Skipping this video.", file=sys.stderr)
                continue

            save_data_to_file({"title": video_title_for_processing, "description": video_description_for_processing, "url": current_video_url, "id": current_video_id},
                              output_video_data_folder, "video_meta.json")
            if original_comments_from_api:
                save_data_to_file(original_comments_from_api, output_video_data_folder, "original_comments.json")
            else:
                print("No comments found via API or to process for this video.", file=sys.stderr)

            transcribed_text = None
            contextual_summary = None

            if EFFECTIVE_FIREWORKS_API_KEY: # Only proceed if key exists
                print(f"\n--- Downloading & Transcribing Audio for '{video_title_for_processing}' ---", file=sys.stderr)
                mp3_file_path = download_audio_mp3_yt_dlp(current_video_url, output_video_data_folder, filename_template='audio_track.%(ext)s')
                if mp3_file_path and os.path.exists(mp3_file_path):
                    transcribed_text = transcribe_audio_fireworks(mp3_file_path, EFFECTIVE_FIREWORKS_API_KEY)
                    if transcribed_text:
                        save_data_to_file(transcribed_text, output_video_data_folder, "transcription.txt")
                    else:
                        print("Transcription failed or returned empty.", file=sys.stderr)
                else:
                    print(f"Audio download failed for {current_video_url}, skipping transcription.", file=sys.stderr)

                if transcribed_text:
                    print("\n--- Generating Contextual Summary (Agent 2) ---", file=sys.stderr)
                    contextual_summary = generate_contextual_summary_langchain(
                        transcribed_text, video_title_for_processing, video_description_for_processing, EFFECTIVE_FIREWORKS_API_KEY
                    )
                    if contextual_summary:
                        save_data_to_file(contextual_summary, output_video_data_folder, "contextual_summary.txt")
                    else:
                        print("Contextual summary generation failed or returned empty.", file=sys.stderr)
                elif EFFECTIVE_FIREWORKS_API_KEY : # Key exists but no transcription
                    print("Skipping contextual summary as transcription is unavailable.", file=sys.stderr)
            else: # No Fireworks key
                print("Skipping audio download, transcription, and contextual summary as FIREWORKS_API_KEY is not set.", file=sys.stderr)


            if original_comments_from_api and EFFECTIVE_FIREWORKS_API_KEY:
                print("\n--- Analyzing Comments for Targeted Sentiment (Agent 3) ---", file=sys.stderr)
                analyzed_comments_for_this_video = []
                
                # `original_comments_from_api` already respects `comments_to_analyze_val_str` via `Workspace_youtube_video_details_and_comments_api`
                comments_to_process_for_video_llm = original_comments_from_api
                
                if comments_to_process_for_video_llm:
                    print(f"Analyzing {len(comments_to_process_for_video_llm)} comments for this video with LLM...", file=sys.stderr)
                    for i, comment_obj in enumerate(comments_to_process_for_video_llm):
                        if not comment_obj or not isinstance(comment_obj.get('comment_text'), str) or not comment_obj.get('comment_text').strip():
                            print(f"  Skipping invalid or empty comment object at index {i}.", file=sys.stderr)
                            analysis_error_obj = {"comment_text": str(comment_obj.get('comment_text', 'INVALID_COMMENT_DATA')), "is_sarcastic": False, "sarcasm_reasoning": "N/A - Invalid/Empty Comment Data",
                                                "primary_target_entity": "General Comment/No Specific Listed Entity", 
                                                "entity_identification_reasoning": "N/A - Invalid Comment Data",
                                                "sentiment_expressed": "neutral", "sentiment_score": 0.0,
                                                "overall_reasoning": "Invalid or empty comment data provided to analysis."}
                            analyzed_comments_for_this_video.append(analysis_error_obj)
                            continue

                        print(f"  Analyzing comment {i+1}/{len(comments_to_process_for_video_llm)}: \"{comment_obj.get('comment_text', '')[:60].replace(os.linesep, ' ')}...\"", file=sys.stderr)
                        analysis_result = analyze_comment_sentiment_analysis_langchain(
                            comment_obj, contextual_summary, video_title_for_processing, video_description_for_processing, EFFECTIVE_FIREWORKS_API_KEY
                        )
                        analyzed_comments_for_this_video.append(analysis_result)
                        time.sleep(1) # Rate limiting for LLM calls

                    if analyzed_comments_for_this_video:
                        # Save as targeted_sentiment_analysis.json as expected by GUI
                        save_data_to_file(analyzed_comments_for_this_video, output_video_data_folder, "targeted_sentiment_analysis.json")
                        print(f"  Finished LLM analysis for {len(analyzed_comments_for_this_video)} comments for this video.", file=sys.stderr)
                        grand_total_analyzed_comments_data.extend(analyzed_comments_for_this_video)
                else:
                     print(f"No comments were fetched or available to analyze with LLM for this video.", file=sys.stderr)
            elif not EFFECTIVE_FIREWORKS_API_KEY:
                print("Skipping LLM comment analysis as FIREWORKS_API_KEY is not set.", file=sys.stderr)
            elif not original_comments_from_api:
                 print("Skipping LLM comment analysis as no comments were fetched from API.", file=sys.stderr)

            print(f"\n=== Finished processing video: {video_title_for_processing} ===", file=sys.stderr)
            if video_idx < len(videos_to_process_fully) - 1:
                print("--- Waiting briefly before next video ---", file=sys.stderr)
                time.sleep(3) # Brief pause

        # After all videos are processed, generate and save overall summaries
        if grand_total_analyzed_comments_data:
            print("\n\n========================================================", file=sys.stderr)
            print("       OVERALL COMBINED TARGETED SENTIMENT ANALYSIS RESULTS", file=sys.stderr)
            print("========================================================", file=sys.stderr)
            
            targeted_sentiment_summary_dict = calculate_targeted_sentiment_summary(
                grand_total_analyzed_comments_data, 
                title="Grand Total (All Selected Videos) Targeted Sentiment"
            )
            save_data_to_file(targeted_sentiment_summary_dict, args.output_dir, "overall_sentiment_summary.json")
            
            visual_summary_content = generate_targeted_sentiment_visual_str(targeted_sentiment_summary_dict)
            save_data_to_file(visual_summary_content, args.output_dir, "overall_sentiment_visual.txt")
        else:
            print("\nNo comments were analyzed across all selected videos, so no overall targeted sentiment summary to display or save.", file=sys.stderr)
            # Create empty summary files as GUI might expect them
            save_data_to_file({"by_entity":{}, "overall_stats":{"total_analyzed_items":0, "valid_analyses":0, "total_sarcastic_comments_overall":0}}, args.output_dir, "overall_sentiment_summary.json")
            save_data_to_file("No data to visualize.", args.output_dir, "overall_sentiment_visual.txt")


        print(f"\nAll selected videos processed. Results saved in {os.path.abspath(args.output_dir)}", file=sys.stderr)

if __name__ == '__main__':
    main()
