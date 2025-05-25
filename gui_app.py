import customtkinter
from customtkinter import CTkEntry, CTkLabel, CTkButton, CTkScrollableFrame, CTkCheckBox, CTkTextbox, CTkTabview
import subprocess
import threading
import queue
import re
import json
import os
import webbrowser
from PIL import Image, ImageTk
import customtkinter # Ensure this is already there

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.title("Political Sentiment Analyzer (Youtube Channel)")
        self.geometry("800x900")

        self.log_queue = queue.Queue()
        self.fetched_videos_data = []
        self.video_checkboxes = []
        self.analysis_output_dir = "analysis_results_gui"
        self.selected_videos_for_analysis_cache = []

        # Main frame for inputs
        input_frame = customtkinter.CTkFrame(self)
        input_frame.pack(pady=10, padx=20, fill="x")

        # API Key and Channel ID inputs
        youtube_api_key_label = CTkLabel(input_frame, text="YouTube Data API Key:")
        youtube_api_key_label.pack(pady=(5, 2), anchor="w")
        self.youtube_api_key_entry = CTkEntry(input_frame, placeholder_text="Enter YouTube API Key")
        self.youtube_api_key_entry.pack(pady=(0, 5), fill="x")
        
        fireworks_api_key_label = CTkLabel(input_frame, text="Fireworks API Key:")
        fireworks_api_key_label.pack(pady=(5, 2), anchor="w")
        self.fireworks_api_key_entry = CTkEntry(input_frame, placeholder_text="Enter Fireworks API Key")
        self.fireworks_api_key_entry.pack(pady=(0, 5), fill="x")
        
        channel_id_label = CTkLabel(input_frame, text="YouTube Channel ID:")
        channel_id_label.pack(pady=(5, 2), anchor="w")
        self.channel_id_entry = CTkEntry(input_frame, placeholder_text="Enter YouTube Channel ID")
        self.channel_id_entry.pack(pady=(0, 5), fill="x")

        api_key_guidance_label = CTkLabel(input_frame,
                                          text="Leave API key fields blank to use keys from .env file.",
                                          font=customtkinter.CTkFont(size=10),
                                          text_color="gray")
        api_key_guidance_label.pack(pady=(0, 10), anchor="w")

        self.settings_frame = customtkinter.CTkFrame(self)
        self.settings_frame.pack(pady=5, padx=20, fill="x")

        max_videos_label = CTkLabel(self.settings_frame, text="Max Videos to Scan:")
        max_videos_label.pack(side="left", padx=(0,5), pady=5)
        self.max_videos_scan_entry = CTkEntry(self.settings_frame, width=60)
        self.max_videos_scan_entry.insert(0, "20")
        self.max_videos_scan_entry.pack(side="left", padx=(0,10), pady=5)

        comments_label = CTkLabel(self.settings_frame, text="Comments to Analyze:")
        comments_label.pack(side="left", padx=(10,5), pady=5)
        self.comments_to_analyze_entry = CTkEntry(self.settings_frame, width=60)
        self.comments_to_analyze_entry.insert(0, "100")
        self.comments_to_analyze_entry.pack(side="left", padx=(0,10), pady=5)

        self.fetch_button = CTkButton(self, text="Fetch Videos", command=self.start_fetch_videos_thread)
        self.fetch_button.pack(pady=5, padx=20)
        
        video_list_label = CTkLabel(self, text="Filtered Political Videos:")
        video_list_label.pack(pady=(5,0), padx=20, anchor="w")
        self.video_list_frame = CTkScrollableFrame(self, height=150)
        self.video_list_frame.pack(pady=5, padx=20, fill="x")

        self.analyze_button = CTkButton(self, text="Analyze Selected Videos", command=self.start_analyze_videos_thread)
        self.analyze_button.pack(pady=10, padx=20)

        self.tab_view = CTkTabview(self, height=280)
        self.tab_view.pack(pady=10, padx=20, fill="both", expand=True)
        self.tab_view.add("Log")
        self.tab_view.add("Overall Summary")
        self.tab_view.add("Individual Video Results") # Already there

        # Setup content for Log tab
        self.log_display = CTkTextbox(self.tab_view.tab("Log"), state='disabled', wrap='word')
        self.log_display.pack(pady=5, padx=5, fill='both', expand=True)

        # New structure for "Overall Summary" tab
        self.overall_summary_tab_frame = customtkinter.CTkScrollableFrame(
                    self.tab_view.tab("Overall Summary"), 
                    fg_color="transparent"
                    # You can also customize scrollbar colors here if needed, e.g.:
                    # scrollbar_button_color="#C0C0C0", 
                    # scrollbar_button_hover_color="#A0A0A0"
                )        
        self.overall_summary_tab_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.overall_summary_tab_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # The plot_image_label and overall_summary_display will now be children of this scrollable frame
        self.plot_image_label = customtkinter.CTkLabel(self.overall_summary_tab_frame, text="Sentiment plot will appear here after analysis.")
        # Pack the image label. It will determine its own size based on the image.
        # Horizontal fill/expand is usually not needed for the label itself if the image is pre-scaled.
        self.plot_image_label.pack(pady=(5, 10), padx=5) 

        self.overall_summary_display = CTkTextbox(self.overall_summary_tab_frame, state='disabled', wrap='word', height=200) # Keep a reasonable default height
        # The textbox will fill available width and expand vertically if there's space *after* the image,
        # or contribute to the scrollable height.
        self.overall_summary_display.pack(pady=(0, 5), padx=5, fill="both", expand=True)
    
        self.individual_results_frame = CTkScrollableFrame(self.tab_view.tab("Individual Video Results"))
        self.individual_results_frame.pack(pady=5, padx=5, fill='both', expand=True)
        
        self.process_log_queue()

    def _sanitize_folder_name(self, name_part, max_len=60):
        sanitized = re.sub(r'[^\w\s-]', '', name_part)
        sanitized = sanitized.strip().replace(' ', '_')
        return sanitized[:max_len]

    def _update_log_display(self, message):
        self.log_display.configure(state='normal')
        self.log_display.insert(customtkinter.END, message)
        self.log_display.configure(state='disabled')
        self.log_display.see(customtkinter.END)

    def process_log_queue(self):
        try:
            while True:
                message = self.log_queue.get_nowait()
                self._update_log_display(message)
        except queue.Empty:
            pass
        self.after(100, self.process_log_queue)

    def start_fetch_videos_thread(self):
        self.log_queue.put("Starting to fetch videos...\n")
        self.fetch_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        channel_id = self.channel_id_entry.get()
        youtube_api_key = self.youtube_api_key_entry.get()
        fireworks_api_key = self.fireworks_api_key_entry.get()

        if not channel_id:
            self.log_queue.put("Error: Channel ID is required.\n")
            self.fetch_button.configure(state="normal")
            self.analyze_button.configure(state="normal")
            return
        
        for widget in self.video_list_frame.winfo_children(): widget.destroy()
        self.video_checkboxes = []
        self.fetched_videos_data = []
        
        thread = threading.Thread(target=self._execute_fetch_videos, args=(channel_id, youtube_api_key, fireworks_api_key))
        thread.daemon = True
        thread.start()

    def _execute_fetch_videos(self, channel_id, yt_api_key, fw_api_key):
        # Ensure python3.9 is in PATH or provide full path if necessary
        command = ["python3.9", "-u", "scraper_v2.py", "--action", "fetch_videos", "--channel-id", channel_id]
        
        if yt_api_key:
            command.extend(["--youtube-api-key", yt_api_key])
        
        max_scan = self.max_videos_scan_entry.get()
        try:
            max_scan_val = int(max_scan)
            if max_scan_val > 0:
                command.extend(["--max-videos-scan", str(max_scan_val)])
            else:
                self.log_queue.put("Warning: Invalid 'Max Videos to Scan'. Using script's default (if any).\n")
        except ValueError:
            self.log_queue.put("Warning: 'Max Videos to Scan' not a valid number. Using script's default (if any).\n")

        if fw_api_key:
            command.extend(["--fireworks-api-key", fw_api_key])
        
        self.log_queue.put(f"Executing: {' '.join(command)}\n")
        try:
            process = subprocess.Popen(command, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, 
                                     text=True, 
                                     bufsize=1, # Line buffered
                                     encoding='utf-8', 
                                     errors='replace')
            
            # Thread to read stderr and put lines into the queue for live logging
            def stream_pipe_to_log_queue(pipe):
                if pipe:
                    for line in iter(pipe.readline, ''):
                        self.log_queue.put(line) # `line` from readline includes newline

            stderr_thread = threading.Thread(target=stream_pipe_to_log_queue, args=(process.stderr,))
            stderr_thread.daemon = True
            stderr_thread.start()

            # Read all stdout after process finishes.
            # stdout is expected to be the JSON data from scraper_v2.py's fetch_videos action.
            stdout_data = ""
            if process.stdout:
                stdout_data = process.stdout.read()
            
            # Wait for stderr thread to finish processing any remaining output & for the main process
            stderr_thread.join(timeout=2) # Timeout to prevent indefinite blocking
            process.wait() 

            if process.returncode == 0:
                self.log_queue.put("Scraper executed successfully. Parsing video data...\n")
                try:
                    self.fetched_videos_data = json.loads(stdout_data)
                    if not isinstance(self.fetched_videos_data, list):
                        self.log_queue.put(f"Warning: Scraper output was not a list. Received: {type(self.fetched_videos_data)}\nContent: {str(self.fetched_videos_data)[:200]}\n")
                        self.fetched_videos_data = []
                except json.JSONDecodeError as e:
                    self.log_queue.put(f"JSON Error decoding scraper output: {e}\n")
                    self.log_queue.put(f"Stdout from scraper was: {stdout_data[:1000]}\n")
                    self.fetched_videos_data = []
            else:
                self.log_queue.put(f"Scraper failed with return code {process.returncode}.\n")
                # If scraper failed, stdout_data might contain partial/error output not in JSON format.
                # scraper_v2.py should primarily use stderr for logs/errors.
                if stdout_data.strip() and process.returncode != 0: # Log if stdout had unexpected content on failure
                     self.log_queue.put(f"Unexpected STDOUT from failed scraper: {stdout_data[:1000]}\n")
                self.fetched_videos_data = []

        except FileNotFoundError:
            self.log_queue.put(f"Error: The command {' '.join(command)} was not found. Ensure python3.9 and scraper_v2.py are accessible.\n")
            self.fetched_videos_data = []
        except Exception as e:
            self.log_queue.put(f"Error running scraper: {str(e)}\n")
            self.fetched_videos_data = []
        finally: 
            self.log_queue.put("Fetch process finished.\n")
            # Schedule GUI updates on the main thread
            self.after(0, self._populate_gui_video_list) 
            self.fetch_button.configure(state="normal")
            self.analyze_button.configure(state="normal")


    def _populate_gui_video_list(self):
        for widget in self.video_list_frame.winfo_children(): widget.destroy()
        self.video_checkboxes = []
        if not self.fetched_videos_data: # Checks if list is empty or None
            CTkLabel(self.video_list_frame, text="No videos fetched or an error occurred. Check logs.").pack(pady=10)
            return
        
        self.log_queue.put(f"Populating GUI with {len(self.fetched_videos_data)} videos.\n")
        self.video_list_frame.update_idletasks() 
        frame_width = self.video_list_frame.winfo_width()
        # Ensure label_wraplength is positive, adjust subtraction based on checkbox and padding
        label_wraplength = max(1, frame_width - 80) 

        for video_data in self.fetched_videos_data:
            title = video_data.get('title', 'N/A')
            video_id = video_data.get('video_id', 'N/A')
            
            if title == 'N/A' or video_id == 'N/A':
                self.log_queue.put(f"Skipping video with missing title or ID: {video_data}\n")
                continue # Skip if essential data is missing

            item_frame = customtkinter.CTkFrame(self.video_list_frame)
            checkbox = customtkinter.CTkCheckBox(item_frame, text="")
            checkbox.pack(side="left", padx=(2,5))
            
            self.video_checkboxes.append({
                "checkbox": checkbox, 
                "video_id": video_id, 
                "video_title": title, 
                "url": video_data.get('url', f'https://www.youtube.com/watch?v={video_id}'), # Fallback URL
                "description": video_data.get('description', ''), 
                "reasoning_for_political": video_data.get('reasoning_for_political', '')
            })
            
            label_text = f"{title} (ID: {video_id})"
            label = CTkLabel(item_frame, text=label_text, anchor="w", wraplength=label_wraplength)
            label.pack(side="left", fill="x", expand=True, padx=(0,2))
            item_frame.pack(fill="x", pady=2, padx=2)
            
        if not self.video_checkboxes: # If all videos were skipped or list was initially empty
            CTkLabel(self.video_list_frame, text="No valid videos to display after filtering.").pack(pady=10)
        
    def start_analyze_videos_thread(self):
        self.selected_videos_for_analysis_cache = []
        for item in self.video_checkboxes:
            if item["checkbox"].get() == 1:
                self.selected_videos_for_analysis_cache.append({
                    "video_id": item["video_id"], "title": item["video_title"],
                    "url": item.get("url", f"https://www.youtube.com/watch?v={item['video_id']}") # Consistent URL
                })
        if not self.selected_videos_for_analysis_cache:
            self.log_queue.put("Error: No videos selected for analysis.\n")
            return
        
        self.log_queue.put(f"Starting analysis for {len(self.selected_videos_for_analysis_cache)} video(s)...\n")
        self.fetch_button.configure(state="disabled")
        self.analyze_button.configure(state="disabled")
        
        temp_input_file_path = "temp_selected_videos.json"
        try:
            with open(temp_input_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.selected_videos_for_analysis_cache, f, indent=4)
        except Exception as e:
            self.log_queue.put(f"Error creating temp input file: {str(e)}\n")
            self.fetch_button.configure(state="normal")
            self.analyze_button.configure(state="normal")
            return
            
        thread = threading.Thread(target=self._execute_analyze_videos, args=(temp_input_file_path,))
        thread.daemon = True
        thread.start()

    def _execute_analyze_videos(self, temp_input_file_path):
        try:
            os.makedirs(self.analysis_output_dir, exist_ok=True)
        except Exception as e:
            self.log_queue.put(f"Error creating output dir: {str(e)}\n")
            self.after(0, self._finalize_analysis_ui)
            return
        
        # Ensure python is in PATH or provide full path if necessary
        command = ["python3.9", "-u", "scraper_v2.py", "--action", "analyze_videos", 
                   "--input-file", temp_input_file_path, 
                   "--output-dir", self.analysis_output_dir]
        
        comments_count_str = self.comments_to_analyze_entry.get()
        # scraper_v2.py expects "all" or a number string for --comments-to-analyze
        command.extend(["--comments-to-analyze", comments_count_str])


        youtube_key = self.youtube_api_key_entry.get()
        fireworks_key = self.fireworks_api_key_entry.get()
        if youtube_key:
            command.extend(["--youtube-api-key", youtube_key])
        if fireworks_key:
            command.extend(["--fireworks-api-key", fireworks_key])
        
        self.log_queue.put(f"Executing analysis: {' '.join(command)}\n")
        try:
            process = subprocess.Popen(command, 
                                     stdout=subprocess.PIPE, # Though analyze_videos primarily uses stderr for logs
                                     stderr=subprocess.PIPE, 
                                     text=True, 
                                     bufsize=1, 
                                     encoding='utf-8', 
                                     errors='replace')

            # Live logging for stderr from the analysis script
            def stream_pipe_to_log_queue(pipe):
                if pipe:
                    for line in iter(pipe.readline, ''):
                        self.log_queue.put(line) 
            
            stderr_thread = threading.Thread(target=stream_pipe_to_log_queue, args=(process.stderr,))
            stderr_thread.daemon = True
            stderr_thread.start()

            # Collect any stdout (though not primary output for analyze_videos)
            stdout_data_analysis = ""
            if process.stdout:
                stdout_data_analysis = process.stdout.read()

            stderr_thread.join(timeout=2)
            process.wait()

            if stdout_data_analysis.strip(): # If analyze_videos unexpectedly prints to stdout
                 self.log_queue.put(f"Analysis STDOUT (unexpected):\n{stdout_data_analysis}\n")
            
            # Display results regardless of return code, as files might still be partially created
            self.after(0, self._display_analysis_results)

        except FileNotFoundError:
            self.log_queue.put(f"Error: The command {' '.join(command)} was not found.\n")
        except Exception as e:
            self.log_queue.put(f"Error running analysis: {str(e)}\n")
        finally:
            self.after(0, self._finalize_analysis_ui)
            if os.path.exists(temp_input_file_path):
                try:
                    os.remove(temp_input_file_path)
                except Exception as e:
                    self.log_queue.put(f"Warning: Could not delete temp file {temp_input_file_path}: {e}\n")

    def _finalize_analysis_ui(self):
        self.fetch_button.configure(state="normal")
        self.analyze_button.configure(state="normal")
        self.log_queue.put("Analysis process finished.\n")

    def _display_analysis_results(self):
        self.log_queue.put("Populating result tabs...\n")
        self.tab_view.set("Overall Summary") # Switch to the Overall Summary tab

        # --- 1. Populate the JSON Summary Textbox ---
        self.overall_summary_display.configure(state='normal')
        self.overall_summary_display.delete("1.0", customtkinter.END)
        
        summary_json_path = os.path.join(self.analysis_output_dir, "overall_sentiment_summary.json")

        if os.path.exists(summary_json_path):
            try:
                with open(summary_json_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                summary_text = "--- Overall Sentiment JSON Summary ---\n\n"
                if 'overall_stats' in json_data:
                    summary_text += f"Total Analyzed Comments: {json_data['overall_stats'].get('valid_analyses', 0)}\n"
                    summary_text += f"Total Sarcastic Comments: {json_data['overall_stats'].get('total_sarcastic_comments_overall', 0)}\n\n"
                
                if 'by_entity' in json_data:
                    entities_found = False
                    for entity, data in json_data['by_entity'].items():
                        if data.get("comment_count", 0) > 0: # Only display if there are comments for the entity
                            entities_found = True
                            summary_text += f"Entity: {entity} ({data['comment_count']} comments)\n"
                            summary_text += f"  Positive: {data.get('positive', 0)}, Negative: {data.get('negative', 0)}, Neutral: {data.get('neutral', 0)}\n"
                            summary_text += f"  Sarcastic: {data.get('sarcastic_count', 0)}\n"
                            avg_score = data.get('average_sentiment_score', 0.0)
                            summary_text += f"  Avg Score: {avg_score:.2f} (from {data.get('valid_score_count',0)} scored comments)\n\n"
                    if not entities_found:
                        summary_text += "No specific entities had comments attributed to them in this analysis.\n\n"
                else:
                    summary_text += "No 'by_entity' data found in summary.\n\n"
                    
                self.overall_summary_display.insert(customtkinter.END, summary_text)
            except Exception as e:
                error_message = f"Error reading/parsing JSON summary: {str(e)}\n"
                self.log_queue.put(error_message)
                self.overall_summary_display.insert(customtkinter.END, error_message)
        else:
            not_found_message = f"JSON summary file not found: {summary_json_path}\n"
            self.log_queue.put(not_found_message)
            self.overall_summary_display.insert(customtkinter.END, not_found_message)
        self.overall_summary_display.configure(state='disabled')

        # --- 2. Display the Sentiment Plot Image ---
        plot_image_path = os.path.join(self.analysis_output_dir, "overall_sentiment_plot.png")

        if os.path.exists(plot_image_path):
            try:
                original_image = Image.open(plot_image_path)
                img_width, img_height = original_image.size

                # Determine a sensible display width for the image.
                # Try to make it fit the width of the scrollable frame's content area.
                # This can be a bit tricky as winfo_width() might not be perfectly accurate
                # before the GUI is fully idle. A practical approach is often needed.
                
                # Ensure the frame has had a chance to determine its size.
                self.overall_summary_tab_frame.update_idletasks() 
                available_width = self.overall_summary_tab_frame.winfo_width()
                
                # Subtract some padding and approximate scrollbar width
                max_display_width = available_width - 30 if available_width > 50 else 700 # Fallback if width is too small initially

                display_width = img_width
                display_height = img_height

                if img_width > max_display_width:
                    # Scale image to fit max_display_width while maintaining aspect ratio
                    ratio = max_display_width / float(img_width)
                    display_width = max_display_width
                    display_height = int(img_height * ratio)
                # else: image is already narrower than our max, use its original dimensions (or scaled if height was also capped previously)

                # Ensure dimensions are positive
                display_width = max(1, display_width)
                display_height = max(1, display_height)

                # Resize the PIL image before creating CTkImage
                # Using Resampling.LANCZOS for better quality downscaling
                resized_image = original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                ctk_img = customtkinter.CTkImage(light_image=resized_image,
                                                 dark_image=resized_image, 
                                                 size=(display_width, display_height))
                
                self.plot_image_label.configure(image=ctk_img, text="") # Set image, clear placeholder
                self.plot_image_label.image = ctk_img # Keep a reference (good practice)
                self.log_queue.put("Sentiment plot displayed.\n")

            except Exception as e:
                error_message_plot = f"Error displaying plot image: {str(e)}\n"
                self.log_queue.put(error_message_plot)
                self.plot_image_label.configure(image=None, text="Error displaying plot.")
        else:
            not_found_message_plot = f"Sentiment plot image not found: {plot_image_path}\n"
            self.log_queue.put(not_found_message_plot)
            self.plot_image_label.configure(image=None, text="Plot image not found.")
            

        # --- 3. Populate Individual Video Results Tab ---
        for widget in self.individual_results_frame.winfo_children(): # Clear previous results
            widget.destroy()

        if not self.selected_videos_for_analysis_cache: # Videos selected for *this specific analysis run*
             CTkLabel(self.individual_results_frame, text="No videos were selected for this analysis run to show individual results.").pack(pady=10)
             return

        any_results_displayed = False
        for video_data in self.selected_videos_for_analysis_cache:
            video_title = video_data.get('title', 'N/A')
            video_id = video_data.get('video_id', 'N/A')

            if video_id == 'N/A':
                continue

            sanitized_title_for_folder = self._sanitize_folder_name(video_title if video_title != 'N/A' else video_id)
            video_subfolder = os.path.join(self.analysis_output_dir, f"{video_id}_{sanitized_title_for_folder}")
            
            video_frame = customtkinter.CTkFrame(self.individual_results_frame, border_width=1)
            video_frame.pack(fill="x", pady=5, padx=5)
            
            title_label_text = f"Video: {video_title} (ID: {video_id})"
            title_label = CTkLabel(video_frame, text=title_label_text, font=customtkinter.CTkFont(weight="bold"))
            title_label.pack(anchor="w", padx=5, pady=(5,2))

            open_button = CTkButton(video_frame, text="Open Video Folder", 
                                    command=lambda folder=video_subfolder: self.open_folder_action(folder))
            open_button.pack(anchor="e", padx=5, pady=2) # Changed to "e" (east/right)
            
            files_to_check = [
                "video_meta.json", 
                "original_comments.json",
                "audio_track.mp3",
                "transcription.txt", 
                "contextual_summary.txt", 
                "targeted_sentiment_analysis.json"
            ]
            for file_name in files_to_check:
                file_path = os.path.join(video_subfolder, file_name)
                status = "Found" if os.path.exists(file_path) else "Not Found"
                CTkLabel(video_frame, text=f"  - {file_name}: {status}").pack(anchor="w", padx=10)
            any_results_displayed = True
        
        if not any_results_displayed: # If loop was empty or all videos skipped
            CTkLabel(self.individual_results_frame, text="No individual results to display for the processed videos.").pack(pady=10)

    def open_folder_action(self, folder_path): # Helper method for the button
        try:
            if os.path.exists(folder_path):
                webbrowser.open(os.path.realpath(folder_path))
            else:
                self.log_queue.put(f"Error: Folder not found: {folder_path}\n")
        except Exception as e:
            self.log_queue.put(f"Error opening folder {folder_path}: {e}\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
