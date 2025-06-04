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
import tkinter # Added for tkinter.Menu
from tkinter import filedialog # Added for askaveasfilename

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
        self.current_resized_pil_plot_image = None # To store the PIL image object for the plot

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
        self.tab_view.add("Individual Video Results") 

        self.log_display = CTkTextbox(self.tab_view.tab("Log"), state='disabled', wrap='word')
        self.log_display.pack(pady=5, padx=5, fill='both', expand=True)

        self.overall_summary_tab_frame = customtkinter.CTkScrollableFrame(
                    self.tab_view.tab("Overall Summary"), 
                    fg_color="transparent"
                )        
        self.overall_summary_tab_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.plot_image_label = customtkinter.CTkLabel(self.overall_summary_tab_frame, text="Sentiment plot will appear here after analysis.")
        self.plot_image_label.pack(pady=(5, 10), padx=5) 
        # Bind right-click event to the plot_image_label
        self.plot_image_label.bind("<Button-3>", self._show_plot_context_menu)


        self.overall_summary_display = CTkTextbox(self.overall_summary_tab_frame, state='disabled', wrap='word', height=200) 
        self.overall_summary_display.pack(pady=(0, 5), padx=5, fill="both", expand=True)
    
        self.individual_results_frame = CTkScrollableFrame(self.tab_view.tab("Individual Video Results"))
        self.individual_results_frame.pack(pady=5, padx=5, fill='both', expand=True)
        
        self.process_log_queue()

    # ... (rest of your existing methods: _sanitize_folder_name, _update_log_display, etc.)
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
            # self.analyze_button.configure(state="normal") # It should remain disabled if fetch fails early
            return
        
        for widget in self.video_list_frame.winfo_children(): widget.destroy()
        self.video_checkboxes = []
        self.fetched_videos_data = []
        
        thread = threading.Thread(target=self._execute_fetch_videos, args=(channel_id, youtube_api_key, fireworks_api_key))
        thread.daemon = True
        thread.start()

    def _execute_fetch_videos(self, channel_id, yt_api_key, fw_api_key):
        # Ensure python3.9 is in PATH or provide full path if necessary
        command = ["python", "-u", "scraper_v2.py", "--action", "fetch_videos", "--channel-id", channel_id]
        
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
                                     bufsize=1, 
                                     encoding='utf-8', 
                                     errors='replace')
            
            def stream_pipe_to_log_queue(pipe):
                if pipe:
                    for line in iter(pipe.readline, ''):
                        self.log_queue.put(line) 

            stderr_thread = threading.Thread(target=stream_pipe_to_log_queue, args=(process.stderr,))
            stderr_thread.daemon = True
            stderr_thread.start()

            stdout_data = ""
            if process.stdout:
                stdout_data = process.stdout.read()
            
            stderr_thread.join(timeout=2) 
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
                if stdout_data.strip() and process.returncode != 0: 
                     self.log_queue.put(f"Unexpected STDOUT from failed scraper: {stdout_data[:1000]}\n")
                self.fetched_videos_data = []

        except FileNotFoundError:
            self.log_queue.put(f"Error: The command {' '.join(command)} was not found. Ensure python and scraper_v2.py are accessible.\n")
            self.fetched_videos_data = []
        except Exception as e:
            self.log_queue.put(f"Error running scraper: {str(e)}\n")
            self.fetched_videos_data = []
        finally: 
            self.log_queue.put("Fetch process finished.\n")
            self.after(0, self._populate_gui_video_list) 
            self.fetch_button.configure(state="normal")
            self.analyze_button.configure(state="normal") # Enable analyze button once fetch is done


    def _populate_gui_video_list(self):
        for widget in self.video_list_frame.winfo_children(): widget.destroy()
        self.video_checkboxes = []
        if not self.fetched_videos_data:
            CTkLabel(self.video_list_frame, text="No videos fetched or an error occurred. Check logs.").pack(pady=10)
            return
        
        self.log_queue.put(f"Populating GUI with {len(self.fetched_videos_data)} videos.\n")
        self.video_list_frame.update_idletasks() 
        frame_width = self.video_list_frame.winfo_width()
        label_wraplength = max(1, frame_width - 80) 

        for video_data in self.fetched_videos_data:
            title = video_data.get('title', 'N/A')
            video_id = video_data.get('video_id', 'N/A')
            
            if title == 'N/A' or video_id == 'N/A':
                self.log_queue.put(f"Skipping video with missing title or ID: {video_data}\n")
                continue

            item_frame = customtkinter.CTkFrame(self.video_list_frame)
            checkbox = customtkinter.CTkCheckBox(item_frame, text="")
            checkbox.pack(side="left", padx=(2,5))
            
            self.video_checkboxes.append({
                "checkbox": checkbox, 
                "video_id": video_id, 
                "video_title": title, 
                "url": video_data.get('url', f'https://www.youtube.com/watch?v={video_id}'), 
                "description": video_data.get('description', ''), 
                "reasoning_for_political": video_data.get('reasoning_for_political', '')
            })
            
            label_text = f"{title} (ID: {video_id})"
            label = CTkLabel(item_frame, text=label_text, anchor="w", wraplength=label_wraplength)
            label.pack(side="left", fill="x", expand=True, padx=(0,2))
            item_frame.pack(fill="x", pady=2, padx=2)
            
        if not self.video_checkboxes: 
            CTkLabel(self.video_list_frame, text="No valid videos to display after filtering.").pack(pady=10)
        
    def start_analyze_videos_thread(self):
        self.selected_videos_for_analysis_cache = []
        for item in self.video_checkboxes:
            if item["checkbox"].get() == 1:
                self.selected_videos_for_analysis_cache.append({
                    "video_id": item["video_id"], "title": item["video_title"],
                    "url": item.get("url", f"https://www.youtube.com/watch?v={item['video_id']}")
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
        
        command = ["python", "-u", "scraper_v2.py", "--action", "analyze_videos", 
                   "--input-file", temp_input_file_path, 
                   "--output-dir", self.analysis_output_dir]
        
        comments_count_str = self.comments_to_analyze_entry.get()
        command.extend(["--comments-to-analyze", comments_count_str])

        youtube_key = self.youtube_api_key_entry.get()
        fireworks_key = self.fireworks_api_key_entry.get()
        if youtube_key: command.extend(["--youtube-api-key", youtube_key])
        if fireworks_key: command.extend(["--fireworks-api-key", fireworks_key])
        
        self.log_queue.put(f"Executing analysis: {' '.join(command)}\n")
        try:
            process = subprocess.Popen(command, 
                                     stdout=subprocess.PIPE, 
                                     stderr=subprocess.PIPE, 
                                     text=True, bufsize=1, 
                                     encoding='utf-8', errors='replace')

            def stream_pipe_to_log_queue(pipe):
                if pipe:
                    for line in iter(pipe.readline, ''): self.log_queue.put(line) 
            
            stderr_thread = threading.Thread(target=stream_pipe_to_log_queue, args=(process.stderr,))
            stderr_thread.daemon = True
            stderr_thread.start()

            stdout_data_analysis = ""
            if process.stdout: stdout_data_analysis = process.stdout.read()

            stderr_thread.join(timeout=2)
            process.wait()

            if stdout_data_analysis.strip():
                 self.log_queue.put(f"Analysis STDOUT (unexpected):\n{stdout_data_analysis}\n")
            
            self.after(0, self._display_analysis_results)

        except FileNotFoundError:
            self.log_queue.put(f"Error: The command {' '.join(command)} was not found.\n")
        except Exception as e:
            self.log_queue.put(f"Error running analysis: {str(e)}\n")
        finally:
            self.after(0, self._finalize_analysis_ui)
            if os.path.exists(temp_input_file_path):
                try: os.remove(temp_input_file_path)
                except Exception as e: self.log_queue.put(f"Warning: Could not delete temp file {temp_input_file_path}: {e}\n")

    def _finalize_analysis_ui(self):
        self.fetch_button.configure(state="normal")
        self.analyze_button.configure(state="normal")
        self.log_queue.put("Analysis process finished.\n")

    def _display_analysis_results(self):
        self.log_queue.put("Populating result tabs...\n")
        self.tab_view.set("Overall Summary")

        self.overall_summary_display.configure(state='normal')
        self.overall_summary_display.delete("1.0", customtkinter.END)
        summary_json_path = os.path.join(self.analysis_output_dir, "overall_sentiment_summary.json")

        if os.path.exists(summary_json_path):
            try:
                with open(summary_json_path, 'r', encoding='utf-8') as f: json_data = json.load(f)
                summary_text = "--- Overall Sentiment JSON Summary ---\n\n"
                if 'overall_stats' in json_data:
                    summary_text += f"Total Analyzed Comments: {json_data['overall_stats'].get('valid_analyses', 0)}\n"
                    summary_text += f"Total Sarcastic Comments: {json_data['overall_stats'].get('total_sarcastic_comments_overall', 0)}\n\n"
                if 'by_entity' in json_data:
                    entities_found = False
                    for entity, data in json_data['by_entity'].items():
                        if data.get("comment_count", 0) > 0:
                            entities_found = True
                            summary_text += f"Entity: {entity} ({data['comment_count']} comments)\n"
                            summary_text += f"  Positive: {data.get('positive', 0)}, Negative: {data.get('negative', 0)}, Neutral: {data.get('neutral', 0)}\n"
                            summary_text += f"  Sarcastic: {data.get('sarcastic_count', 0)}\n"
                            avg_score = data.get('average_sentiment_score', 0.0)
                            summary_text += f"  Avg Score: {avg_score:.2f} (from {data.get('valid_score_count',0)} scored comments)\n\n"
                    if not entities_found: summary_text += "No specific entities had comments attributed to them in this analysis.\n\n"
                else: summary_text += "No 'by_entity' data found in summary.\n\n"
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

        plot_image_path = os.path.join(self.analysis_output_dir, "overall_sentiment_plot.png")
        self.current_resized_pil_plot_image = None # Reset before loading new image

        if os.path.exists(plot_image_path):
            try:
                original_image = Image.open(plot_image_path)
                img_width, img_height = original_image.size
                self.overall_summary_tab_frame.update_idletasks() 
                available_width = self.overall_summary_tab_frame.winfo_width()
                max_display_width = available_width - 30 if available_width > 50 else 700
                display_width, display_height = img_width, img_height
                if img_width > max_display_width:
                    ratio = max_display_width / float(img_width)
                    display_width = max_display_width
                    display_height = int(img_height * ratio)
                display_width, display_height = max(1, display_width), max(1, display_height)
                
                # Store the resized PIL Image object
                self.current_resized_pil_plot_image = original_image.resize((display_width, display_height), Image.Resampling.LANCZOS)
                
                ctk_img = customtkinter.CTkImage(light_image=self.current_resized_pil_plot_image,
                                                 dark_image=self.current_resized_pil_plot_image, 
                                                 size=(display_width, display_height))
                self.plot_image_label.configure(image=ctk_img, text="")
                self.plot_image_label.image = ctk_img 
                self.log_queue.put("Sentiment plot displayed.\n")
            except Exception as e:
                error_message_plot = f"Error displaying plot image: {str(e)}\n"
                self.log_queue.put(error_message_plot)
                self.plot_image_label.configure(image=None, text="Error displaying plot.")
                self.current_resized_pil_plot_image = None
        else:
            not_found_message_plot = f"Sentiment plot image not found: {plot_image_path}\n"
            self.log_queue.put(not_found_message_plot)
            self.plot_image_label.configure(image=None, text="Plot image not found.")
            self.current_resized_pil_plot_image = None
            
        for widget in self.individual_results_frame.winfo_children(): widget.destroy()
        if not self.selected_videos_for_analysis_cache:
             CTkLabel(self.individual_results_frame, text="No videos were selected for this analysis run to show individual results.").pack(pady=10)
             return
        any_results_displayed = False
        for video_data in self.selected_videos_for_analysis_cache:
            video_title, video_id = video_data.get('title', 'N/A'), video_data.get('video_id', 'N/A')
            if video_id == 'N/A': continue
            sanitized_title_for_folder = self._sanitize_folder_name(video_title if video_title != 'N/A' else video_id)
            video_subfolder = os.path.join(self.analysis_output_dir, f"{video_id}_{sanitized_title_for_folder}")
            video_frame = customtkinter.CTkFrame(self.individual_results_frame, border_width=1)
            video_frame.pack(fill="x", pady=5, padx=5)
            title_label = CTkLabel(video_frame, text=f"Video: {video_title} (ID: {video_id})", font=customtkinter.CTkFont(weight="bold"))
            title_label.pack(anchor="w", padx=5, pady=(5,2))
            open_button = CTkButton(video_frame, text="Open Video Folder", command=lambda folder=video_subfolder: self.open_folder_action(folder))
            open_button.pack(anchor="e", padx=5, pady=2)
            files_to_check = ["video_meta.json", "original_comments.json", "audio_track.mp3", "transcription.txt", "contextual_summary.txt", "targeted_sentiment_analysis.json"]
            for file_name in files_to_check:
                file_path = os.path.join(video_subfolder, file_name)
                CTkLabel(video_frame, text=f"  - {file_name}: {'Found' if os.path.exists(file_path) else 'Not Found'}").pack(anchor="w", padx=10)
            any_results_displayed = True
        if not any_results_displayed:
            CTkLabel(self.individual_results_frame, text="No individual results to display for the processed videos.").pack(pady=10)

    def open_folder_action(self, folder_path):
        try:
            if os.path.exists(folder_path): webbrowser.open(os.path.realpath(folder_path))
            else: self.log_queue.put(f"Error: Folder not found: {folder_path}\n")
        except Exception as e: self.log_queue.put(f"Error opening folder {folder_path}: {e}\n")

    def _show_plot_context_menu(self, event):
        """Shows a context menu for the plot image."""
        if not self.current_resized_pil_plot_image: # Don't show menu if no image
            return
            
        context_menu = tkinter.Menu(self, tearoff=0)
        context_menu.add_command(label="Copy Image to Clipboard (Info)", command=self._copy_plot_image_to_clipboard_info)
        context_menu.add_command(label="Save Image As...", command=self._save_plot_image_as)
        try:
            context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            context_menu.grab_release() # Ensure menu doesn't block other events

    def _save_plot_image_as(self):
        """Saves the current plot image to a file chosen by the user."""
        if self.current_resized_pil_plot_image:
            try:
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
                    title="Save Image As"
                )
                if file_path:
                    self.current_resized_pil_plot_image.save(file_path)
                    self.log_queue.put(f"Image saved to {file_path}\n")
                else:
                    self.log_queue.put("Image save cancelled by user.\n")
            except Exception as e:
                self.log_queue.put(f"Error saving image: {e}\n")
        else:
            self.log_queue.put("No image available to save.\n")

    def _copy_plot_image_to_clipboard_info(self):
        """Provides information about copying images to the clipboard."""
        if self.current_resized_pil_plot_image:
            self.log_queue.put(
                "INFO: Direct image copy to the system clipboard for use in other applications\n"
                "is a complex feature that typically requires platform-specific libraries\n"
                "(e.g., pywin32 on Windows, or scripting 'pbcopy'/'xclip' on macOS/Linux).\n"
                "A reliable, simple cross-platform solution using only standard Python\n"
                "libraries like Tkinter and Pillow for this purpose is not readily available.\n\n"
                "Please use the 'Save Image As...' option from the context menu to save the\n"
                "image to a file, which you can then use as needed.\n"
            )
        else:
            self.log_queue.put("No image available to copy.\n")


if __name__ == "__main__":
    app = App()
    app.mainloop()
