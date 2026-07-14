import os
import uuid
import shutil
import pickle
import threading
import subprocess
import string
import requests
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

# Import clipsai
from clipsai import Transcriber, ClipFinder, resize, MediaEditor, AudioVideoFile
from clipsai.clip.clip import Clip
import nltk

# Initialize FastAPI app
app = FastAPI(title="ClippedAI API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
INPUT_DIR = "input"
OUTPUT_DIR = "output"
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Global status tracking
tasks_status: Dict[str, dict] = {}
status_lock = threading.Lock()

# Setup NLTK
nltk.download('punkt', quiet=True)

def safe_filename(s: str) -> str:
    """Remove characters not allowed in filenames."""
    safe_chars = string.ascii_letters + string.digits + " -_."
    safe_chars += "!?,:;@#$%^&+=[]{}"
    # Common emojis
    emoji_chars = "".join(chr(i) for i in range(0x1F600, 0x1F64F)) + \
                  "".join(chr(i) for i in range(0x1F300, 0x1F5FF)) + \
                  "".join(chr(i) for i in range(0x1F900, 0x1F9FF)) + \
                  "".join(chr(i) for i  in range(0x1FA70, 0x1FAFF))
    valid_chars = safe_chars + emoji_chars
    return ''.join(c for c in s if c in valid_chars)

def ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centisecs:02d}"

def calculate_engagement_score(clip, transcription):
    """Calculate a custom engagement score for a clip."""
    clip_words = [w for w in transcription.get_word_info() 
                  if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    
    if not clip_words:
        return 0.0
    
    duration = clip.end_time - clip.start_time
    word_count = len(clip_words)
    word_density = word_count / duration if duration > 0 else 0
    
    engagement_words = 0
    for word_info in clip_words:
        word = word_info["word"]
        if any(char.isdigit() for char in word) or '$' in word or '!' in word:
            engagement_words += 1
            
    word_density_score = min(word_density / 3.0, 1.0)
    engagement_ratio = engagement_words / word_count if word_count > 0 else 0
    duration_score = min(duration / 75.0, 1.0)
    
    engagement_score = (word_density_score * 0.45 + 
                        engagement_ratio * 0.30 + 
                        duration_score * 0.25)
    
    return engagement_score

def get_viral_title(transcript_text: str, groq_api_key: str, log_callback) -> str:
    """Generate a catchy viral title using Groq API."""
    if not groq_api_key or groq_api_key == "your_groq_api_key_here":
        log_callback("Groq API key not provided or placeholder. Bypassing viral title.")
        return "Untitled Clip"
        
    log_callback("Generating viral title using Groq...")
    examples = [
        "She was almost dead 😵", "He made $1,000,000 in 1 hour 💸", "This changed everything... 😲", 
        "They couldn't believe what happened! 😱", "He risked it all for this 😬"
    ]
    prompt = (
        "Given the following transcript, generate a catchy, viral YouTube Shorts title (max 7 words). "
        "ALWAYS include an emoji in the title. ONLY output the title, nothing else. Do NOT use hashtags. "
        "Do NOT explain, do NOT repeat the prompt, do NOT add quotes. The title should be in the style of these examples: "
        + ", ".join(examples) + ".\n\nTranscript:\n" + transcript_text
    )
    headers = {
        'Authorization': f'Bearer {groq_api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30,
        "temperature": 0.7,
        "top_p": 0.9
    }
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=15
        )
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        lines = [l.strip('"') for l in content.strip().split('\n') if l.strip() and not l.lower().startswith('here') and not l.lower().startswith('title:')]
        return lines[0] if lines else "Untitled Clip"
    except Exception as e:
        log_callback(f"Failed to generate title using Groq: {e}")
        return "Untitled Clip"

def create_animated_subtitles(video_path: str, transcription, clip, output_path: str, log_callback) -> str:
    """Create ASS subtitle file and add to video using FFmpeg."""
    log_callback("Creating styled subtitles...")
    word_info = [w for w in transcription.get_word_info() if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time]
    if not word_info:
        log_callback("No word-level transcript found for this clip. Skipping subtitles.")
        return video_path
        
    cues = []
    current_cue = {'words': [], 'start_time': None, 'end_time': None}
    
    for w in word_info:
        word = w["word"]
        start_time = w["start_time"] - clip.start_time
        end_time = w["end_time"] - clip.start_time
        
        should_start_new = False
        if current_cue['start_time'] is None:
            should_start_new = True
        elif len(' '.join(current_cue['words']) + ' ' + word) > 25:
            should_start_new = True
        elif start_time - current_cue['end_time'] > 0.5:
            should_start_new = True
            
        if should_start_new:
            if current_cue['words']:
                cues.append({
                    'start': current_cue['start_time'],
                    'end': current_cue['end_time'],
                    'text': ' '.join(current_cue['words'])
                })
            current_cue = {
                'words': [word],
                'start_time': start_time,
                'end_time': end_time
            }
        else:
            current_cue['words'].append(word)
            current_cue['end_time'] = end_time
            
    if current_cue['words']:
        cues.append({
            'start': current_cue['start_time'],
            'end': current_cue['end_time'],
            'text': ' '.join(current_cue['words'])
        })
        
    ass_file = os.path.join(OUTPUT_DIR, f'temp_subtitles_{uuid.uuid4().hex[:8]}.ass')
    with open(ass_file, 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat Extra Bold,80,&H00FFFFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1
Style: Yellow,Montserrat Extra Bold,80,&H0000FFFF,&H000000FF,&H40000000,&HFF000000,-1,0,0,0,100,100,2,0,1,15,0,8,30,30,120,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        for cue in cues:
            start = ass_time(cue['start'])
            end = ass_time(cue['end'])
            words = cue['text'].split()
            line = ''
            for w in words:
                if any(char.isdigit() for char in w) or '$' in w or (',' in w and w.replace(',', '').isdigit()):
                    line += f'{{\\style Yellow}}{w} '
                else:
                    line += f'{w} '
            line = line.strip()
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{line}\n")
            
    final_output = output_path.replace('.mp4', '_with_subtitles.mp4')
    # Escape path separators for FFmpeg filter on Windows
    escaped_ass_file = ass_file.replace('\\', '/')
    
    ffmpeg_cmd = [
        'ffmpeg', '-i', video_path,
        '-vf', f"ass='{escaped_ass_file}'",
        '-c:a', 'copy',
        '-y',
        final_output
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        if os.path.exists(ass_file):
            os.remove(ass_file)
        log_callback("Subtitles burned successfully.")
        return final_output
    except Exception as e:
        log_callback(f"Failed to burn subtitles: {e}")
        if os.path.exists(ass_file):
            os.remove(ass_file)
        return video_path

def pipeline_thread(
    task_id: str,
    video_path: str,
    hf_token: str,
    groq_key: str,
    min_dur: int,
    max_dur: int,
    model_size: str,
    max_clips: int
):
    def log(message: str):
        with status_lock:
            tasks_status[task_id]["logs"].append(message)
            print(f"[{task_id}] {message}")

    try:
        log("Pipeline started.")
        log(f"Whisper Model: {model_size} | Min Clip: {min_dur}s | Max Clip: {max_dur}s | Max Clips Count: {max_clips}")
        
        # Determine transcription cache file
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        transcription_path = os.path.join(INPUT_DIR, f"{base_name}_transcription.pkl")
        
        transcriber = Transcriber(model_size=model_size)
        transcription = None
        
        if os.path.exists(transcription_path):
            log("Found cached transcription. Loading...")
            try:
                with open(transcription_path, "rb") as f:
                    transcription = pickle.load(f)
                log("Successfully loaded cached transcription.")
            except Exception as e:
                log(f"Failed to load cached transcription: {e}. Re-transcribing...")

        if transcription is None:
            log("Transcribing audio (this might take a while if downloading the model)...")
            transcription = transcriber.transcribe(audio_file_path=video_path, iso6391_lang_code='en')
            log("Transcription completed. Caching results...")
            with open(transcription_path, "wb") as f:
                pickle.dump(transcription, f)
            log(f"Transcription cached at {transcription_path}")

        # Find clips
        log("Running ClipFinder...")
        clipfinder = ClipFinder()
        clips = clipfinder.find_clips(transcription=transcription)
        
        if not clips:
            log("No clips found in the video.")
            with status_lock:
                tasks_status[task_id]["status"] = "completed"
                tasks_status[task_id]["progress"] = 100
            return

        # Filter and select clips
        log("Filtering and scoring clips...")
        valid_clips = [c for c in clips if min_dur <= (c.end_time - c.start_time) <= max_dur]
        selected_clips = []
        
        if valid_clips:
            clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in valid_clips]
            clip_scores.sort(key=lambda x: x[1], reverse=True)
            for i, (clip, score) in enumerate(clip_scores):
                if i < 2 or score >= 0.6:
                    if len(selected_clips) < max_clips:
                        selected_clips.append((clip, score))
                else:
                    break
        else:
            log(f"No clips found between {min_dur} and {max_dur} seconds. Falling back...")
            # Try short clips
            short_clips = [c for c in clips if c.end_time - c.start_time < min_dur]
            if short_clips:
                log("Extending short clips to minimum duration...")
                short_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in short_clips]
                short_clip_scores.sort(key=lambda x: x[1], reverse=True)
                for i, (clip, score) in enumerate(short_clip_scores[:max_clips]):
                    extended_clip = Clip(
                        start_time=clip.start_time,
                        end_time=clip.start_time + min_dur,
                        start_char=clip.start_char,
                        end_char=clip.end_char
                    )
                    selected_clips.append((extended_clip, score))
            else:
                # Try trimming long clips
                log("Trimming long clips to maximum duration...")
                long_clip_scores = [(clip, calculate_engagement_score(clip, transcription)) for clip in clips]
                long_clip_scores.sort(key=lambda x: x[1], reverse=True)
                for i, (clip, score) in enumerate(long_clip_scores[:max_clips]):
                    trimmed_clip = Clip(
                        start_time=clip.start_time,
                        end_time=clip.start_time + max_dur,
                        start_char=clip.start_char,
                        end_char=clip.end_char
                    )
                    selected_clips.append((trimmed_clip, score))

        log(f"Selected {len(selected_clips)} clips for processing.")
        
        media_editor = MediaEditor()
        media_file = AudioVideoFile(video_path)
        output_clips = []
        
        for idx, (clip, score) in enumerate(selected_clips):
            clip_name = f"Clip {idx + 1}"
            log(f"--- Slicing {clip_name} ({clip.start_time:.1f}s - {clip.end_time:.1f}s, Score: {score:.2f}) ---")
            
            # Trim
            trimmed_path = os.path.join(OUTPUT_DIR, f"trimmed_{task_id}_{idx+1}.mp4")
            media_editor.trim(
                media_file=media_file,
                start_time=clip.start_time,
                end_time=clip.end_time,
                trimmed_media_file_path=trimmed_path
            )
            log(f"Finished trimming {clip_name}.")
            
            # Resize
            resized_path = os.path.join(OUTPUT_DIR, f"resized_{task_id}_{idx+1}.mp4")
            resized_ok = False
            try:
                log("Attempting smart 9:16 aspect ratio resizing...")
                crops = resize(
                    video_file_path=trimmed_path,
                    pyannote_auth_token=hf_token,
                    aspect_ratio=(9, 16)
                )
                media_editor.resize_video(
                    original_video_file=AudioVideoFile(trimmed_path),
                    resized_video_file_path=resized_path,
                    width=crops.crop_width,
                    height=crops.crop_height,
                    segments=crops.to_dict()["segments"]
                )
                log(f"Smart resizing completed for {clip_name}.")
                resized_ok = True
                working_path = resized_path
            except Exception as resize_err:
                log(f"Resizing failed or bypassed: {resize_err}. Falling back to default ratio.")
                working_path = trimmed_path

            # Subtitles
            final_subtitled = create_animated_subtitles(working_path, transcription, clip, working_path, log)
            
            # Title
            clip_text = " ".join([w["word"] for w in transcription.get_word_info() 
                                  if w["start_time"] >= clip.start_time and w["end_time"] <= clip.end_time])
            title = get_viral_title(clip_text, groq_key, log)
            
            # Save final file named after the viral title
            safe_title = safe_filename(title).strip()
            if not safe_title:
                safe_title = f"Clip_{idx+1}"
            final_filename = f"{safe_title}_{task_id}_{idx+1}.mp4"
            final_dest = os.path.join(OUTPUT_DIR, final_filename)
            
            shutil.copy(final_subtitled, final_dest)
            log(f"Saved final clip: {final_filename}")
            
            # Cleanup temp files
            for temp_f in [trimmed_path, resized_path, final_subtitled]:
                if temp_f != final_dest and os.path.exists(temp_f):
                    try:
                        os.remove(temp_f)
                    except Exception:
                        pass
            
            output_clips.append({
                "filename": final_filename,
                "title": title,
                "score": round(score, 3),
                "duration": round(clip.end_time - clip.start_time, 1),
                "start": round(clip.start_time, 1),
                "end": round(clip.end_time, 1)
            })

            with status_lock:
                tasks_status[task_id]["progress"] = int(((idx + 1) / len(selected_clips)) * 100)
                tasks_status[task_id]["clips"] = output_clips

        log("Pipeline completed successfully!")
        with status_lock:
            tasks_status[task_id]["status"] = "completed"
            tasks_status[task_id]["progress"] = 100

    except Exception as err:
        log(f"PIPELINE ERROR: {err}")
        with status_lock:
            tasks_status[task_id]["status"] = "failed"

def download_youtube_thread(task_id: str, url: str):
    import re
    def log(message: str):
        with status_lock:
            tasks_status[task_id]["logs"].append(message)

    def extract_video_id(yt_url: str) -> Optional[str]:
        pattern = r'(?:https?://)?(?:www\.)?(?:youtube\.com/(?:watch\?v=|embed/|live/|shorts/)|youtu\.be/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, yt_url)
        return match.group(1) if match else None

    target_path = os.path.join(INPUT_DIR, f"yt_download_{task_id}.mp4")
    download_success = False

    try:
        video_id = extract_video_id(url)
        if video_id:
            log(f"Extracted YouTube Video ID: {video_id}")
            log("Attempting instant API download (ANDROID_VR client)...")
            r = requests.post(
                "https://www.youtube.com/youtubei/v1/player",
                params={"key": "AIzaSyDCU8hByM-4DrUqRUYnGn-3llEO78bcxq8"},
                json={
                    "videoId": video_id,
                    "context": {
                        "client": {
                            "clientName": "ANDROID_VR",
                            "clientVersion": "1.57.29",
                            "androidSdkVersion": 30,
                            "hl": "en",
                            "gl": "US"
                        }
                    }
                },
                timeout=12
            )
            if r.status_code == 200:
                data = r.json()
                play_status = data.get("playabilityStatus", {}).get("status")
                if "streamingData" in data:
                    formats = data["streamingData"].get("formats", [])
                    download_url = None
                    for f in formats:
                        if "url" in f:
                            download_url = f["url"]
                            break
                    if download_url:
                        log("Direct download URL acquired. Streaming file...")
                        response = requests.get(download_url, stream=True, timeout=30)
                        response.raise_for_status()
                        with open(target_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=1024*1024):
                                if chunk:
                                    f.write(chunk)
                        log("Instant API download completed successfully!")
                        download_success = True
                    else:
                        log("API returned streaming data, but no direct combined video URL was found.")
                else:
                    log(f"API playability check failed/restricted: {play_status}")
            else:
                log(f"API request failed with status: {r.status_code}")
        else:
            log("Could not extract a valid YouTube video ID from URL. Falling back...")

    except Exception as api_err:
        log(f"Instant API download method failed/bypassed: {api_err}")

    # Fallback to standard yt-dlp
    if not download_success:
        try:
            log("Running standard yt-dlp downloader fallback (with Node.js engine)...")
            output_tmpl = os.path.join(INPUT_DIR, f"yt_download_{task_id}.%(ext)s")
            
            import sys
            ytdlp_path = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
            if os.name == "nt":
                ytdlp_path += ".exe"
            if not os.path.exists(ytdlp_path):
                ytdlp_path = "yt-dlp"

            cmd = [
                ytdlp_path,
                "--js-runtimes", "node",
                "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
                "--merge-output-format", "mp4",
                "-o", output_tmpl,
                url
            ]
            
            log(f"Executing command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed: {result.stderr}")
                
            if not os.path.exists(target_path):
                downloaded_files = [f for f in os.listdir(INPUT_DIR) if f.startswith(f"yt_download_{task_id}")]
                if downloaded_files:
                    # Rename/copy it to target_path
                    shutil.move(os.path.join(INPUT_DIR, downloaded_files[0]), target_path)
                else:
                    raise FileNotFoundError("Could not find downloaded file.")
            
            log("yt-dlp download completed successfully!")
            download_success = True
        except Exception as e:
            log(f"YouTube download failed: {e}")
            with status_lock:
                tasks_status[task_id]["status"] = "failed"
            return

    # If successfully downloaded by either method
    if download_success:
        log(f"YouTube download finalized: {os.path.basename(target_path)}")
        with status_lock:
            tasks_status[task_id]["status"] = "idle"  # Ready to process
            tasks_status[task_id]["video_path"] = target_path

# Endpoints
@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """Uploads an MP4 video file."""
    if not file.filename.endswith('.mp4'):
        raise HTTPException(status_code=400, detail="Only MP4 videos are supported.")
        
    task_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(INPUT_DIR, f"upload_{task_id}.mp4")
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    with status_lock:
        tasks_status[task_id] = {
            "status": "idle",
            "progress": 0,
            "logs": [f"Uploaded file saved as {os.path.basename(file_path)}"],
            "video_path": file_path,
            "clips": []
        }
        
    return {"task_id": task_id, "filename": file.filename}

@app.post("/api/youtube")
async def download_youtube(background_tasks: BackgroundTasks, url: str = Form(...)):
    """Triggers background download of a YouTube video."""
    task_id = str(uuid.uuid4())[:8]
    
    with status_lock:
        tasks_status[task_id] = {
            "status": "downloading",
            "progress": 0,
            "logs": [f"Queued YouTube download for URL: {url}"],
            "video_path": None,
            "clips": []
        }
        
    background_tasks.add_task(download_youtube_thread, task_id, url)
    return {"task_id": task_id}

@app.post("/api/process/{task_id}")
async def process_video(
    task_id: str,
    background_tasks: BackgroundTasks,
    hf_token: str = Form("your_huggingface_token_here"),
    groq_key: str = Form("your_groq_api_key_here"),
    min_dur: int = Form(10),
    max_dur: int = Form(30),
    model_size: str = Form("tiny"),
    max_clips: int = Form(2)
):
    """Triggers background processing of the uploaded or downloaded video."""
    with status_lock:
        if task_id not in tasks_status:
            raise HTTPException(status_code=404, detail="Task not found.")
        task = tasks_status[task_id]
        if task["status"] in ["downloading", "running"]:
            raise HTTPException(status_code=400, detail="Task is already busy.")
            
        task["status"] = "running"
        task["progress"] = 0
        task["logs"].append("Triggering ClippedAI video processing...")
        video_path = task["video_path"]

    background_tasks.add_task(
        pipeline_thread,
        task_id,
        video_path,
        hf_token,
        groq_key,
        min_dur,
        max_dur,
        model_size,
        max_clips
    )
    return {"status": "started"}

@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Gets status, progress, logs and output clips of a task."""
    with status_lock:
        if task_id not in tasks_status:
            raise HTTPException(status_code=404, detail="Task not found.")
        return JSONResponse(tasks_status[task_id])

@app.get("/api/clips")
async def list_all_clips():
    """Lists all final processed video files in the output directory."""
    files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.mp4') and not f.startswith('temp_') and not f.startswith('trimmed_') and not f.startswith('resized_')]
    clips_list = []
    for f in files:
        path = os.path.join(OUTPUT_DIR, f)
        clips_list.append({
            "filename": f,
            "size": os.path.getsize(path),
            "url": f"/output/{f}"
        })
    # Sort files by newest first
    clips_list.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIR, x["filename"])), reverse=True)
    return clips_list

# Serve generated videos
@app.get("/output/{filename}")
async def get_output_video(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Video not found.")
    return FileResponse(path, media_type="video/mp4")

# Serve website static directory at root
app.mount("/", StaticFiles(directory="website", html=True), name="website")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
