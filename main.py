import os
import random
import shutil
import tempfile
import traceback
import uuid
import secrets
from enum import Enum as PyEnum
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import subprocess
import json
import psutil
import redis

from fastapi import FastAPI, File, Query, UploadFile, HTTPException, Depends, Form, BackgroundTasks, Header
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum, desc, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import ffmpeg
from dotenv import load_dotenv
from fastapi.param_functions import Form as FormAlias
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import Index

# Load environment variables from .env file
load_dotenv()

# Get the database URL from the environment variable
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
redis_client = redis.from_url(REDIS_URL)

# Model
class VideoStatus(str, PyEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    UPLOADED = "UPLOADED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    status = Column(Enum(VideoStatus), default=VideoStatus.PENDING)
    account_name = Column(String)
    game_mode = Column(String)
    weapon = Column(String)
    map_name = Column(String)
    thumbnail = Column(String)
    background_music = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index('ix_videos_status', 'status'),
    )

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # For Render | For Windows: os.path.dirname(os.path.abspath(__file__))

UPLOAD_DIR = os.path.join(UPLOAD_BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(UPLOAD_BASE_DIR, "outputs")
THUMBNAIL_DIR = os.path.join(UPLOAD_BASE_DIR, "thumbnails")

MUSIC_DIR = os.path.join(BASE_DIR, "musics")
INTRO_VIDEO = os.path.join(BASE_DIR, "INTRO.mp4")
OUTRO_VIDEO = os.path.join(BASE_DIR, "OUTRO.mp4")
WATERMARK = os.path.join(BASE_DIR, "WATERMARK.png")

for directory in [UPLOAD_DIR, OUTPUT_DIR, THUMBNAIL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=5)

# Lock key for ensuring single video processing
PROCESSING_LOCK_KEY = "video_processing_lock"

def validate_file_integrity(file_path):
    try:
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError("File is empty")
        
        # Quick check for MP4 file signature
        with open(file_path, 'rb') as f:
            file_start = f.read(8)
        if file_start[4:8] not in (b'ftyp', b'moov'):
            raise ValueError("File does not appear to be a valid MP4")
        
        return True
    except Exception as e:
        print(f"File integrity check failed: {str(e)}")
        return False
    
# Function to acquire lock
def acquire_lock():
    lock = redis_client.lock("video_processing_lock", timeout=300)  # 5 minutes timeout
    acquired = lock.acquire(blocking=False)
    if acquired:
        print("Lock acquired successfully")
    else:
        print("Failed to acquire lock")
    return acquired, lock

# Function to release lock
def release_lock(lock):
    try:
        lock.release()
        print("Lock released successfully")
    except redis.exceptions.LockError:
        print("Error releasing lock: Lock was not held")

def enqueue_video(video_id):
    redis_client.rpush("video_queue", video_id)
    print(f"Enqueued video {video_id}. Current queue: {redis_client.lrange('video_queue', 0, -1)}")

def get_queue_position(video_id):
    queue = redis_client.lrange("video_queue", 0, -1)
    try:
        return queue.index(str(video_id).encode()) + 1
    except ValueError:
        return 0

def dequeue_video():
    return redis_client.lpop("video_queue")

def clean_queue():
    db = SessionLocal()
    try:
        queue = redis_client.lrange("video_queue", 0, -1)
        for video_id in queue:
            video_id = int(video_id)
            video = db.query(Video).filter(Video.id == video_id).first()
            if not video or video.status != VideoStatus.PENDING:
                redis_client.lrem("video_queue", 0, video_id)
                print(f"Removed video {video_id} from queue as it's not pending")
    finally:
        db.close()
    print(f"Cleaned queue. Current queue: {redis_client.lrange('video_queue', 0, -1)}")

def check_and_update_long_running_processes(db: Session):
    thirty_minutes_ago = datetime.utcnow() - timedelta(minutes=30)
    long_running_videos = db.query(Video).filter(
        Video.status == VideoStatus.PROCESSING,
        Video.updated_at <= thirty_minutes_ago
    ).all()

    for video in long_running_videos:
        video.status = VideoStatus.FAILED
        redis_client.setex(f"video:{video.id}:status", 3600, json.dumps({
            "status": VideoStatus.FAILED,
            "error_message": "Video processing timed out after 30 minutes"
        }))
        print(f"Video {video.id} marked as failed due to timeout")
    
    db.commit()
    
def process_queue():
    pubsub = redis_client.pubsub()
    pubsub.subscribe('video_processed')
    
    empty_queue_sleep_time = 30  # seconds
    while True:
        # Check for long-running processes
        db = SessionLocal()
        try:
            check_and_update_long_running_processes(db)
        finally:
            db.close()

        if redis_client.llen("video_queue") == 0:
            time.sleep(empty_queue_sleep_time)
            continue

        acquired, lock = acquire_lock()
        if acquired:
            try:
                video_id = dequeue_video()
                if video_id:
                    video_id = int(video_id)
                    db = SessionLocal()
                    try:
                        db_video = db.query(Video).filter(Video.id == video_id).first()
                        if db_video and db_video.status == VideoStatus.PENDING:
                            db_video.status = VideoStatus.PROCESSING
                            db_video.updated_at = datetime.utcnow()
                            db.commit()

                            process_video_background(db_video.filename, db, video_id)
                            
                            # Wait for the 'video_processed' signal
                            for message in pubsub.listen():
                                if message['type'] == 'message':
                                    break
                        else:
                            print(f"Skipping video {video_id}: status is {db_video.status if db_video else 'None'}")
                    except Exception as e:
                        print(f"Error processing video {video_id}: {str(e)}")
                    finally:
                        db.close()
                else:
                    time.sleep(5)
            finally:
                release_lock(lock)
        else:
            time.sleep(5)

def check_and_set_permissions(directories):
       for dir in directories:
           if not os.path.exists(dir):
               os.makedirs(dir, exist_ok=True)
           os.chmod(dir, 0o755)  # rwxr-xr-x
           print(f"Permissions set for {dir}")

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"FFmpeg is installed. Version info:\n{result.stdout}")
            return True
        else:
            print(f"FFmpeg check failed. Error:\n{result.stderr}")
            return False
    except Exception as e:
        print(f"Error checking FFmpeg: {str(e)}")
        return False
    
def get_random_music(exclude=None):
    music_folder = MUSIC_DIR
    music_files = [f for f in os.listdir(music_folder) if f.endswith(('.mp3', '.wav'))]
    if exclude:
        music_files = [f for f in music_files if f != exclude]
    if not music_files:
        raise ValueError("No suitable music files found in the 'musics' folder")
    return os.path.join(music_folder, random.choice(music_files))

def validate_video(file_path):
    info = get_video_info(file_path)
    
    if info['width'] <= info['height']:
        raise ValueError("Video must be in landscape orientation")
    
    if info['duration'] < 10:
        raise ValueError("Video must be at least 10 seconds long")
    
    return info

def get_video_info(file_path):
    try:
        command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
                   '-show_entries', 'stream=width,height,r_frame_rate,duration',
                   '-of', 'json', file_path]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        probe = json.loads(result.stdout)
        video_stream = probe['streams'][0]
        return {
            'width': int(video_stream['width']),
            'height': int(video_stream['height']),
            'duration': float(video_stream['duration']),
            'rotation': int(video_stream.get('tags', {}).get('rotate', '0')),
            'fps': float(eval(video_stream['r_frame_rate']))
        }
    except subprocess.CalledProcessError as e:
        print(f"FFprobe error. Return code: {e.returncode}")
        print(f"FFprobe stderr: {e.stderr}")
        raise ValueError(f"FFprobe failed: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error in get_video_info: {str(e)}")
        raise

def process_video_background(filename: str, db: Session, video_id: int):
    try:
        db_video = db.query(Video).filter(Video.id == video_id).first()
        db_video.status = VideoStatus.PROCESSING
        db_video.updated_at = datetime.utcnow()  # Update the timestamp when processing starts
        db.commit()

        redis_client.setex(f"video:{video_id}:status", 3600, json.dumps({
            "status": VideoStatus.PROCESSING,
            "queue_position": 0
        }))

        print(f"Starting video processing for video_id: {video_id}")

        file_path = os.path.join(UPLOAD_DIR, filename)

        if not validate_file_integrity(file_path):
            raise ValueError("File integrity check failed before processing")
        
        adjusted_path = os.path.join(UPLOAD_DIR, f"adjusted_{filename}")
        output_path = os.path.join(OUTPUT_DIR, f"final_{filename}")
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{video_id}.jpg")

        # Step 1: Validate video
        print(f"Validating video: {file_path}")
        validate_video(file_path)

        # Step 2: Adjust aspect ratio
        print(f"Adjusting aspect ratio: {file_path} -> {adjusted_path}")
        try:
            adjust_aspect_ratio(file_path, adjusted_path)
        except Exception as e:
            print(f"Error adjusting aspect ratio: {str(e)}")
            raise

        # Check if custom background music was provided
        custom_bg_music = None
        if db_video.background_music:
            custom_bg_music = os.path.join(MUSIC_DIR, db_video.background_music)
            if not os.path.exists(custom_bg_music):
                print(f"Custom background music not found: {custom_bg_music}")
                custom_bg_music = None

        # Step 3: Combine videos (add intro, outro, watermark, background music)
        print(f"Combining videos: {adjusted_path} -> {output_path}")
        combine_videos(adjusted_path, output_path, custom_bg_music)

        # Step 4: Generate thumbnail
        print(f"Generating thumbnail: {file_path} -> {thumbnail_path}")
        generate_thumbnail(file_path, thumbnail_path)

        db_video.status = VideoStatus.COMPLETED
        db_video.thumbnail = os.path.basename(thumbnail_path)
        db.commit()

        redis_client.setex(f"video:{video_id}:status", 3600, json.dumps({
            "status": VideoStatus.COMPLETED,
            "queue_position": 0
        }))

        print(f"Video processing completed for video_id: {video_id}")
    except Exception as e:
        print(f"Error processing video {video_id}: {str(e)}")
        traceback.print_exc()
        db_video.status = VideoStatus.FAILED
        db_video.updated_at = datetime.utcnow()  # Update the timestamp when processing fails
        db.commit()

        redis_client.setex(f"video:{video_id}:status", 3600, json.dumps({
            "status": VideoStatus.FAILED,
            "error_message": str(e)
        }))

        raise
    finally:
        # Clean up temporary files
        for path in [file_path, adjusted_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"Removed temporary file: {path}")
                except Exception as e:
                    print(f"Error removing file {path}: {str(e)}")

        # Signal that processing is complete
        redis_client.publish('video_processed', video_id)

def combine_videos(main_video: str, output_path: str, custom_bg_music: str = None):
    intro_file = INTRO_VIDEO
    main_file = main_video
    outro_file = OUTRO_VIDEO
    watermark_file = WATERMARK

    def get_video_fps(file_path):
        probe = ffmpeg.probe(file_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if video_stream:
            fps = eval(video_stream['r_frame_rate'])
            return fps if isinstance(fps, (int, float)) else fps.numerator / fps.denominator
        return None

    main_fps = get_video_fps(main_file)
    if main_fps is None:
        raise ValueError("Could not determine frame rate of main video")

    target_fps = min(30, main_fps)  # Use original FPS if less than 30, otherwise cap at 30

    def scale_video(input_file):
        return (
            ffmpeg.input(input_file)
            .filter('scale', 1920, 1080, force_original_aspect_ratio='decrease')
            .filter('pad', 1920, 1080, '(ow-iw)/2', '(oh-ih)/2')
            .filter('setsar', 1)
            .filter('fps', fps=target_fps)
        )

    intro = scale_video(intro_file)
    main = scale_video(main_file)
    outro = scale_video(outro_file)
    watermark = ffmpeg.input(watermark_file)

    intro_duration = float(ffmpeg.probe(intro_file)['streams'][0]['duration'])
    main_duration = float(ffmpeg.probe(main_file)['streams'][0]['duration'])
    outro_duration = float(ffmpeg.probe(outro_file)['streams'][0]['duration'])
    total_duration = intro_duration + main_duration + outro_duration

    watermark_scaled = watermark.filter('scale', w=200, h=200)
    main_with_watermark = ffmpeg.overlay(main, watermark_scaled, x=10, y=10)

    fade_duration = 2
    main_with_fade = (
        main_with_watermark
        .filter('fade', type='in', duration=fade_duration)
        .filter('fade', type='out', start_time=main_duration-fade_duration, duration=fade_duration)
    )

    video = ffmpeg.concat(intro, main_with_fade, outro)

    bg_music_files = []
    current_duration = 0
    fade_duration = 2
    last_music_file = None

    while current_duration < total_duration:
        if not bg_music_files and custom_bg_music:
            music_file = custom_bg_music
        else:
            music_file = get_random_music(exclude=last_music_file)
        
        music_duration = float(ffmpeg.probe(music_file)['streams'][0]['duration'])
        
        if current_duration + music_duration > total_duration:
            music_duration = total_duration - current_duration
        
        bg_music_files.append((music_file, music_duration))
        current_duration += music_duration
        last_music_file = music_file

    bg_music_tracks = []
    for music_file, duration in bg_music_files:
        bg_music = ffmpeg.input(music_file)
        segment_fade_duration = min(fade_duration, duration / 2)
        bg_music = (
            bg_music
            .filter('atrim', duration=duration)
            .filter('afade', type='in', duration=segment_fade_duration)
            .filter('afade', type='out', start_time=max(0, duration-segment_fade_duration), duration=segment_fade_duration)
        )
        bg_music_tracks.append(bg_music)

    if bg_music_tracks:
        concatenated_bg_music = ffmpeg.concat(*bg_music_tracks, v=0, a=1)
        bg_music_adjusted = concatenated_bg_music.filter('volume', volume=0.35)
    else:
        bg_music_adjusted = ffmpeg.input('anullsrc=r=44100:cl=stereo', f='lavfi', t=total_duration)

    audio_inputs = []
    for input_file in [intro_file, main_file, outro_file]:
        audio_streams = ffmpeg.probe(input_file)['streams']
        audio_stream = next((stream for stream in audio_streams if stream['codec_type'] == 'audio'), None)
        
        if audio_stream:
            audio_inputs.append(ffmpeg.input(input_file)['a'])
        else:
            silent_duration = float(ffmpeg.probe(input_file)['streams'][0]['duration'])
            silent_audio = ffmpeg.input('anullsrc=r=44100:cl=stereo', f='lavfi', t=silent_duration)
            audio_inputs.append(silent_audio)

    original_audio = ffmpeg.concat(*audio_inputs, v=0, a=1)

    mixed_audio = ffmpeg.filter([original_audio, bg_music_adjusted], 'amix', inputs=2)

    output_params = {
        'vcodec': 'libx264',
        'preset': 'veryfast',
        'crf': '23',
        'acodec': 'aac',
        'audio_bitrate': '192k',
        'r': target_fps
    }

    output = ffmpeg.output(video, mixed_audio, output_path, **output_params)
    
    try:
        ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print(f"FFmpeg stderr:\n{e.stderr.decode()}")
        raise Exception(f"FFmpeg command failed: {e.stderr.decode()}")

def is_landscape(width, height, rotation):
    if rotation in [90, 270]:
        width, height = height, width
    return width > height

def adjust_aspect_ratio(input_path, output_path):
    probe = ffmpeg.probe(input_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    
    if not video_stream:
        raise ValueError("No video stream found in the input file")

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    
    input_stream = ffmpeg.input(input_path)
    
    # Calculate scaling and padding
    target_aspect_ratio = 16 / 9
    current_aspect_ratio = width / height

    if current_aspect_ratio > target_aspect_ratio:  # If wider than 16:9
        new_width = width
        new_height = int(width / target_aspect_ratio)
        pad_width = width
        pad_height = new_height
    else:  # If taller than 16:9 or exactly 16:9
        new_height = height
        new_width = int(height * target_aspect_ratio)
        pad_width = new_width
        pad_height = height

    # Scale the video
    video = (
        input_stream
        .filter('scale', w=new_width, h=new_height)
        .filter('pad', w=pad_width, h=pad_height, x='(ow-iw)/2', y='(oh-ih)/2')
    )

    output_params = {
        'vcodec': 'libx264',
        'preset': 'veryfast',
        'crf': '23',
        'acodec': 'aac',
        'audio_bitrate': '192k'
    }
    
    if audio_stream:
        output = ffmpeg.output(video, input_stream.audio, output_path, **output_params)
    else:
        output = ffmpeg.output(video, output_path, **output_params)
    
    try:
        ffmpeg.run(output, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print(f"FFmpeg stderr:\n{e.stderr.decode()}")
        raise Exception(f"FFmpeg command failed: {e.stderr.decode()}")
    except Exception as e:
        print(f"Error in adjust_aspect_ratio: {str(e)}")
        raise

def generate_unique_filename(original_filename):
    unique_id = str(uuid.uuid4())
    name, extension = os.path.splitext(original_filename)
    return f"{name}_{unique_id}{extension}"

def generate_thumbnail(input_path, output_path):
    duration = float(ffmpeg.probe(input_path)['streams'][0]['duration'])
    (
        ffmpeg
        .input(input_path, ss=duration/2)
        .filter('scale', 480, -1)
        .output(output_path, vframes=1)
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

# New helper function for chunk uploads
def save_upload_file_tmp(upload_file: UploadFile) -> str:
    try:
        suffix = os.path.splitext(upload_file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(upload_file.file, tmp)
        return tmp.name
    finally:
        upload_file.file.close()

@app.post("/upload-music-chunk/")
async def upload_music_chunk(
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    db: Session = Depends(get_db)
):
    chunk_dir = os.path.join(MUSIC_DIR, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    if chunk_number == 1:
        unique_filename = generate_unique_filename(filename)
        redis_client.setex(f"music_upload:{filename}", 3600, unique_filename)
    else:
        unique_filename = redis_client.get(f"music_upload:{filename}")
        if not unique_filename:
            raise HTTPException(status_code=400, detail="Upload session expired or invalid")
        unique_filename = unique_filename.decode('utf-8')
    
    file_uuid = unique_filename.split('.')[0]
    chunk_filename = f"{file_uuid}_{chunk_number}"
    chunk_path = os.path.join(chunk_dir, chunk_filename)
    
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if chunk_number == total_chunks:
        final_filename = unique_filename
        final_path = os.path.join(MUSIC_DIR, final_filename)
        
        with open(final_path, "wb") as final_file:
            for i in range(1, total_chunks + 1):
                chunk_file = os.path.join(chunk_dir, f"{file_uuid}_{i}")
                if os.path.exists(chunk_file):
                    with open(chunk_file, "rb") as cf:
                        shutil.copyfileobj(cf, final_file)
                    os.remove(chunk_file)
                else:
                    raise HTTPException(status_code=400, detail=f"Music chunk file {i} is missing")
        
        redis_client.delete(f"music_upload:{filename}")
        
        return JSONResponse(content={
            "message": "Music upload complete",
            "filename": final_filename
        })
    else:
        return JSONResponse(content={"message": f"Music chunk {chunk_number} of {total_chunks} received"})

@app.post("/process-video/")
async def process_video(
    video_data: dict,
    db: Session = Depends(get_db)
):
    video_filename = video_data.get("video_filename")
    music_filename = video_data.get("music_filename")

    if not video_filename:
        raise HTTPException(status_code=400, detail="Video filename is required")

    video = db.query(Video).filter(Video.filename == video_filename).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video.background_music = music_filename
    video.status = VideoStatus.PENDING
    db.commit()

    enqueue_video(video.id)
    clean_queue()
    queue_position = get_queue_position(video.id)

    redis_client.setex(f"video:{video.id}:status", 3600, json.dumps({
        "status": VideoStatus.PENDING,
        "queue_position": queue_position
    }))

    return JSONResponse(content={
        "message": "Video queued for processing",
        "video_id": video.id,
        "status": VideoStatus.PENDING,
        "queue_position": queue_position
    })

@app.post("/upload-video-chunk/")
async def upload_video_chunk(
    file: UploadFile = File(...),
    chunk_number: int = Form(...),
    total_chunks: int = Form(...),
    filename: str = Form(...),
    account_name: str = Form(...),
    game_mode: str = Form(...),
    weapon: str = Form(...),
    map_name: str = Form(...),
    db: Session = Depends(get_db)
):
    chunk_dir = os.path.join(UPLOAD_DIR, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    
    if chunk_number == 1:
        unique_filename = generate_unique_filename(filename)
        redis_client.setex(f"upload:{filename}", 3600, unique_filename)
    else:
        unique_filename = redis_client.get(f"upload:{filename}")
        if not unique_filename:
            raise HTTPException(status_code=400, detail="Upload session expired or invalid")
        unique_filename = unique_filename.decode('utf-8')
    
    file_uuid = unique_filename.split('.')[0]
    chunk_filename = f"{file_uuid}_{chunk_number}"
    chunk_path = os.path.join(chunk_dir, chunk_filename)
    
    print(f"Receiving chunk {chunk_number} of {total_chunks} for file {unique_filename}")
    
    with open(chunk_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"Saved chunk {chunk_number} to {chunk_path}")
    
    if chunk_number == total_chunks:
        print(f"All chunks received for {unique_filename}. Combining chunks...")
        final_filename = unique_filename
        final_path = os.path.join(UPLOAD_DIR, final_filename)
        
        with open(final_path, "wb") as final_file:
            for i in range(1, total_chunks + 1):
                chunk_file = os.path.join(chunk_dir, f"{file_uuid}_{i}")
                if os.path.exists(chunk_file):
                    print(f"Appending chunk {i} to final file")
                    with open(chunk_file, "rb") as cf:
                        shutil.copyfileobj(cf, final_file)
                    os.remove(chunk_file)
                    print(f"Removed chunk file {chunk_file}")
                else:
                    print(f"Error: Chunk file {i} is missing")
                    raise HTTPException(status_code=400, detail=f"Chunk file {i} is missing")
        
        print(f"All chunks combined into {final_path}")
        
        redis_client.delete(f"upload:{filename}")
        
        if not validate_file_integrity(final_path):
            os.remove(final_path)
            raise HTTPException(status_code=400, detail="The uploaded file appears to be corrupt or incomplete. Please try uploading again.")
        
        print("File integrity validated")
        
        db_video = Video(
            filename=final_filename,
            status=VideoStatus.UPLOADED,
            account_name=account_name,
            game_mode=game_mode,
            weapon=weapon,
            map_name=map_name
        )
        db.add(db_video)
        db.commit()
        db.refresh(db_video)
        
        print(f"Database entry created for video ID: {db_video.id}")
        
        return JSONResponse(content={
            "message": "Upload complete",
            "video_id": db_video.id,
            "filename": final_filename,
            "status": VideoStatus.UPLOADED
        })
    else:
        return JSONResponse(content={"message": f"Chunk {chunk_number} of {total_chunks} received"})

@app.get("/video/{video_id}/status")
async def get_video_status(video_id: int, db: Session = Depends(get_db)):
    cached_status = redis_client.get(f"video:{video_id}:status")
    if cached_status:
        return json.loads(cached_status)
    
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    queue_position = get_queue_position(video_id)
    status_data = {
        "status": video.status,
        "queue_position": queue_position
    }
    
    redis_client.setex(f"video:{video_id}:status", 3600, json.dumps(status_data))
    return status_data

@app.get("/videos")
def list_videos(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    # Calculate offset
    offset = (page - 1) * per_page

    # Query videos, ordered by created_at descending, with pagination
    videos = db.query(Video).order_by(desc(Video.created_at)).offset(offset).limit(per_page).all()

    # Get total count of videos
    total_videos = db.query(Video).count()

    # Calculate total pages
    total_pages = (total_videos + per_page - 1) // per_page

    return {
        "videos": [{
            "id": video.id,
            "status": video.status,
            "account_name": video.account_name,
            "game_mode": video.game_mode,
            "weapon": video.weapon,
            "map_name": video.map_name,
            "created_at": video.created_at,
            "updated_at": video.updated_at
        } for video in videos],
        "page": page,
        "per_page": per_page,
        "total_videos": total_videos,
        "total_pages": total_pages
    }

@app.get("/video/{video_id}/thumbnail")
async def get_thumbnail(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None or video.thumbnail is None:
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    thumbnail_path = os.path.join(THUMBNAIL_DIR, video.thumbnail)
    if not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="Thumbnail file not found")
    
    return FileResponse(thumbnail_path, media_type="image/jpeg")

@app.get("/video/{video_id}/download")
async def download_video(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.status not in [VideoStatus.COMPLETED, VideoStatus.APPROVED]:
        raise HTTPException(status_code=400, detail="Video is not ready for download")
    
    video_path = os.path.join(OUTPUT_DIR, f"final_{video.filename}")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(video_path, media_type="video/mp4", filename=f"processed_{video.filename}")

@app.post("/admin/reset")
async def reset_everything(db: Session = Depends(get_db), admin_password: str = Header(...)):
    if not secrets.compare_digest(admin_password, '1nf0rmM@tic$'):
        raise HTTPException(status_code=403, detail="Invalid admin password")
    
    try:
        # Reset database
        db.query(Video).delete()
        db.commit()

        # Clear Redis
        redis_client.flushall()

        # Delete files
        for directory in [UPLOAD_DIR, OUTPUT_DIR, THUMBNAIL_DIR]:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f'Failed to delete {file_path}. Reason: {e}')

        # Re-initialize necessary Redis structures
        redis_client.delete("video_queue")
        redis_client.delete("video_processing_lock")

        return {"message": "Reset successful. All video records, files, and Redis data have been cleared."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred during reset: {str(e)}")

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    health_status = {
        "status": "healthy",
        "checks": {}
    }

    try:
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = f"error: {str(e)}"

    try:
        redis_client.ping()
        health_status["checks"]["redis"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["redis"] = f"error: {str(e)}"

    cpu_usage = psutil.cpu_percent()
    health_status["checks"]["cpu_usage"] = f"{cpu_usage}%"
    if cpu_usage > 80:
        health_status["status"] = "unhealthy"

    memory_usage = psutil.virtual_memory().percent
    health_status["checks"]["memory_usage"] = f"{memory_usage}%"
    if memory_usage > 80:
        health_status["status"] = "unhealthy"
    
    disk_usage = psutil.disk_usage('/')
    free_disk_percent = 100 - disk_usage.percent
    health_status["checks"]["disk_space"] = f"{free_disk_percent:.1f}% free"
    if free_disk_percent < 10:
        health_status["status"] = "unhealthy"

    return health_status

@app.on_event("startup")
async def startup_event():
    if not check_ffmpeg():
        print("Warning: FFmpeg is not installed or not accessible. Video processing may fail.")

    check_and_set_permissions([UPLOAD_DIR, OUTPUT_DIR, THUMBNAIL_DIR])

    # Clean the queue on startup
    clean_queue()

    # Requeue pending videos
    db = SessionLocal()
    try:
        pending_videos = db.query(Video).filter(Video.status == VideoStatus.PENDING).all()
        for video in pending_videos:
            if get_queue_position(video.id) == 0:  # Only enqueue if not already in queue
                enqueue_video(video.id)
        print(f"Requeued {len(pending_videos)} pending videos on startup")
    finally:
        db.close()

    # Start queue processing
    import threading
    threading.Thread(target=process_queue, daemon=True).start()
    app.state.queue_processing_started = True

    # Clear any existing locks
    redis_client.delete("video_processing_lock")
    print("Cleared existing locks on startup")


port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)