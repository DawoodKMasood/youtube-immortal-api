import os
import random
import shutil
import traceback
import uuid
import secrets
from enum import Enum as PyEnum
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
import subprocess
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks, Header
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import ffmpeg
from dotenv import load_dotenv
from fastapi.param_functions import Form as FormAlias
import redis

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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

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
executor = ThreadPoolExecutor(max_workers=4)

# Lock key for ensuring single video processing
PROCESSING_LOCK_KEY = "video_processing_lock"

# Function to acquire lock
def acquire_lock():
    return redis_client.set(PROCESSING_LOCK_KEY, "locked", nx=True, ex=3600)  # Lock expires after 1 hour

# Function to release lock
def release_lock():
    redis_client.delete(PROCESSING_LOCK_KEY)

# Redis queue functions
def enqueue_video(video_id):
    redis_client.rpush("video_queue", video_id)

def get_queue_position(video_id):
    queue = redis_client.lrange("video_queue", 0, -1)
    return queue.index(str(video_id).encode()) + 1 if str(video_id).encode() in queue else 0

def dequeue_video():
    return redis_client.lpop("video_queue")

def process_queue():
    while True:
        if acquire_lock():
            try:
                video_id = dequeue_video()
                if video_id:
                    video_id = int(video_id)
                    db = SessionLocal()
                    try:
                        db_video = db.query(Video).filter(Video.id == video_id).first()
                        if db_video and db_video.status == VideoStatus.PENDING:
                            process_video_background(db_video.filename, db, video_id)
                    finally:
                        db.close()
                else:
                    break  # No more videos in the queue
            finally:
                release_lock()
        else:
            # Another instance is processing, wait and try again
            time.sleep(5)

# Caching decorator
def cache_response(expire_time=3600):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            cache_key = f"cache:{func.__name__}:{str(args)}:{str(kwargs)}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, expire_time, json.dumps(result))
            return result
        return wrapper
    return decorator

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
    if not is_landscape(info['width'], info['height'], info['rotation']):
        raise ValueError("Video must be in landscape orientation")
    if min(info['width'], info['height']) < 720:
        raise ValueError("Video resolution must be at least 720p")
    if info['fps'] < 30:
        raise ValueError("Video frame rate must be at least 30 FPS")
    return info

def get_video_info(file_path):
    probe = ffmpeg.probe(file_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {file_path}")
    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'duration': float(video_stream['duration']),
        'rotation': int(video_stream.get('tags', {}).get('rotate', '0')),
        'fps': float(eval(video_stream['r_frame_rate']))
    }

def process_video_background(filename: str, db: Session, video_id: int):
    try:
        db_video = db.query(Video).filter(Video.id == video_id).first()
        db_video.status = VideoStatus.PROCESSING
        db.commit()

        redis_client.setex(f"video:{video_id}:status", 3600, json.dumps({
            "status": VideoStatus.PROCESSING,
            "queue_position": 0
        }))

        print(f"Starting video processing for video_id: {video_id}")

        file_path = os.path.join(UPLOAD_DIR, filename)
        adjusted_path = os.path.join(UPLOAD_DIR, f"adjusted_{filename}")
        output_path = os.path.join(OUTPUT_DIR, f"final_{filename}")
        thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{video_id}.jpg")

        # Step 1: Validate video
        print(f"Validating video: {file_path}")
        validate_video(file_path)

        # Step 2: Adjust aspect ratio
        print(f"Adjusting aspect ratio: {file_path} -> {adjusted_path}")
        adjust_aspect_ratio(file_path, adjusted_path)

        # Step 3: Combine videos (add intro, outro, watermark, background music)
        print(f"Combining videos: {adjusted_path} -> {output_path}")
        bg_music_path = get_random_music()
        combine_videos(adjusted_path, output_path, bg_music_path)

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
        db.commit()

        redis_client.setex(f"video:{video_id}:status", 3600, json.dumps({
            "status": VideoStatus.FAILED,
            "queue_position": 0
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

        # Process the next video in the queue
        process_queue()

def combine_videos(main_video: str, output_path: str, custom_bg_music: str = None):
    intro_file = INTRO_VIDEO
    main_file = main_video
    outro_file = OUTRO_VIDEO
    watermark_file = WATERMARK

    main_info = ffmpeg.probe(main_file)
    main_video_stream = next((stream for stream in main_info['streams'] if stream['codec_type'] == 'video'), None)
    if main_video_stream is None:
        raise ValueError(f"No video stream found in {main_file}")
    
    fps = eval(main_video_stream['r_frame_rate'])

    intro = ffmpeg.input(intro_file)
    main = ffmpeg.input(main_file)
    outro = ffmpeg.input(outro_file)
    watermark = ffmpeg.input(watermark_file)

    intro_duration = float(ffmpeg.probe(intro_file)['streams'][0]['duration'])
    main_duration = float(main_info['streams'][0]['duration'])
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

    video = ffmpeg.concat(intro['v'], main_with_fade, outro['v'], v=1, a=0)

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
        bg_music_adjusted = concatenated_bg_music.filter('volume', volume=0.5)
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

    output = ffmpeg.output(video, mixed_audio, output_path, format='mp4', r=fps)
    ffmpeg.run(output, overwrite_output=True)

def is_landscape(width, height, rotation):
    if rotation in [90, 270]:
        width, height = height, width
    return width > height

def adjust_aspect_ratio(input_path, output_path):
    probe = ffmpeg.probe(input_path)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    
    input_stream = ffmpeg.input(input_path)
    video = input_stream.video.filter('scale', w='iw*min(1920/iw,1080/ih)', h='ih*min(1920/iw,1080/ih)')
    video = video.filter('crop', w='1920', h='1080')
    
    stream = ffmpeg.output(video, input_stream.audio, output_path) if audio_stream else ffmpeg.output(video, output_path)
    
    try:
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
    except ffmpeg.Error as e:
        print(f"FFmpeg stderr:\n{e.stderr.decode()}")
        raise

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

@app.post("/upload/")
async def upload_video(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    account_name: str = Form(...),
    game_mode: str = Form(...),
    weapon: str = Form(...),
    map_name: str = Form(...),
    background_music: Union[UploadFile, str, None] = FormAlias(default=None),
):
    if not file.filename.lower().endswith(('.mp4')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    file_uuid = uuid.uuid4()
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{file_uuid}{file_extension}"
    
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    db_video = Video(
        filename=safe_filename,
        status=VideoStatus.PENDING,
        account_name=account_name,
        game_mode=game_mode,
        weapon=weapon,
        map_name=map_name
    )
    db.add(db_video)
    db.commit()
    db.refresh(db_video)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    print(f"Video file saved: {file_path}, Size: {os.path.getsize(file_path)} bytes")
    
    enqueue_video(db_video.id)
    queue_position = get_queue_position(db_video.id)

    redis_client.setex(f"video:{db_video.id}:status", 3600, json.dumps({
        "status": VideoStatus.PENDING,
        "queue_position": queue_position
    }))

    background_tasks.add_task(process_queue)
    
    return {"video_id": db_video.id, "status": VideoStatus.PENDING, "queue_position": queue_position}

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
@cache_response(expire_time=300)  # Cache for 5 minutes
def list_videos(db: Session = Depends(get_db)):
    videos = db.query(Video).filter(Video.status.in_([VideoStatus.COMPLETED, VideoStatus.APPROVED, VideoStatus.REJECTED])).all()
    return [{
        "id": video.id,
        "status": video.status,
        "account_name": video.account_name,
        "game_mode": video.game_mode,
        "weapon": video.weapon,
        "map_name": video.map_name,
        "created_at": video.created_at,
        "updated_at": video.updated_at
    } for video in videos]

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

@app.post("/admin/reset-database")
async def reset_database(db: Session = Depends(get_db), admin_password: str = Header(...)):
    if not secrets.compare_digest(admin_password, '1nf0rmM@tic$'):
        raise HTTPException(status_code=403, detail="Invalid admin password")
    
    try:
        db.query(Video).delete()
        db.commit()

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

        return {"message": "Database reset successful. All video records and files have been deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while resetting the database: {str(e)}")

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    health_status = {
        "status": "healthy",
        "checks": {}
    }

    try:
        # Check database connection
        db.execute(text("SELECT 1"))
        health_status["checks"]["database"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = f"error: {str(e)}"

    # Check required directories
    required_dirs = [UPLOAD_DIR, OUTPUT_DIR, THUMBNAIL_DIR, MUSIC_DIR]
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            health_status["checks"][dir_name] = "exists"
        else:
            health_status["status"] = "unhealthy"
            health_status["checks"][dir_name] = "missing"

    # Check for required files
    required_files = [INTRO_VIDEO, OUTRO_VIDEO, WATERMARK]
    for file_name in required_files:
        if os.path.exists(file_name) and os.path.isfile(file_name):
            health_status["checks"][file_name] = "exists"
        else:
            health_status["status"] = "unhealthy"
            health_status["checks"][file_name] = "missing"

    try:
        redis_client.ping()
        health_status["checks"]["redis"] = "connected"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["redis"] = f"error: {str(e)}"

    # For Render: Always return a 200 OK status
    # Include detailed health information in the response body
    return health_status

@app.on_event("startup")
async def startup_event():
    if not check_ffmpeg():
        print("Warning: FFmpeg is not installed or not accessible. Video processing may fail.")

@app.on_event("startup")
async def startup_event():
    check_and_set_permissions([UPLOAD_DIR, OUTPUT_DIR, THUMBNAIL_DIR])

# Start the queue processing in a background task
@app.on_event("startup")
def start_queue_processing():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(process_queue)

port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)