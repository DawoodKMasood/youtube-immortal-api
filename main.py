import os
import random
import shutil
import uuid
import secrets
import time
import psutil
import logging
from enum import Enum as PyEnum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks, Header
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import ffmpeg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for resource management
MAX_CPU_PERCENT = 80
MAX_MEMORY_PERCENT = 80
CHUNK_DURATION = 30  # Process 30 seconds at a time
MAX_PROCESSING_TIME = 540  # 9 minutes (leaving buffer for 10-minute timeout)

# Get the database URL from the environment variable
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Model
class VideoStatus(str, PyEnum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"

class ProcessingStage(str, PyEnum):
    VALIDATE = "validate"
    ADJUST = "adjust"
    COMBINE = "combine"
    THUMBNAIL = "thumbnail"

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    status = Column(Enum(VideoStatus), default=VideoStatus.PENDING)
    current_stage = Column(Enum(ProcessingStage))
    account_name = Column(String)
    game_mode = Column(String)
    weapon = Column(String)
    map_name = Column(String)
    thumbnail = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    current_stage = Column(Enum(ProcessingStage), nullable=True)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_BASE_DIR = "/mnt/data" # For Render | For Windows: os.path.dirname(os.path.abspath(__file__))

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

def get_resource_usage() -> Tuple[float, float]:
    cpu_percent = psutil.cpu_percent(interval=1)
    memory_percent = psutil.virtual_memory().percent
    return cpu_percent, memory_percent

def log_resource_usage():
    cpu_percent, memory_percent = get_resource_usage()
    logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")

def check_resources() -> bool:
    cpu_percent, memory_percent = get_resource_usage()
    return cpu_percent < MAX_CPU_PERCENT and memory_percent < MAX_MEMORY_PERCENT

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

def process_video(db: Session, video_id: int, background_tasks: BackgroundTasks):
    video = db.query(Video).filter(Video.id == video_id).first()
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    video.status = VideoStatus.PROCESSING
    db.commit()

    file_path = os.path.join(UPLOAD_DIR, video.filename)
    adjusted_path = os.path.join(UPLOAD_DIR, f"adjusted_{video.filename}")
    output_path = os.path.join(OUTPUT_DIR, f"final_{video.filename}")
    thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{video.id}.jpg")

    try:
        stages = [
            (ProcessingStage.VALIDATE, lambda: validate_video(file_path)),
            (ProcessingStage.ADJUST, lambda: adjust_aspect_ratio(file_path, adjusted_path)),
            (ProcessingStage.COMBINE, lambda: combine_videos(adjusted_path, output_path)),
            (ProcessingStage.THUMBNAIL, lambda: generate_thumbnail(file_path, thumbnail_path))
        ]

        for stage, process_func in stages:
            if not process_stage(db, video, stage, process_func):
                # If a stage couldn't complete, requeue the task
                background_tasks.add_task(process_video, db, video_id, background_tasks)
                return

        video.status = VideoStatus.COMPLETED
        video.thumbnail = os.path.basename(thumbnail_path)
        db.commit()
        logger.info(f"Video processing completed for video_id: {video_id}")

    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        video.status = VideoStatus.FAILED
        db.commit()
    finally:
        for path in [file_path, adjusted_path]:
            if path and os.path.exists(path):
                os.remove(path)

def process_stage(db: Session, video: Video, stage: ProcessingStage, process_func: callable) -> bool:
    start_time = time.time()
    video.current_stage = stage
    db.commit()

    while True:
        if not check_resources():
            logger.warning(f"Insufficient resources. Pausing processing of video {video.id}")
            time.sleep(60)  # Wait for a minute before checking resources again
            continue

        try:
            process_func()
            return True
        except Exception as e:
            logger.error(f"Error in {stage} stage for video {video.id}: {str(e)}")
            if time.time() - start_time > MAX_PROCESSING_TIME:
                logger.warning(f"Processing time exceeded for video {video.id} in {stage} stage")
                return False
            time.sleep(30)  # Wait for 30 seconds before retrying

def combine_videos(main_video: str, output_path: str, custom_bg_music: str = None):
    logger.info(f"Combining videos: {main_video} -> {output_path}")
    intro_file = INTRO_VIDEO
    outro_file = OUTRO_VIDEO
    watermark_file = WATERMARK

    video_duration = get_video_duration(main_video)
    chunk_paths = []

    for start_time in range(0, int(video_duration), CHUNK_DURATION):
        chunk_output = f"{output_path}.chunk_{start_time}.mp4"
        process_video_chunk(main_video, chunk_output, start_time, CHUNK_DURATION)
        chunk_paths.append(chunk_output)

    # Combine all chunks
    with open('chunk_list.txt', 'w') as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{chunk_path}'\n")

    (
        ffmpeg
        .input('chunk_list.txt', format='concat', safe=0)
        .output(output_path, codec='copy')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

    # Clean up chunk files
    for chunk_path in chunk_paths:
        os.remove(chunk_path)
    os.remove('chunk_list.txt')

    # Add intro, outro, watermark, and background music
    add_intro_outro_watermark(output_path, intro_file, outro_file, watermark_file)
    if custom_bg_music:
        add_background_music(output_path, custom_bg_music)

def process_video_chunk(input_path: str, output_path: str, start_time: float, duration: float):
    (
        ffmpeg
        .input(input_path, ss=start_time, t=duration)
        .output(output_path, codec='libx264')
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )

def get_video_duration(file_path: str) -> float:
    probe = ffmpeg.probe(file_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    return float(video_info['duration'])

def add_intro_outro_watermark(video_path: str, intro_path: str, outro_path: str, watermark_path: str):
    temp_output = f"{video_path}.temp.mp4"
    
    main = ffmpeg.input(video_path)
    intro = ffmpeg.input(intro_path)
    outro = ffmpeg.input(outro_path)
    watermark = ffmpeg.input(watermark_path)

    watermark_scaled = watermark.filter('scale', w=200, h=200)
    main_with_watermark = ffmpeg.overlay(main, watermark_scaled, x=10, y=10)

    video = ffmpeg.concat(intro, main_with_watermark, outro)
    
    output = ffmpeg.output(video, temp_output)
    ffmpeg.run(output, overwrite_output=True)
    
    os.replace(temp_output, video_path)

def add_background_music(video_path: str, music_path: str):
    logger.info(f"Adding background music to {video_path}")
    temp_output = f"{video_path}.with_music.mp4"
    (
        ffmpeg
        .input(video_path)
        .audio
        .filter('volume', volume=0.5)
        .output(temp_output, audio=ffmpeg.input(music_path).audio.filter('volume', volume=0.5))
        .overwrite_output()
        .run(capture_stdout=True, capture_stderr=True)
    )
    os.replace(temp_output, video_path)

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
    ffmpeg.run(stream, overwrite_output=True)

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
    background_music: UploadFile = File(None),
):
    if not file.filename.lower().endswith(('.mp4')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    file_uuid = uuid.uuid4()
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{file_uuid}{file_extension}"
    
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    
    bg_music_path = None
    if background_music:
        bg_music_extension = os.path.splitext(background_music.filename)[1]
        bg_music_filename = f"{file_uuid}_bg{bg_music_extension}"
        bg_music_path = os.path.join(UPLOAD_DIR, bg_music_filename)
        with open(bg_music_path, "wb") as buffer:
            buffer.write(await background_music.read())
    
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
    
    background_tasks.add_task(process_video, db, db_video.id, background_tasks)
    
    return {"task_id": db_video.id, "status": db_video.status}

@app.get("/videos")
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

@app.get("/video/{video_id}/status")
def get_video_status(video_id: int, db: Session = Depends(get_db)):
    video = db.query(Video).filter(Video.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {
        "id": video.id,
        "status": video.status,
        "current_stage": video.current_stage,
        "created_at": video.created_at,
        "updated_at": video.updated_at
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
                    logger.error(f'Failed to delete {file_path}. Reason: {e}')

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

    # Check system resources
    cpu_percent, memory_percent = get_resource_usage()
    health_status["checks"]["cpu_usage"] = f"{cpu_percent}%"
    health_status["checks"]["memory_usage"] = f"{memory_percent}%"

    if cpu_percent > MAX_CPU_PERCENT or memory_percent > MAX_MEMORY_PERCENT:
        health_status["status"] = "warning"

    return health_status

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)