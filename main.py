import os
import random
import shutil
import uuid
import secrets
from enum import Enum as PyEnum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks, Header
from fastapi.responses import FileResponse
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import ffmpeg
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
THUMBNAIL_DIR = "thumbnails"
INTRO_VIDEO = "INTRO.mp4"
OUTRO_VIDEO = "OUTRO.mp4"

for directory in [UPLOAD_DIR, OUTPUT_DIR, THUMBNAIL_DIR]:
    os.makedirs(directory, exist_ok=True)

# Thread pool for concurrent processing
executor = ThreadPoolExecutor(max_workers=4)

def get_random_music(exclude=None):
    music_folder = "musics"
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

def process_video(file_path: str, adjusted_path: str, output_path: str, thumbnail_path: str, db: Session, video_id: int, bg_music_path: str = None):
    try:
        db_video = db.query(Video).filter(Video.id == video_id).first()
        db_video.status = VideoStatus.PROCESSING
        db.commit()

        validate_video(file_path)
        adjust_aspect_ratio(file_path, adjusted_path)
        combine_videos(adjusted_path, output_path, bg_music_path)
        generate_thumbnail(file_path, thumbnail_path)

        db_video.status = VideoStatus.COMPLETED
        db_video.thumbnail = os.path.basename(thumbnail_path)
        db.commit()
    except Exception as e:
        db_video.status = VideoStatus.FAILED
        db.commit()
        raise
    finally:
        for path in [file_path, adjusted_path, bg_music_path]:
            if path and os.path.exists(path):
                os.remove(path)

def combine_videos(main_video: str, output_path: str, custom_bg_music: str = None):
    intro_file = INTRO_VIDEO
    main_file = main_video
    outro_file = OUTRO_VIDEO
    watermark_file = "WATERMARK.png"

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
    adjusted_path = os.path.join(UPLOAD_DIR, f"adjusted_{safe_filename}")
    output_path = os.path.join(OUTPUT_DIR, f"final_{safe_filename}")
    thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{file_uuid}.jpg")
    
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
    
    background_tasks.add_task(
        executor.submit,
        process_video,
        file_path,
        adjusted_path,
        output_path,
        thumbnail_path,
        db,
        db_video.id,
        bg_music_path
    )
    
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
                    print(f'Failed to delete {file_path}. Reason: {e}')

        return {"message": "Database reset successful. All video records and files have been deleted."}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"An error occurred while resetting the database: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # You can add more checks here, e.g., database connection
        return {"status": "healthy", "message": "Service is running"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Service is unhealthy: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)