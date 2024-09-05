from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Form, BackgroundTasks, Header
from fastapi.responses import FileResponse
import ffmpeg
import os
import uuid
import shutil
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Enum
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
from enum import Enum as PyEnum
from concurrent.futures import ThreadPoolExecutor
import secrets
from starlette.status import HTTP_403_FORBIDDEN
import random
from typing import List

# Database setup
SQLALCHEMY_DATABASE_URL = "postgresql://immortal_youtube_api_user:rxbFTa4Er7gIDR1Y4j4dTbrwP3c3QXfr@dpg-crd0t688fa8c73bfoqgg-a.oregon-postgres.render.com/immortal_youtube_api"
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

def get_random_music() -> str:
    music_folder = "musics"
    music_files = [f for f in os.listdir(music_folder) if f.endswith(('.mp3', '.wav'))]
    if not music_files:
        raise ValueError("No music files found in the 'musics' folder")
    return os.path.join(music_folder, random.choice(music_files))

def process_video(file_path: str, adjusted_path: str, output_path: str, thumbnail_path: str, db: Session, video_id: int):
    try:
        db_video = db.query(Video).filter(Video.id == video_id).first()
        db_video.status = VideoStatus.PROCESSING
        db.commit()

        info = validate_video(file_path)
        adjust_aspect_ratio(file_path, adjusted_path)
        
        # Apply fade-out effect to the adjusted video
        faded_path = os.path.join(UPLOAD_DIR, f"faded_{os.path.basename(adjusted_path)}")
        apply_fade_out(adjusted_path, faded_path)
        
        combine_videos(faded_path, output_path)
        generate_thumbnail(file_path, thumbnail_path)

        db_video.status = VideoStatus.COMPLETED
        db_video.thumbnail = os.path.basename(thumbnail_path)
        db.commit()
    except Exception as e:
        db_video.status = VideoStatus.FAILED
        db.commit()
        print(f"Error processing video: {str(e)}")
    finally:
        for path in [file_path, adjusted_path, faded_path]:
            if os.path.exists(path):
                os.remove(path)

def apply_fade_out(input_path: str, output_path: str, duration: float = 3.0):
    probe = ffmpeg.probe(input_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    total_duration = float(video_info['duration'])
    start_time = total_duration - duration

    # Create the fade filter
    video = ffmpeg.input(input_path)
    video = video.filter('fade', type='out', start_time=start_time, duration=duration)
    
    # Apply the filter and output
    output = ffmpeg.output(video, output_path)
    ffmpeg.run(output, overwrite_output=True)

def combine_videos(main_video: str, output_path: str):
    # Load the input video files
    intro_file = INTRO_VIDEO
    main_file = main_video
    outro_file = OUTRO_VIDEO
    watermark_file = "WATERMARK.png"

    # Input streams from video files
    intro = ffmpeg.input(intro_file)
    main = ffmpeg.input(main_file)
    outro = ffmpeg.input(outro_file)
    watermark = ffmpeg.input(watermark_file)

    # Concatenate video streams (video only, no audio)
    video = ffmpeg.concat(intro['v'], main['v'], outro['v'], v=1, a=0)

    # Split the watermark into two streams for different scaling
    watermark_split = watermark.filter_multi_output('split')
    watermark_top_left = watermark_split[0].filter('scale', w='iw*0.4', h='ih*0.4')
    watermark_bottom_right = watermark_split[1].filter('scale', w='iw*0.4', h='ih*0.4')

    # Add watermark to the top-left corner
    video = ffmpeg.filter(
        [video, watermark_top_left],
        'overlay',
        x=10,
        y=10,
        enable='between(t,0,30)',
        alpha=0.25  # Set the transparency to 25%
    )

    # Add another watermark to the bottom-right corner
    video = ffmpeg.filter(
        [video, watermark_bottom_right],
        'overlay',
        x='main_w-overlay_w-10',
        y='main_h-overlay_h-10',
        enable='between(t,0,30)',
        alpha=0.25  # Set the transparency to 25%
    )

    # Get random background music
    bg_music = ffmpeg.input(get_random_music())

    # Calculate total duration
    intro_duration = float(ffmpeg.probe(intro_file)['streams'][0]['duration'])
    main_duration = float(ffmpeg.probe(main_file)['streams'][0]['duration'])
    outro_duration = float(ffmpeg.probe(outro_file)['streams'][0]['duration'])
    total_duration = intro_duration + main_duration + outro_duration

    # Apply fade-in and fade-out to background music
    bg_music = (
        bg_music
        .filter('afade', type='in', duration=3)
        .filter('afade', type='out', start_time=total_duration - 3, duration=3)
        .filter('volume', volume=0.4)
        .filter('atrim', duration=total_duration)
    )

    # Concatenate audio streams, creating silent audio if an input does not have audio
    audio_inputs = []
    for input_file in [intro_file, main_file, outro_file]:
        # Probe to check if audio exists
        audio_streams = ffmpeg.probe(input_file)['streams']
        audio_stream = next((stream for stream in audio_streams if stream['codec_type'] == 'audio'), None)
        
        if audio_stream:
            # Use audio stream if present
            audio_inputs.append(ffmpeg.input(input_file)['a'])
        else:
            # Create a silent audio track for the duration of the video
            silent_duration = float(ffmpeg.probe(input_file)['streams'][0]['duration'])
            silent_audio = ffmpeg.input('anullsrc=r=44100:cl=stereo', f='lavfi', t=silent_duration)
            audio_inputs.append(silent_audio)

    # Concatenate audio streams
    audio = ffmpeg.concat(*audio_inputs, v=0, a=1)

    # Mix original audio with background music
    mixed_audio = ffmpeg.filter([audio, bg_music], 'amix', inputs=2)

    # Output final video with mixed audio
    output = ffmpeg.output(video, mixed_audio, output_path)
    ffmpeg.run(output, overwrite_output=True)

def get_video_info(file_path):
    probe = ffmpeg.probe(file_path)
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        raise ValueError(f"No video stream found in {file_path}")
    return {
        'width': int(video_stream['width']),
        'height': int(video_stream['height']),
        'duration': float(video_stream['duration']),
        'rotation': int(video_stream.get('tags', {}).get('rotate', '0'))
    }

def is_landscape(width, height, rotation):
    if rotation in [90, 270]:
        width, height = height, width
    return width > height

def validate_video(file_path):
    info = get_video_info(file_path)
    if not is_landscape(info['width'], info['height'], info['rotation']):
        raise ValueError("Video must be in landscape orientation")
    if min(info['width'], info['height']) < 720:
        raise ValueError("Video resolution must be at least 720p")
    return info

def adjust_aspect_ratio(input_path, output_path):
    probe = ffmpeg.probe(input_path)
    audio_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'audio'), None)
    
    input_stream = ffmpeg.input(input_path)
    video = input_stream.video.filter('scale', w='iw*min(1920/iw,1080/ih)', h='ih*min(1920/iw,1080/ih)')
    video = video.filter('crop', w='1920', h='1080')
    
    if audio_stream:
        audio = input_stream.audio
        stream = ffmpeg.output(video, audio, output_path)
    else:
        stream = ffmpeg.output(video, output_path)
    
    ffmpeg.run(stream, overwrite_output=True)

def generate_thumbnail(input_path, output_path):
    probe = ffmpeg.probe(input_path)
    duration = float(probe['streams'][0]['duration'])
    mid_time = duration / 2

    (
        ffmpeg
        .input(input_path, ss=mid_time)
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
    map_name: str = Form(...)
):
    if not file.filename.lower().endswith(('.mp4', '.mov', '.avi')):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a video file.")
    
    file_uuid = uuid.uuid4()
    file_extension = os.path.splitext(file.filename)[1]
    safe_filename = f"{file_uuid}{file_extension}"
    
    file_path = os.path.join(UPLOAD_DIR, safe_filename)
    adjusted_path = os.path.join(UPLOAD_DIR, f"adjusted_{safe_filename}")
    output_path = os.path.join(OUTPUT_DIR, f"final_{safe_filename}")
    thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{file_uuid}.jpg")
    
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
        shutil.copyfileobj(file.file, buffer)
    
    background_tasks.add_task(
        executor.submit,
        process_video,
        file_path,
        adjusted_path,
        output_path,
        thumbnail_path,
        db,
        db_video.id
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
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    if video.thumbnail is None:
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
    
    if video.status != VideoStatus.COMPLETED and video.status != VideoStatus.APPROVED:
        raise HTTPException(status_code=400, detail="Video is not ready for download")
    
    video_path = os.path.join(OUTPUT_DIR, f"final_{video.filename}")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(video_path, media_type="video/mp4", filename=f"processed_{video.filename}")

@app.post("/admin/reset-database")
async def reset_database(db: Session = Depends(get_db), admin_password: str = Header(...)):
    if not secrets.compare_digest(admin_password, '1nf0rmM@tic$'):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid admin password"
        )
    
    try:
        # Delete all rows from the videos table
        db.query(Video).delete()
        db.commit()

        # Optionally, clear the file storage
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
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)