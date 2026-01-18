"""
Web application for video generation - Simple Version
Allows users to input prompts and download generated videos
"""
import os
import threading
import uuid
import time
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
import shutil
import logging

# Import the main function directly
from generate_edit import main as generate_video

app = Flask(__name__)
app.config['SECRET_KEY'] = 'aquaverse-video-generator-2026'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Disable Flask/Werkzeug access logs
log = logging.getLogger('werkzeug')
log.disabled = True
app.logger.disabled = True

# Job storage (in production, use database)
jobs_db = {}
jobs_lock = threading.Lock()

OUTPUT_DIR = Path("output")
JOBS_DIR = OUTPUT_DIR / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_old_jobs():
    """Clean up jobs older than 24 hours"""
    with jobs_lock:
        current_time = time.time()
        to_delete = []
        for job_id, job in jobs_db.items():
            created = datetime.fromisoformat(job['created_at']).timestamp()
            if current_time - created > 86400:  # 24 hours
                to_delete.append(job_id)
                # Delete job files
                job_dir = JOBS_DIR / job_id
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
        
        for job_id in to_delete:
            del jobs_db[job_id]

def process_video_generation(job_id, prompt):
    """Background task to generate video"""
    try:
        with jobs_lock:
            jobs_db[job_id]['status'] = 'processing'
            jobs_db[job_id]['message'] = 'Processing: Downloading and cutting clips...'
            jobs_db[job_id]['progress'] = 20
        
        # Generate video
        generate_video(prompt)
        
        with jobs_lock:
            jobs_db[job_id]['message'] = 'Copying files...'
            jobs_db[job_id]['progress'] = 90
        
        # Copy outputs to job-specific directory
        job_dir = JOBS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        output_169 = OUTPUT_DIR / "final_16x9_with_outro.mp4"
        output_916 = OUTPUT_DIR / "final_9x16_with_outro.mp4"
        
        if not output_169.exists() or not output_916.exists():
            raise Exception("Generated video files not found. Please check your prompt.")
        
        # Copy to job directory
        dest_169 = job_dir / "video_16-9.mp4"
        dest_916 = job_dir / "video_9-16.mp4"
        
        shutil.copy2(output_169, dest_169)
        shutil.copy2(output_916, dest_916)
        
        with jobs_lock:
            jobs_db[job_id]['status'] = 'completed'
            jobs_db[job_id]['message'] = 'Video created successfully!'
            jobs_db[job_id]['progress'] = 100
            jobs_db[job_id]['files'] = {
                '169': str(dest_169),
                '916': str(dest_916)
            }
            jobs_db[job_id]['completed_at'] = datetime.now().isoformat()
        
    except Exception as e:
        with jobs_lock:
            jobs_db[job_id]['status'] = 'failed'
            jobs_db[job_id]['message'] = f'Error: {str(e)}'
            jobs_db[job_id]['progress'] = 0
        print(f"Error in job {job_id}: {e}")
        import traceback
        traceback.print_exc()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def api_generate():
    """Start video generation"""
    data = request.get_json()
    prompt = data.get('prompt', '').strip()
    
    if not prompt:
        return jsonify({'error': 'Please enter a prompt'}), 400
    
    if len(prompt) > 500:
        return jsonify({'error': 'Prompt too long (maximum 500 characters)'}), 400
    
    # Create new job
    job_id = str(uuid.uuid4())
    
    with jobs_lock:
        jobs_db[job_id] = {
            'id': job_id,
            'prompt': prompt,
            'status': 'queued',
            'message': 'Job created, waiting to start...',
            'progress': 0,
            'created_at': datetime.now().isoformat()
        }
    
    # Start background thread
    thread = threading.Thread(
        target=process_video_generation,
        args=(job_id, prompt),
        daemon=True
    )
    thread.start()
    
    return jsonify({'job_id': job_id})

@app.route('/api/jobs')
def get_jobs():
    """Get all active jobs"""
    cleanup_old_jobs()
    with jobs_lock:
        return jsonify({
            'jobs': [
                {
                    'id': job_id,
                    'status': job.get('status', 'unknown'),
                    'message': job.get('message', ''),
                    'progress': job.get('progress', 0),
                    'created_at': job.get('created_at', ''),
                    'completed_at': job.get('completed_at'),
                }
                for job_id, job in jobs_db.items()
            ]
        })

@app.route('/api/status/<job_id>')
def api_status(job_id):
    """Get job status"""
    with jobs_lock:
        if job_id not in jobs_db:
            return jsonify({'error': 'Job not found'}), 404
        
        job = jobs_db[job_id].copy()
    
    response = {
        'id': job['id'],
        'status': job['status'],
        'message': job['message'],
        'progress': job['progress'],
        'prompt': job['prompt'],
        'created_at': job['created_at']
    }
    
    if job['status'] == 'completed':
        response['download_urls'] = {
            '169': f'/api/download/{job_id}/169',
            '916': f'/api/download/{job_id}/916'
        }
        response['completed_at'] = job.get('completed_at')
    
    return jsonify(response)

@app.route('/api/download/<job_id>/<aspect>')
def api_download(job_id, aspect):
    """Download generated video"""
    with jobs_lock:
        if job_id not in jobs_db:
            return jsonify({'error': 'Job not found'}), 404
        
        if jobs_db[job_id]['status'] != 'completed':
            return jsonify({'error': 'Job not completed yet'}), 400
        
        files = jobs_db[job_id].get('files', {})
    
    if aspect not in files:
        return jsonify({'error': 'Video file not found'}), 404
    
    file_path = Path(files[aspect])
    if not file_path.exists():
        return jsonify({'error': 'Video file not found on disk'}), 404
    
    # Generate filename
    aspect_name = "vertical" if aspect == "916" else "landscape"
    filename = f"aquaverse_video_{aspect_name}_{job_id[:8]}.mp4"
    
    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype='video/mp4'
    )

def periodic_cleanup():
    """Periodic cleanup of old jobs"""
    while True:
        time.sleep(3600)  # Run every hour
        cleanup_old_jobs()

# Start cleanup thread
cleanup_thread = threading.Thread(target=periodic_cleanup, daemon=True)
cleanup_thread.start()

if __name__ == '__main__':
    # Get port from environment (for Railway/Render) or default to 5000
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "="*60)
    print("🎬 Aquaverse Video Generator")
    print("="*60)
    print(f"Server starting at: http://localhost:{port}")
    print(f"Jobs directory: {JOBS_DIR.absolute()}")
    print("="*60 + "\n")
    
    # Disable werkzeug logs completely
    import werkzeug
    werkzeug._internal._log = lambda *a, **kw: None
    
    app.run(
        debug=os.environ.get('DEBUG', 'False').lower() == 'true',
        host='0.0.0.0',
        port=port,
        threaded=True,
        use_reloader=False
    )