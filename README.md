# Aquaverse Video Generator - Localhost Deployment

🎬 **Video generation web application for localhost deployment**

## 📋 Requirements

- Python 3.8+
- FFmpeg installed on system
- Internet connection for downloading video clips

## 🚀 Quick Setup

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg
**Windows:**
- Download from https://ffmpeg.org/download.html
- Add to PATH environment variable

**macOS:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt install ffmpeg  # Ubuntu/Debian
sudo yum install ffmpeg  # CentOS/RHEL
```

### 3. Run Application
```bash
python web_app.py
```

### 4. Access Application
Open browser and go to: http://localhost:5000

## 📁 File Structure
```
genvideo2.5/
├── web_app.py              # Main Flask web application
├── generate_edit.py        # Video generation engine
├── ffmpeg_utils.py         # Video processing utilities
├── prompt_rules.py         # Prompt parsing logic
├── clip_selector.py        # Clip selection algorithms
├── downloader.py           # Video download utilities
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
└── canto_clip_tags_with_urls.csv  # Sample video database
```

## 🎯 Usage

1. Enter a video prompt (e.g., "Action video 15s in Jumanji zone")
2. Click "Generate My Video"
3. Wait for processing to complete
4. Download both 16:9 (landscape) and 9:16 (vertical) versions

## ⚠️ Important Notes

- First run may take longer as it downloads required assets
- Ensure stable internet connection for video downloads
- Generated videos are saved in `output/` directory
- Sample data provided - replace with real video URLs for production

## 🔧 Troubleshooting

**Common Issues:**
- **FFmpeg not found:** Make sure FFmpeg is installed and in PATH
- **Module not found:** Run `pip install -r requirements.txt`
- **Port in use:** Change port in web_app.py or stop other processes

**Support:**
If you encounter issues, check the console output for detailed error messages.