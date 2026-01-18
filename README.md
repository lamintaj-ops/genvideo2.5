# Aquaverse Video Generator - Localhost Deployment

🎬 **Video generation web application for localhost deployment**

## 📋 Requirements

- Python 3.8+
- FFmpeg installed on system
- Internet connection for downloading video clips

## 🚀 Quick Setup

### 1. Install Python Dependencies
```bash
# Option 1: Using requirements.txt (recommended)
pip install -r requirements.txt

# Option 2: Manual installation (if requirements.txt fails)
pip install --upgrade pip setuptools wheel
pip install flask pandas tqdm requests pathlib
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
├── lut/
│   └── aquaverse_fun.cube # Color grading LUT
└── canto_clip_tags_with_urls.csv  # Sample video database
```

## 🎯 Usage

1. Enter a video prompt (e.g., "Action video 15s in Jumanji zone, start with wide shot of slide tower, then fast sliding down, big water splash close-up")
2. Click "Generate My Video"
3. Wait for processing to complete (30-60 seconds)
4. Download both 16:9 (landscape) and 9:16 (vertical) versions

## ⚠️ Important Notes

- First run may take longer as it downloads required assets
- Ensure stable internet connection for video downloads
- Generated videos are saved in `output/` directory
- Sample data provided - replace with real video URLs for production
- The web app uses a simple, stable version without complex progress tracking

## 🔧 Troubleshooting

**Common Issues:**

1. **FFmpeg not found:**
   - Make sure FFmpeg is installed and in PATH
   - Test with: `ffmpeg -version`

2. **Module not found errors:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install flask pandas tqdm requests pathlib
   ```

3. **Port in use:**
   - Change port in web_app.py (default: 5000)
   - Or stop other processes using port 5000

4. **numpy installation failed:**
   - Use: `pip install --upgrade pip setuptools wheel`
   - Then: `pip install pandas` (includes numpy)

5. **Video generation errors:**
   - Check internet connection
   - Ensure CSV file exists and has proper format
   - Check console output for detailed error messages

## 🚩 Features

- ✅ Simple web interface
- ✅ Text-to-video generation
- ✅ Dual format export (16:9 and 9:16)
- ✅ Background music integration
- ✅ Color grading and effects
- ✅ No complex dependencies
- ✅ Localhost deployment ready

## 📞 Support

If you encounter issues:
1. Check the terminal/console output for detailed error messages
2. Ensure all dependencies are installed correctly
3. Verify FFmpeg installation and PATH configuration
4. Make sure you have a stable internet connection

**Known Working Configuration:**
- Python 3.8-3.12
- Flask 3.1+
- pandas 2.3+
- FFmpeg latest version

For additional help, check the console logs when running the application.