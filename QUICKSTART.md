# Video Captioner - Quick Start Guide

## 🚀 Getting Started (First Time)

### Prerequisites
- ✅ Windows 10 or 11
- ✅ Visual C++ Redistributable installed ([Download here](https://aka.ms/vs/17/release/vc_redist.x64.exe))
- ✅ Internet connection

### Step 1: Download the Project

1. Go to: https://github.com/sarahapplebaum/NewCaptionApp
2. Click the green **"Code"** button
3. Click **"Download ZIP"**
4. Extract the ZIP file to a folder (e.g., `C:\VideoCaptioner\`)

### Step 2: Run the Setup

1. Open the extracted folder
2. **Double-click `START_HERE.bat`**
3. Wait for automatic setup (5-10 minutes first time)
   - Python will be installed if needed
   - All dependencies will be downloaded
   - FFmpeg will be downloaded
4. The application will launch automatically!

**That's it!** The script handles everything for you.

---

## 📝 Using the Application

### First Time Launch
- Double-click `START_HERE.bat`
- Wait for setup to complete
- Application opens automatically

### Subsequent Launches
- Double-click **`run.bat`** (much faster!)
- Application opens in seconds

---

## 🎬 How to Use Video Captioner

1. **Select Files**
   - Click "📁 Select Files"
   - Choose your video files (.mp4, .mov, .avi, etc.)

2. **Choose Output Folder**
   - Click "📋 Output Folder"
   - Select where to save captions

3. **Configure Settings** (optional)
   - **AI Model**: `small` (recommended) or others for accuracy
   - **Max chars/line**: Leave at 42
   - **Generate timestamps**: Leave checked

4. **Start Processing**
   - Click "🚀 Start Processing"
   - Wait for transcription to complete
   - Files are saved automatically!

### Output Files
- **`.txt`** - Plain text transcript
- **`.vtt`** - Web video subtitles (for HTML5 video)
- **`.srt`** - Standard subtitle format (for most video players)

---

## 🔧 Troubleshooting

### "Python not found" Error
**Solution 1:** Run `START_HERE.bat` - it installs Python automatically

**Solution 2:** Install manually:
1. Download Python 3.11 from https://www.python.org/downloads/
2. During installation, check ☑️ "Add Python to PATH"
3. Run `START_HERE.bat` again

### "DLL Load Failed" Error
**Install Visual C++ Redistributable:**
1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
2. Run the installer
3. Restart your computer
4. Run `START_HERE.bat` again

### Application Won't Start
**Check the log file:**
1. Press `Windows + R`
2. Type: `%TEMP%`
3. Look for `videocaptioner_debug.log`
4. Open it in Notepad to see error details

### "FFmpeg not found" Warning
**Install FFmpeg:**
```
winget install ffmpeg
```
Or download from: https://ffmpeg.org/download.html

---

## 💡 Tips

### For Faster Processing
- Use **"small"** model (good balance of speed/accuracy)
- Use **"tiny"** or **"base"** for very fast results
- Use **"large-v3"** for maximum accuracy (slower)

### For Better Accuracy
- Add a **Context Prompt** (e.g., "This video is about Unity game development")
- Use the **Vocabulary Correction** feature with a CSV file of technical terms

### For Technical Videos
1. Enable "Vocabulary Correction"
2. Click "📂 Select CSV"
3. Choose `Official English Term List - English Term List.csv` (included)
4. This fixes capitalization of technical terms (e.g., "rigidbody" → "Rigidbody")

---

## 📂 File Structure

After setup, your folder will look like this:

```
VideoCaptioner/
├── START_HERE.bat          ← Run this first time
├── run.bat                 ← Run this after setup
├── captioner_compact.py    ← Main application
├── requirements.txt        ← Dependency list
├── ffmpeg.exe             ← Downloaded automatically
├── venv/                  ← Python environment (created)
├── Official English Term List.csv  ← Vocabulary file
└── ... (other project files)
```

**Don't delete the `venv/` folder** - it contains all installed packages!

---

## ❓ Need Help?

1. Check the log file: `%TEMP%\videocaptioner_debug.log`
2. Make sure Visual C++ Redistributable is installed
3. Try running `START_HERE.bat` again
4. Contact Sarah for assistance

---

## 🎉 You're All Set!

Just double-click `run.bat` to use Video Captioner anytime!

**Happy Captioning! 🎬✨**
