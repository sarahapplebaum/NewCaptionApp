#!/bin/bash

# Batch process all MP4 files in the specified directory
VIDEO_DIR="/Users/sarah.applebaum/Documents/FinalRenders/HDRP Lighting Fundamentals/Camtasia/mp4"
CAPTIONER_SCRIPT="/Users/sarah.applebaum/Desktop/NewCaptionApp/captioner_compact.py"

echo "🎬 Batch Video Caption Processing"
echo "================================="
echo "Directory: $VIDEO_DIR"
echo "Model: small"
echo ""

# Count total MP4 files
TOTAL=$(find "$VIDEO_DIR" -name "*.mp4" -type f | wc -l | tr -d ' ')
CURRENT=0

# Process each MP4 file
for video in "$VIDEO_DIR"/*.mp4; do
    if [ -f "$video" ]; then
        CURRENT=$((CURRENT + 1))
        BASENAME=$(basename "$video")
        
        echo "[$CURRENT/$TOTAL] Processing: $BASENAME"
        echo "-----------------------------------"
        
        # Run captioner.py with the specified parameters
        python "$CAPTIONER_SCRIPT" "$video" -o "$VIDEO_DIR" -m small
        
        # Check if successful
        if [ $? -eq 0 ]; then
            echo "✅ Success: $BASENAME"
        else
            echo "❌ Failed: $BASENAME"
        fi
        
        echo ""
    fi
done

echo "================================="
echo "🏁 Batch processing complete!"
echo "Processed $CURRENT videos"
