#!/usr/bin/env python3
"""
Test the captioner script with multiple MP4 videos from target folder
Verify all tests pass consistently across different videos
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
import time

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from test_real_video import RealVideoTest


def find_mp4_files(folder_path: str, limit: int = 5) -> List[str]:
    """Find MP4 files in the specified folder"""
    mp4_files = []
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"❌ Folder not found: {folder_path}")
        return []
    
    # Look for MP4 files
    for file_path in folder.rglob("*.mp4"):
        mp4_files.append(str(file_path))
        if len(mp4_files) >= limit:
            break
    
    return mp4_files


def test_multiple_videos(video_paths: List[str]) -> Dict[str, bool]:
    """Test multiple videos and collect results"""
    results = {}
    
    print("🎬 TESTING MULTIPLE VIDEOS")
    print("=" * 80)
    print(f"📊 Found {len(video_paths)} videos to test")
    print()
    
    for i, video_path in enumerate(video_paths, 1):
        print(f"\n{'='*80}")
        print(f"📹 VIDEO {i}/{len(video_paths)}: {Path(video_path).name}")
        print(f"{'='*80}")
        
        try:
            test = RealVideoTest(video_path)
            success = test.run_full_test()
            results[video_path] = success
            
            # Add a small delay between videos to avoid overwhelming the system
            if i < len(video_paths):
                print("\n⏳ Waiting 2 seconds before next video...")
                time.sleep(2)
                
        except Exception as e:
            print(f"💥 Error testing video: {e}")
            results[video_path] = False
    
    return results


def print_summary(results: Dict[str, bool]):
    """Print summary of all test results"""
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY - ALL VIDEOS")
    print("=" * 80)
    
    passed = sum(1 for success in results.values() if success)
    total = len(results)
    
    for video_path, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {Path(video_path).name}")
    
    print(f"\n🎯 OVERALL: {passed}/{total} videos passed all tests ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL VIDEOS PASSED! The captioner is working correctly.")
    else:
        print("⚠️  Some videos failed. Please review the individual test results above.")


def main():
    """Main test function"""
    # Target folder
    folder_path = "/Users/sarah.applebaum/Documents/FinalRenders/HDRP Lighting Fundamentals/Camtasia/mp4"
    
    print("🔍 Searching for MP4 files...")
    video_paths = find_mp4_files(folder_path, limit=5)
    
    if not video_paths:
        print("❌ No MP4 files found in the specified folder")
        return 1
    
    # Test all videos
    results = test_multiple_videos(video_paths)
    
    # Print summary
    print_summary(results)
    
    # Return success if all videos passed
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
