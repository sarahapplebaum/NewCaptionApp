#!/usr/bin/env python3
"""
Test script to verify the standalone app build
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def test_ffmpeg_in_bundle():
    """Check if FFmpeg is bundled with the app"""
    print("Checking for bundled FFmpeg...")
    
    # Check if ffmpeg exists in the dist directory
    dist_dir = Path("dist")
    ffmpeg_path = dist_dir / "ffmpeg"
    
    if ffmpeg_path.exists():
        print(f"✅ FFmpeg found in bundle: {ffmpeg_path}")
        return True
    else:
        print("⚠️  FFmpeg not bundled - app will require system FFmpeg")
        return False

def test_app_launch():
    """Test if the app can launch"""
    print("\nTesting app launch...")
    
    app_path = Path("dist/VideoCaptioner")
    if not app_path.exists():
        print("❌ VideoCaptioner executable not found")
        return False
    
    try:
        # Test with --help to see if it launches
        result = subprocess.run(
            [str(app_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ App launches successfully")
            print("Help output preview:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
            return True
        else:
            print(f"❌ App failed to launch: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ App launch timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing app: {e}")
        return False

def test_cli_mode():
    """Test CLI mode with a dummy video"""
    print("\nTesting CLI mode...")
    
    # Create a test video using FFmpeg (if available)
    ffmpeg_cmd = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True)
    if ffmpeg_cmd.returncode != 0:
        print("⚠️  FFmpeg not available for creating test video")
        return False
    
    # Create a simple test video
    with tempfile.TemporaryDirectory() as temp_dir:
        test_video = Path(temp_dir) / "test.mp4"
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()
        
        # Create a 5-second test video
        create_video_cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=5",
            "-c:v", "libx264", "-c:a", "aac", "-y", str(test_video)
        ]
        
        print("Creating test video...")
        result = subprocess.run(create_video_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Failed to create test video")
            return False
        
        print(f"✅ Test video created: {test_video}")
        
        # Test the app with the video
        app_path = Path("dist/VideoCaptioner")
        test_cmd = [
            str(app_path), str(test_video),
            "-o", str(output_dir),
            "-m", "tiny",  # Use tiny model for quick test
            "--no-timestamps"  # Just test transcript
        ]
        
        print("\nRunning app with test video...")
        print(f"Command: {' '.join(test_cmd)}")
        
        try:
            result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("✅ CLI mode test successful")
                
                # Check if output was created
                txt_file = output_dir / "test.txt"
                if txt_file.exists():
                    print(f"✅ Output file created: {txt_file}")
                else:
                    print("⚠️  Output file not found")
                    
                return True
            else:
                print(f"❌ CLI mode failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ CLI mode test timed out")
            return False
        except Exception as e:
            print(f"❌ Error in CLI test: {e}")
            return False

def check_app_size():
    """Check the size of the built app"""
    print("\nChecking app size...")
    
    app_path = Path("dist/VideoCaptioner")
    if app_path.exists():
        size_mb = app_path.stat().st_size / (1024 * 1024)
        print(f"📦 App size: {size_mb:.1f} MB")
        
        if size_mb > 500:
            print("⚠️  App is quite large - consider optimization")
        else:
            print("✅ App size is reasonable")
    
    # Check .app bundle too
    app_bundle = Path("dist/VideoCaptioner.app")
    if app_bundle.exists():
        # Get total size of app bundle
        total_size = sum(f.stat().st_size for f in app_bundle.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        print(f"📦 App bundle size: {size_mb:.1f} MB")

def main():
    """Run all tests"""
    print("🧪 Testing standalone Video Captioner build")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Run tests
    tests = [
        ("FFmpeg bundling", test_ffmpeg_in_bundle),
        ("App launch", test_app_launch),
        ("CLI mode", test_cli_mode),
        ("App size", check_app_size),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n### {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Summary:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {test_name}: {status}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The app is ready for distribution.")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")

if __name__ == "__main__":
    main()
