#!/usr/bin/env python3
"""
Test script to verify word order fix for VTT generation
"""

def create_vtt_preserving_order(segments, max_chars_per_line=42, max_lines=2):
    """
    Create VTT with word order preservation by splitting long segments
    """
    if not segments:
        return "WEBVTT\n\n"
    
    vtt_lines = ["WEBVTT", ""]
    
    for segment in sorted(segments, key=lambda x: x.get("start", 0)):
        text = segment.get("text", "").strip()
        if not text:
            continue
        
        start_time = segment.get("start", 0.0)
        end_time = segment.get("end", start_time + 1.25)
        
        # Split text into words
        words = text.replace('\n', ' ').split()
        if not words:
            continue
        
        # Calculate how many subtitles we need for all words
        # Each subtitle can have max 2 lines of max_chars_per_line characters
        max_chars_per_subtitle = max_chars_per_line * max_lines
        
        # Create sub-segments that fit within constraints
        sub_segments = []
        current_words = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            
            # Check if adding this word would exceed subtitle capacity
            # Account for spaces between words
            space_needed = 1 if current_words else 0
            total_with_word = current_length + space_needed + word_len
            
            # Rough estimate of whether this will fit in 2 lines
            if current_words and total_with_word > max_chars_per_subtitle * 0.9:  # 90% capacity
                # Create a sub-segment with current words
                sub_segments.append(current_words)
                current_words = [word]
                current_length = word_len
            else:
                current_words.append(word)
                current_length += space_needed + word_len
        
        # Add remaining words
        if current_words:
            sub_segments.append(current_words)
        
        # Generate VTT entries for each sub-segment
        if len(sub_segments) == 1:
            # Original segment fits, process normally
            words = sub_segments[0]
            
            # Format into lines
            lines = []
            current_line = ""
            
            for word in words:
                test_line = f"{current_line} {word}" if current_line else word
                
                if len(test_line) <= max_chars_per_line:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            # Limit to max_lines
            lines = lines[:max_lines]
            
            formatted_text = '\n'.join(lines)
            
            # Position based on line count
            line_count = len(lines)
            position = {
                1: "align:middle line:90%",
                2: "align:middle line:84%"
            }.get(line_count, "align:middle line:80%")
            
            # Convert times
            start_str = format_time(start_time)
            end_str = format_time(end_time)
            
            vtt_lines.extend([
                f"{start_str} --> {end_str} {position}",
                formatted_text,
                ""
            ])
        else:
            # Split into multiple subtitle entries
            segment_duration = end_time - start_time
            sub_duration = segment_duration / len(sub_segments)
            
            for i, word_group in enumerate(sub_segments):
                sub_start = start_time + (i * sub_duration)
                sub_end = sub_start + sub_duration
                
                # Format words into lines
                lines = []
                current_line = ""
                
                for word in word_group:
                    test_line = f"{current_line} {word}" if current_line else word
                    
                    if len(test_line) <= max_chars_per_line:
                        current_line = test_line
                    else:
                        if current_line:
                            lines.append(current_line)
                        current_line = word
                
                if current_line:
                    lines.append(current_line)
                
                # Limit to max_lines
                lines = lines[:max_lines]
                
                formatted_text = '\n'.join(lines)
                
                # Position based on line count
                line_count = len(lines)
                position = {
                    1: "align:middle line:90%",
                    2: "align:middle line:84%"
                }.get(line_count, "align:middle line:80%")
                
                # Convert times
                start_str = format_time(sub_start)
                end_str = format_time(sub_end)
                
                vtt_lines.extend([
                    f"{start_str} --> {end_str} {position}",
                    formatted_text,
                    ""
                ])
    
    return "\n".join(vtt_lines)


def format_time(seconds):
    """Convert seconds to VTT time format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# Test the problematic segment
test_segment = {
    "start": 221.060,
    "end": 225.780,
    "text": "This gives a more accurate result, but it is more resource intensive and it can lead to some ghosting. Additionally, this algorithm cannot be used on transparent materials as those will always use"
}

print("Testing word order preservation...")
print()

# Show the problem
words = test_segment["text"].split()
print(f"Total words: {len(words)}")
print(f"Text: {test_segment['text']}")
print()

# Test with old approach (truncating)
print("OLD APPROACH (truncating):")
lines = []
current_line = ""
for word in words:
    test_line = f"{current_line} {word}" if current_line else word
    if len(test_line) <= 42:
        current_line = test_line
    else:
        if current_line:
            lines.append(current_line)
        current_line = word
if current_line:
    lines.append(current_line)

# Show what happens with truncation
print(f"Lines created: {len(lines)}")
for i, line in enumerate(lines[:2]):  # Only first 2 lines
    print(f"Line {i+1}: {line}")
print()

# Test new approach
print("NEW APPROACH (splitting):")
result = create_vtt_preserving_order([test_segment])
print(result)
