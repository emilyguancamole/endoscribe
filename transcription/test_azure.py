#!/usr/bin/env python3
"""
Test script for Azure Speech Service integration.
This script helps validate Azure setup and compare results with WhisperX.
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transcription.transcription_service import transcribe_unified
from dotenv import load_dotenv

load_dotenv()

def check_azure_credentials():
    azure_key = os.getenv("AZURE_SPEECH_KEY")
    if not azure_key:
        print("AZURE_SPEECH_KEY not found in environment")
        return False

def test_azure_transcription(audio_file: str):
    """Test Azure Speech Service transcription."""
    print("\n" + "="*80)
    print("TESTING AZURE SPEECH SERVICE")
    print("="*80)
    
    try:
        result = transcribe_unified(
            audio_file=audio_file,
            service="azure",
            language="en-US"
        )
        print(f"\nSuccessful Azure transcription!")
        print(f"   Service: {result['service']}")
        print(f"   Duration: {result.get('duration', 'N/A'):.2f}s")
        print(f"   Segments: {len(result.get('segments', []))}")
        print(f"\n   Transcript preview:")
        print(f"   {result['text'][:200]}...")
        
        return result
        
    except Exception as e:
        print(f"\nAzure transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_whisperx_transcription(audio_file: str):
    """Test WhisperX transcription (if available)."""
    print("\n" + "="*80)
    print("TESTING WHISPERX (for comparison)")
    print("="*80)
    
    try:
        result = transcribe_unified(
            audio_file=audio_file,
            service="whisperx",
            whisper_model="large-v3"
        )
        print(f"\nWhisperX transcription successful!")
        print(f"   Service: {result['service']}")
        print(f"   Language: {result.get('language', 'N/A')}")
        print(f"   Duration: {result.get('duration', 'N/A'):.2f}s")
        print(f"   Segments: {len(result.get('segments', []))}")
        print(f"\n   Transcript preview (first 200 chars):")
        print(f"   {result['text'][:200]}...")
        
        return result
        
    except Exception as e:
        print(f"\n⚠️  WhisperX not available or failed: {e}")
        return None


def compare_results(azure_result, whisperx_result):
    """Compare Azure and WhisperX results."""
    if not azure_result or not whisperx_result:
        return
    
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    
    azure_text = azure_result['text']
    whisperx_text = whisperx_result['text']
    
    print(f"\nAzure length: {len(azure_text)} chars")
    print(f"WhisperX length: {len(whisperx_text)} chars")
    
    # Simple similarity check
    azure_words = set(azure_text.lower().split())
    whisperx_words = set(whisperx_text.lower().split())
    
    common = azure_words & whisperx_words
    similarity = len(common) / max(len(azure_words), len(whisperx_words)) * 100
    
    print(f"\nWord overlap: {similarity:.1f}%")
    
    if similarity > 80:
        print("✅ Results are very similar!")
    elif similarity > 60:
        print("⚠️  Results have moderate similarity")
    else:
        print("❌ Results differ significantly")


def main():
    """Main test function."""
    print("="*80)
    print("AZURE SPEECH SERVICE TEST")
    print("="*80)
    
    # Check credentials
    if not check_azure_credentials():
        print("\n❌ Please configure Azure credentials before testing")
        return 1
    
    # Get test audio file
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        # Look for test audio in common locations
        test_locations = [
            "transcription/recordings",
            "pep_risk/recordings",
            "web_app/uploads",
        ]
        
        audio_file = None
        for location in test_locations:
            location_path = Path(__file__).parent.parent / location
            if location_path.exists():
                audio_files = list(location_path.glob("*.wav")) + list(location_path.glob("*.mp3"))
                if audio_files:
                    audio_file = str(audio_files[0])
                    break
        
        if not audio_file:
            print("\n❌ No test audio file found")
            print("\nUsage:")
            print(f"  python {Path(__file__).name} path/to/audio.wav")
            return 1
    
    if not Path(audio_file).exists():
        print(f"\n❌ Audio file not found: {audio_file}")
        return 1
    
    print(f"\nTest audio file: {audio_file}")
    
    # Test Azure
    azure_result = test_azure_transcription(audio_file)
    
    # Test WhisperX for comparison (optional)
    print("\n" + "-"*80)
    compare_whisperx = input("Would you like to compare with WhisperX? (y/N): ").lower().strip()
    
    if compare_whisperx == 'y':
        whisperx_result = test_whisperx_transcription(audio_file)
        compare_results(azure_result, whisperx_result)
    
    # Summary
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    
    if azure_result:
        print("\n✅ Azure Speech Service is working correctly!")
        print("\nNext steps:")
        print("1. Review the transcription quality")
        print("2. Adjust language settings if needed (AZURE_SPEECH_REGION)")
        print("3. Update your code to use transcribe_unified() or azure_transcribe()")
        print("4. Set TRANSCRIPTION_SERVICE=azure in .env to make it default")
        return 0
    else:
        print("\n❌ Azure Speech Service test failed")
        print("\nTroubleshooting:")
        print("1. Verify your AZURE_SPEECH_KEY is correct")
        print("2. Check that your Azure Speech Service is deployed in the correct region")
        print("3. Ensure your audio file format is supported (WAV, MP3, etc.)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
