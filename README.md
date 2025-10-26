# AudioVisualDisplayApp

A powerful Qt-based application for real-time audio spectrogram visualization that enables users to correlate visual events in video with their corresponding audio features. This application displays synchronized audio spectrograms (waterfall plots) alongside video frames, allowing users to see what visual events cause specific audio characteristics.

<div align="center">
  <img src="demoRecord.gif" width="50%"/>
</div>

<div align="center">
  <img src="demoDrone.gif" width="50%"/>
</div>


## Overview

The AudioVisualDisplayApp provides an interface for analyzing audio-visual relationships by displaying real-time spectrograms synchronized with video playback or live audio recording. This is particularly useful for:

- **Audio-visual synchronization analysis**: Understanding which visual events produce specific audio frequencies
- **Drone and environmental audio analysis**: Correlating visual motion with acoustic signatures
- **Educational demonstrations**: Teaching the relationship between visual events and their acoustic properties
- **Real-time audio monitoring**: Live visualization of audio spectrum from microphone input

## Key Features

### 🎥 Video File Processing
- Load and process MP4 video files
- Extract and analyze audio tracks from video
- Synchronized frame-by-frame video playback with audio spectrograms
- Choose to show or hide video frames for faster processing

### 🎤 Real-time Audio Recording
- Live microphone input with real-time spectrogram display
- Configurable audio device selection
- Real-time waterfall plot visualization

### 📊 Advanced Spectrogram Visualization
- **Waterfall Plot**: Time vs. Frequency visualization with configurable color mapping
- **Kaiser Window Filtering**: Adjustable beta parameter (0-20) for spectral analysis quality
- **Configurable Frequency Range**: Display up to 20 kHz (customizable based on Nyquist frequency)
- **Dynamic Color Scaling**: Automatic noise floor detection and dynamic range adjustment
- **Noise Floor Control**: Adjustable percentile-based color scaling

### ⚙️ Customizable STFT Parameters
- Window size: 1024 samples (fixed for optimal performance)
- Kaiser window beta parameter: 0.0 to 20.0
- Noise floor percentile: 0% to 99%
- Maximum frequency: 0.1 to 48 kHz (limited by Nyquist frequency)

### 🎨 User Interface
- **Left Control Panel**: Audio device selection, STFT parameters, and playback controls
- **Center Waterfall Display**: Real-time or file-based spectrogram visualization
- **Right Video Frame**: Synchronized video frame display (optional)
- **Progress Tracking**: Real-time progress bar and time display
- **Status Indicators**: Visual feedback for recording and processing states

## How It Works

### Audio-Visual Correlation

The application allows users to observe the direct relationship between visual events and audio features:

1. **Video Processing Mode**: 
   - Load an MP4 file containing both video and audio
   - The app processes each video frame's corresponding audio segment
   - Display the video frame alongside its audio spectrogram
   - Observe how visual events (e.g., drone movements, object impacts) correspond to specific frequency patterns

2. **Live Recording Mode**:
   - Capture audio from a microphone in real-time
   - Display the spectrogram as it's being recorded
   - Useful for monitoring live audio and understanding acoustic properties

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Key dependencies:
- PyQt5
- numpy
- sounddevice
- pyfftw
- scipy
- librosa
- pyqtgraph
- opencv-python

### Running the Application

```bash
python audio_waterfall_app.py
```

## Usage

### Processing a Video File

1. Click "Load MP4 File" to select a video file
2. Adjust STFT parameters as needed:
   - Kaiser Beta: Controls spectral leakage (higher = sharper peaks)
   - Noise Floor %: Controls color scaling sensitivity
   - Max Frequency: Upper frequency limit for display
3. Optionally toggle "Show Video Frames" on/off
4. Click "Process File" to generate the synchronized spectrogram
5. Watch the waterfall plot update alongside video frames
6. Click "Stop Processing" to pause

### Recording Live Audio

1. Select your microphone from the device list
2. Adjust STFT parameters
3. Click "Start Recording" to begin live audio capture
4. The waterfall plot updates in real-time
5. Click "Stop Recording" to end

### Interpreting the Spectrogram

- **X-axis**: Time (seconds)
- **Y-axis**: Frequency (Hz)
- **Color intensity**: Audio amplitude (dB)
  - Brighter colors indicate stronger audio at that frequency
  - Color scaling adapts to noise floor automatically
- **Feature identification**:
  - Horizontal lines: Sustained tones or whistles
  - Vertical bands: Short-duration events or impacts
  - Diagonal lines: Frequency-modulated sounds (e.g., Doppler shifts from moving sources)

## Performance Notes

- Video processing speed depends on video resolution and frame rate
- Disable video frame display for faster processing (video still analyzed for audio)
- Real-time recording performance depends on microphone sample rate and system capabilities
- The app throttles updates to maintain smooth GUI performance

## License

This application is provided as-is for educational and research purposes.