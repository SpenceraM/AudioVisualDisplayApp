"""
Real-time Audio Waterfall Plot Application
==========================================

A Qt-based application for real-time audio analysis with waterfall plot
and current spectrum display. Features real-time STFT processing with
configurable window size, Kaiser window beta, and microphone selection.

Author: AI Assistant
Date: 2024
"""

import sys
import numpy as np
import sounddevice as sd
import pyfftw
from scipy import signal
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QSlider, QComboBox, QPushButton, 
                            QSpinBox, QDoubleSpinBox, QGroupBox, QGridLayout,
                            QCheckBox, QProgressBar, QStatusBar, QFileDialog)
from PyQt5.QtCore import QTimer, QThread, pyqtSignal, Qt, QMutex
from PyQt5.QtGui import QFont
import pyqtgraph as pg
from collections import deque
import threading
import time
import warnings
import librosa
import os
import subprocess
import tempfile

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

# Constants
STFT_WINDOW_SIZE = 1024
STFT_ZERO_PAD_SIZE = 2048
HOP_OVERLAP_RATIO = 0.5
DEFAULT_MAX_FREQUENCY_HZ = 20000  # Default maximum frequency to display
MAX_WATERFALL_COLUMNS = 200
THROTTLE_UPDATE_MS = 50  # 50ms = 20 FPS
INITIAL_SPECTRUM_SIZE = 853  # Frequency bins after limiting to 20 kHz


def load_video_audio(video_path: str) -> tuple:
    """
    Load video file and extract audio with sample rate.
    Optimized for standalone executable.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        tuple: Audio signal and sample rate
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        Exception: If audio extraction fails
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    try:
        print("Loading video and extracting audio...")
        
        # For standalone executable, try multiple methods
        audio = None
        sample_rate = None
        
        # Method 1: Try librosa directly (works for many formats)
        try:
            print("Trying librosa direct loading...")
            audio, sample_rate = librosa.load(video_path, sr=None)
            print("Successfully loaded audio using librosa direct method")
        except Exception as e1:
            print(f"Librosa direct failed: {e1}")
            
            # Method 2: Try with OpenCV + librosa
            try:
                print("Trying OpenCV + librosa method...")
                import cv2
                
                # Use OpenCV to extract audio
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise Exception("Could not open video with OpenCV")
                
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                
                print(f"Video properties: {frame_count} frames, {fps:.2f} FPS, {duration:.2f}s")
                
                # Try librosa with different parameters
                audio, sample_rate = librosa.load(video_path, sr=44100)
                print("Successfully loaded audio using OpenCV + librosa method")
                
                cap.release()
                
            except Exception as e2:
                print(f"OpenCV method failed: {e2}")
                
                # Method 3: Try with fixed sample rate
                try:
                    print("Trying fixed sample rate method...")
                    audio, sample_rate = librosa.load(video_path, sr=44100)
                    print("Successfully loaded audio with fixed sample rate")
                except Exception as e3:
                    print(f"Fixed sample rate method failed: {e3}")
                    
                    # Method 4: Try ffmpeg if available
                    try:
                        print("Trying ffmpeg method...")
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                            temp_wav_path = temp_wav.name
                        
                        ffmpeg_cmd = [
                            'ffmpeg', '-i', video_path, 
                            '-vn', '-acodec', 'pcm_s16le', 
                            '-ar', '44100', '-ac', '1', '-y', temp_wav_path
                        ]
                        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=30)
                        
                        if result.returncode == 0:
                            audio, sample_rate = librosa.load(temp_wav_path, sr=None)
                            os.unlink(temp_wav_path)
                            print("Successfully extracted audio using ffmpeg")
                        else:
                            raise Exception("FFmpeg failed or not available")
                    except Exception as e4:
                        print(f"FFmpeg method failed: {e4}")
                        raise Exception("All audio extraction methods failed")
        
        if audio is None or sample_rate is None:
            raise Exception("Failed to extract audio from video")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        print(f"Audio loaded successfully:")
        print(f"  - Duration: {len(audio) / sample_rate:.2f} seconds")
        print(f"  - Sample rate: {sample_rate} Hz")
        print(f"  - Audio shape: {audio.shape}")
        
        return audio, sample_rate
        
    except Exception as e:
        print(f"Error loading video audio: {str(e)}")
        raise Exception(f"Failed to load audio from video: {str(e)}")


def compute_stft_spectrum(audio_data, sample_rate, kaiser_beta, limit_freq=True, max_freq=None):
    """
    Compute STFT spectrum with Kaiser windowing and zero-padding.
    
    Args:
        audio_data: Audio signal (must be exactly STFT_WINDOW_SIZE samples)
        sample_rate: Sample rate in Hz
        kaiser_beta: Kaiser window beta parameter
        limit_freq: If True, limit to max_freq
        max_freq: Maximum frequency to include (Hz). If None, uses DEFAULT_MAX_FREQUENCY_HZ
        
    Returns:
        magnitude_db: Magnitude spectrum in dB
    """
    # Create Kaiser window
    window = signal.windows.kaiser(STFT_WINDOW_SIZE, kaiser_beta)
    
    # Apply window
    windowed_data = audio_data * window
    
    # Zero-pad to 2048
    padded_data = np.zeros(STFT_ZERO_PAD_SIZE)
    padded_data[:STFT_WINDOW_SIZE] = windowed_data
    
    # Compute FFT
    fft_result = np.fft.fft(padded_data)
    
    # Get magnitude spectrum (only positive frequencies)
    magnitude = np.abs(fft_result[:STFT_WINDOW_SIZE])
    
    # Convert to dB
    magnitude_db = 20 * np.log10(magnitude + 1e-10)
    
    # Optionally limit to max_freq
    if limit_freq:
        if max_freq is None:
            max_freq = DEFAULT_MAX_FREQUENCY_HZ
        frequencies = np.linspace(0, sample_rate/2, len(magnitude))
        freq_mask = frequencies <= max_freq
        magnitude_db = magnitude_db[freq_mask]
    
    return magnitude_db


class AudioProcessor(QThread):
    """Thread for real-time audio processing with FFTW optimization."""
    
    # Signals for communication with GUI
    waterfall_ready = pyqtSignal(np.ndarray)  # spectrum row for waterfall
    
    def __init__(self, sample_rate=48000, chunk_size=STFT_WINDOW_SIZE, stft_size=STFT_WINDOW_SIZE, 
                 overlap_ratio=HOP_OVERLAP_RATIO, kaiser_beta=5.0, max_freq=None):
        super().__init__()
        
        # Audio parameters
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stft_size = STFT_WINDOW_SIZE
        self.overlap_ratio = overlap_ratio
        self.kaiser_beta = kaiser_beta
        self.max_freq = max_freq if max_freq is not None else DEFAULT_MAX_FREQUENCY_HZ
        
        # Audio stream
        self.stream = None
        self.is_recording = False
        
        # Processing buffers
        self.audio_buffer = deque(maxlen=STFT_WINDOW_SIZE * 2)
        self.hop_length = int(STFT_WINDOW_SIZE * (1 - overlap_ratio))
        
        # FFTW optimization
        self.setup_fftw()
        
        # Threading
        self.mutex = QMutex()
        self.running = False
        
    def setup_fftw(self):
        """Setup FFTW for optimized FFT computation."""
        # Create FFTW plan for STFT_ZERO_PAD_SIZE
        self.fftw_plan = pyfftw.FFTW(
            pyfftw.empty_aligned(STFT_ZERO_PAD_SIZE, dtype='complex128'),
            pyfftw.empty_aligned(STFT_ZERO_PAD_SIZE, dtype='complex128'),
            direction='FFTW_FORWARD',
            flags=('FFTW_MEASURE',)
        )
        
        # Input and output arrays for FFTW
        self.fftw_input = pyfftw.empty_aligned(STFT_ZERO_PAD_SIZE, dtype='complex128')
        self.fftw_output = pyfftw.empty_aligned(STFT_ZERO_PAD_SIZE, dtype='complex128')
        
        print(f"FFTW plan created for size {STFT_ZERO_PAD_SIZE}")
    
    
    def audio_callback(self, indata, frames, time, status):
        """Callback function for audio input."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono and add to buffer
        if indata.shape[1] > 1:
            audio_data = np.mean(indata, axis=1)
        else:
            audio_data = indata[:, 0]
        
        self.mutex.lock()
        self.audio_buffer.extend(audio_data)
        self.mutex.unlock()
    
    def start_recording(self, device=None):
        """Start audio recording."""
        try:
            self.stream = sd.InputStream(
                device=device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                callback=self.audio_callback,
                dtype=np.float32
            )
            self.stream.start()
            self.is_recording = True
            self.running = True
            self.start()
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording."""
        self.running = False
        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.wait()
    
    def run(self):
        """Main processing loop."""
        while self.running:
            if len(self.audio_buffer) >= self.stft_size:
                self.mutex.lock()
                # Copy last stft_size samples from buffer
                buffer_list = list(self.audio_buffer)
                self.mutex.unlock()
                
                audio_chunk = np.array(buffer_list[-self.stft_size:])
                
                # Compute STFT using shared function (with frequency limiting)
                magnitude = compute_stft_spectrum(audio_chunk, self.sample_rate, self.kaiser_beta, 
                                                  limit_freq=True, max_freq=self.max_freq)
                
                # Emit signals
                self.waterfall_ready.emit(magnitude)
            
            # Throttle updates to prevent excessive CPU usage
            time.sleep(THROTTLE_UPDATE_MS / 1000.0)


class VideoProcessor(QThread):
    """Thread for processing video files and generating waterfall data."""
    
    # Signals for communication with GUI
    waterfall_ready = pyqtSignal(np.ndarray)  # spectrum row for waterfall
    progress_update = pyqtSignal(int)  # progress percentage
    processing_complete = pyqtSignal()  # when processing is done
    video_duration_ready = pyqtSignal(float)  # video duration in seconds
    frame_time_update = pyqtSignal(float)  # current video frame time
    
    def __init__(self, video_path, video_frame_widget, show_video_checkbox, stft_size=STFT_WINDOW_SIZE, kaiser_beta=5.0, max_freq=None):
        super().__init__()
        self.video_path = video_path
        self.video_frame_widget = video_frame_widget
        self.show_video_checkbox = show_video_checkbox
        self.stft_size = STFT_WINDOW_SIZE
        self.kaiser_beta = kaiser_beta
        self.max_freq = max_freq if max_freq is not None else DEFAULT_MAX_FREQUENCY_HZ
        self.running = False
        self.current_frame = 0
        self.total_frames = 0
        
    def run(self):
        """Process video file frame-by-frame with synchronized audio processing."""
        try:
            self.running = True
            print("VideoProcessor started - loading audio...")
            
            # Load video audio
            audio, sample_rate = load_video_audio(self.video_path)
            print(f"Audio loaded: {len(audio)} samples at {sample_rate} Hz")
            
            # Calculate video duration and emit it
            video_duration = len(audio) / sample_rate
            self.video_duration_ready.emit(video_duration)
            
            # Get total frames from video frame widget
            self.total_frames = self.video_frame_widget.total_frames
            
            print(f"Processing {self.total_frames} video frames...")
            
            # Don't emit initial empty spectrum - wait for real data
            # Process frame by frame
            while self.running and self.current_frame < self.total_frames:
                # Check if we should show video frames
                show_video = self.show_video_checkbox.isChecked()
                
                if show_video:
                    # Advance video frame first
                    frame_start, frame_end = self.video_frame_widget.advance_frame()
                    
                    if frame_start is None or frame_end is None:
                        # End of video
                        break
                else:
                    # Calculate frame times without loading video frames
                    frame_duration = 1.0 / self.video_frame_widget.fps
                    frame_start = self.current_frame * frame_duration
                    frame_end = frame_start + frame_duration
                    
                    # Check if we've reached the end
                    if frame_start >= self.video_frame_widget.duration:
                        break
                
                # Process audio for this frame's time window
                start_sample = int(frame_start * sample_rate)
                end_sample = int(frame_end * sample_rate)
                
                # Ensure we don't go beyond audio data
                if start_sample >= len(audio):
                    break
                
                if end_sample > len(audio):
                    end_sample = len(audio)
                
                # Extract audio for this frame
                frame_audio = audio[start_sample:end_sample]
                
                # Ensure we have exactly STFT_WINDOW_SIZE samples
                if len(frame_audio) < STFT_WINDOW_SIZE:
                    # Pad with zeros if frame is too short
                    padded_audio = np.zeros(STFT_WINDOW_SIZE)
                    padded_audio[:len(frame_audio)] = frame_audio
                    frame_audio = padded_audio
                else:
                    # Take the first STFT_WINDOW_SIZE samples
                    frame_audio = frame_audio[:STFT_WINDOW_SIZE]
                
                                 # Compute STFT using shared function (with frequency limiting)
                magnitude_db = compute_stft_spectrum(frame_audio, sample_rate, self.kaiser_beta, 
                                                     limit_freq=True, max_freq=self.max_freq)
                
                # Emit spectrum for waterfall
                self.waterfall_ready.emit(magnitude_db)
                
                # Emit current frame time
                self.frame_time_update.emit(frame_start)
                
                # Update progress
                progress = int((self.current_frame / self.total_frames) * 100)
                self.progress_update.emit(progress)
                
                # Debug output for standalone executable
                if self.current_frame % 100 == 0:  # Every 100 frames
                    print(f"VideoProcessor: Processed {self.current_frame}/{self.total_frames} frames ({progress}%)")
                
                # Increment frame counter
                self.current_frame += 1
                
                # Consistent timing regardless of video display
                # Faster for first 10 frames to show waterfall quickly
                if self.current_frame <= 10:
                    # Very short delay for first 10 frames to get immediate visual feedback
                    time.sleep(0.001)
                elif show_video:
                    # When showing video, use shorter delay since video loading provides natural throttling
                    time.sleep(0.005)
                else:
                    # When not showing video, use longer delay to match video frame rate
                    time.sleep(1.0 / self.video_frame_widget.fps)  # Match video frame rate
            
            self.processing_complete.emit()
            
        except Exception as e:
            print(f"Error processing video: {e}")
            self.processing_complete.emit()
    
    def stop_processing(self):
        """Stop video processing."""
        self.running = False


class WaterfallWidget(pg.PlotWidget):
    """Custom widget for waterfall plot display."""
    
    # Signal for video frame updates
    video_frame_update = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        
        # Setup plot - Time on Y-axis, Frequency on X-axis
        self.setLabel('left', 'Frequency', units='Hz')
        self.setLabel('bottom', 'Time', units='s')
        self.setTitle('Real-time Waterfall Plot')
        
        # Waterfall parameters
        self.max_columns = MAX_WATERFALL_COLUMNS  # Maximum number of time columns (time axis on X)
        self.waterfall_data = deque(maxlen=self.max_columns)
        self.time_step = 0.1  # Time step between columns (seconds) - will be updated based on hop length
        self.current_time = 0.0
        self.video_duration = 0.0  # Total video duration
        self.processing_start_time = 0.0  # When processing started
        self.total_frames_processed = 0  # Track total frames processed
        self.noise_floor_percentile = 40.0  # Default noise floor percentile
        self.max_frequency = DEFAULT_MAX_FREQUENCY_HZ  # Maximum frequency to display
        self.sample_rate = 48000  # Will be updated based on actual sample rate
        
        # Create image item for waterfall
        self.waterfall_image = pg.ImageItem()
        self.addItem(self.waterfall_image)
        
        # Setup color map (viridis like in drone spectrogram)
        self.setup_colormap()
        
        # Initialize with empty data
        self.initialized = False
        
        # Set axis limits - Time on X (0 to max_columns * time_step), Frequency on Y (0 to max_frequency)
        # X-axis will be updated when time_step is set correctly
        self.setXRange(0, 20)  # Initial range, will be updated
        self.setYRange(0, self.max_frequency)  # Will be updated when max_frequency changes
        
        # Add grid
        self.showGrid(x=True, y=True, alpha=0.3)
        
        # Add time display text item
        self.time_text = pg.TextItem(text="Time: 0.0s", color='white', anchor=(0, 0))
        self.time_text.setPos(0, self.max_frequency)  # Position at top-left of frequency axis
        self.addItem(self.time_text)
        
    def setup_colormap(self):
        """Setup viridis colormap using matplotlib's continuous colormap."""
        # Use matplotlib's viridis colormap only
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        
        # Get viridis colormap from matplotlib
        matplotlib_cmap = cm.get_cmap('viridis')
        
        # Generate lookup table with many colors (256 levels)
        n_colors = 256
        colors = np.zeros((n_colors, 4), dtype=np.uint8)
        
        for i in range(n_colors):
            # Get color at position i/(n_colors-1) in colormap
            rgba = matplotlib_cmap(i / (n_colors - 1))
            # Convert from [0,1] to [0,255] and store as RGBA
            colors[i] = [int(rgba[0] * 255), int(rgba[1] * 255), 
                         int(rgba[2] * 255), int(rgba[3] * 255)]
        
        # Create pyqtgraph ColorMap
        cmap = pg.ColorMap(pos=np.linspace(0, 1, n_colors), color=colors)
        self.waterfall_image.setLookupTable(cmap.getLookupTable())
        
        print(f"Using matplotlib 'viridis' colormap with {n_colors} colors")
        
    def update_waterfall(self, spectrum):
        """Update waterfall plot with new spectrum data."""
        # Always add spectrum data (don't lose data)
        self.waterfall_data.append(spectrum)
        self.total_frames_processed += 1
        
        # Throttle GUI updates to prevent overwhelming the display
        # BUT always update on first spectrum or when not yet initialized
        current_time = time.time()
        if not hasattr(self, 'last_waterfall_update_time'):
            self.last_waterfall_update_time = 0
        
        # Only update display every THROTTLE_UPDATE_MS for smooth display
        # Always update if not yet initialized (first spectrum)
        # Also always update for first 5 frames to show data quickly
        throttle_seconds = THROTTLE_UPDATE_MS / 1000.0
        time_since_last_update = current_time - self.last_waterfall_update_time
        should_throttle = (self.initialized and 
                          len(self.waterfall_data) > 5 and 
                          time_since_last_update < throttle_seconds)
        
        # Skip throttled update only if we should throttle
        if should_throttle:
            return
        
        self.last_waterfall_update_time = current_time
        
        if len(self.waterfall_data) > 0:
            # Convert to 2D array - each column is a spectrum at a time point (transposed for display)
            waterfall_array = np.array(list(self.waterfall_data))
            
            # Update the display with the waterfall array
            if not self.initialized:
                self._initialize_waterfall_display(waterfall_array)
            else:
                self._update_waterfall_display(waterfall_array)
            
            # Update time display and emit video frame update
            self._update_time_and_video_frame()
    
    def _calculate_color_levels(self, waterfall_array):
        """Calculate color scaling levels."""
        # Handle edge cases for very small arrays
        if waterfall_array.size == 0:
            return -100, 0
        
        # Calculate percentiles
        noise_floor = np.percentile(waterfall_array, self.noise_floor_percentile)
        vmax = np.percentile(waterfall_array, 99)
        vmin = noise_floor
        
        # Ensure vmax > vmin (handle edge case where all values are the same)
        if vmax <= vmin:
            # Add small range if all values are identical
            vmin = noise_floor - 1
            vmax = noise_floor + 1
        
        return vmin, vmax
    
    def _update_time_and_video_frame(self):
        """Update time display and emit video frame update signal."""
        current_time = self.total_frames_processed * self.time_step
        
        # Update time display (only if time_text exists)
        if hasattr(self, 'time_text') and self.time_text:
            self.time_text.setText(f"Time: {current_time:.1f}s")
        
        # Emit signal for video frame update (throttled)
        if not hasattr(self, 'last_video_update_time'):
            self.last_video_update_time = 0
        
        # Only emit video frame updates every 0.1 seconds for better sync
        if current_time - self.last_video_update_time >= 0.1:
            self.video_frame_update.emit(current_time)
            self.last_video_update_time = current_time
    
    def _setup_image_transform(self, waterfall_array):
        """Setup the image position and transform for the waterfall."""
        self.waterfall_image.setPos(0, 0)
        
        # Scale: X = time step, Y = frequency step
        time_step = self.time_step  # seconds per pixel
        freq_step = self.max_frequency / waterfall_array.shape[1]  # Hz per pixel (scaled to fill height)
        
        from PyQt5.QtGui import QTransform
        transform = QTransform()
        transform.scale(time_step, freq_step)
        self.waterfall_image.setTransform(transform)
    
    def _initialize_waterfall_display(self, waterfall_array):
        """Initialize the waterfall display on first update."""
        # First time setup
        self.waterfall_image.setImage(waterfall_array, autoLevels=False)
        
        # Calculate and set color levels
        vmin, vmax = self._calculate_color_levels(waterfall_array)
        self.waterfall_image.setLevels([vmin, vmax])
        
        # Setup image transform
        self._setup_image_transform(waterfall_array)
        
        self.initialized = True
        
        # Update time display and emit video frame update
        self._update_time_and_video_frame()
        
        print(f"Waterfall initialized with shape: {waterfall_array.shape}")
        print(f"Noise floor (40th percentile): {vmin:.1f} dB")
        print(f"Color scaling: vmin={vmin:.1f} dB, vmax={vmax:.1f} dB")
        print(f"Dynamic range: {vmax - vmin:.1f} dB above noise floor")
    
    def _update_waterfall_display(self, waterfall_array):
        """Update existing waterfall display."""
        # Update existing image with same color scaling
        self.waterfall_image.setImage(waterfall_array, autoLevels=False)
        
        # Recalculate color levels dynamically
        vmin, vmax = self._calculate_color_levels(waterfall_array)
        self.waterfall_image.setLevels([vmin, vmax])
        
        # Update transform (in case max_frequency changed)
        self._setup_image_transform(waterfall_array)
        
        # Update time display and emit video frame update
        self._update_time_and_video_frame()

    def set_video_duration(self, duration):
        """Set the total video duration for time axis scaling."""
        self.video_duration = duration
        self.processing_start_time = time.time()
        print(f"Video duration set to {duration:.2f} seconds")
    
    def set_time_step(self, hop_length, sample_rate):
        """Set the correct time step based on hop length and sample rate."""
        self.time_step = hop_length / sample_rate
        self.sample_rate = sample_rate
        # Update X-axis range with correct time step
        max_time = self.max_columns * self.time_step
        self.setXRange(0, max_time)
        print(f"Time step set to {self.time_step:.4f} seconds (hop_length={hop_length}, sample_rate={sample_rate})")
        print(f"X-axis range updated to 0-{max_time:.2f} seconds")
    
    def set_max_frequency(self, max_freq):
        """Set the maximum frequency to display and update the Y-axis range."""
        # Limit to Nyquist frequency (half of sample rate)
        nyquist = self.sample_rate / 2.0
        self.max_frequency = min(max_freq, nyquist)
        
        print(f"Setting max frequency to {self.max_frequency:.0f} Hz (requested: {max_freq:.0f} Hz, Nyquist: {nyquist:.0f} Hz)")
        
        # Update Y-axis range
        self.setYRange(0, self.max_frequency)
        
        # Update time text position
        if hasattr(self, 'time_text') and self.time_text:
            self.time_text.setPos(0, self.max_frequency)
        
        # If already initialized, update transform
        if self.initialized and len(self.waterfall_data) > 0:
            waterfall_array = np.array(list(self.waterfall_data))
            print(f"Updating transform for waterfall with shape {waterfall_array.shape}, max_freq={self.max_frequency:.0f} Hz")
            self._setup_image_transform(waterfall_array)
    
    def update_frame_time(self, frame_time):
        """Update the time display with the current video frame time."""
        if hasattr(self, 'time_text') and self.time_text:
            self.time_text.setText(f"Time: {frame_time:.1f}s")
    
    def set_noise_floor_percentile(self, percentile):
        """Set the noise floor percentile for color scaling."""
        self.noise_floor_percentile = percentile
        # Recalculate color levels if we have data
        if len(self.waterfall_data) > 0:
            waterfall_array = np.array(list(self.waterfall_data))
            vmin, vmax = self._calculate_color_levels(waterfall_array)
            self.waterfall_image.setLevels([vmin, vmax])
            print(f"Noise floor percentile updated to {percentile:.0f}%")
    
    def get_current_video_time(self):
        """Get the current time in the video being processed."""
        if self.video_duration > 0:
            # Calculate time based on total frames processed
            return self.total_frames_processed * self.time_step
        return 0.0
    
    def reset_waterfall(self):
        """Reset waterfall for new recording session."""
        # Clear waterfall data
        self.waterfall_data.clear()
        self.total_frames_processed = 0
        
        # Reset video duration
        self.video_duration = 0.0
        self.processing_start_time = 0.0
        
        # Clear the image
        if self.waterfall_image:
            self.waterfall_image.clear()
        
        # Reset time display
        if hasattr(self, 'time_text') and self.time_text:
            self.time_text.setText("Time: 0.0s")
        
        # Reset initialization
        self.initialized = False
        
        print("Waterfall reset for new recording session")
    

class ControlsWidget(QWidget):
    """Widget containing all control elements."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the controls UI."""
        layout = QVBoxLayout()
        
        # File loading
        file_group = QGroupBox("File Input")
        file_layout = QVBoxLayout()
        
        self.load_file_button = QPushButton("Load MP4 File")
        self.load_file_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; }")
        file_layout.addWidget(self.load_file_button)
        
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("QLabel { color: gray; }")
        file_layout.addWidget(self.file_label)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Audio device selection
        device_group = QGroupBox("Audio Device")
        device_layout = QVBoxLayout()
        
        self.device_combo = QComboBox()
        self.populate_devices()
        device_layout.addWidget(QLabel("Microphone:"))
        device_layout.addWidget(self.device_combo)
        
        device_group.setLayout(device_layout)
        layout.addWidget(device_group)
        
        # STFT parameters
        stft_group = QGroupBox("STFT Parameters")
        stft_layout = QGridLayout()
        
        # Window size (hardcoded to 1024, displayed as info only)
        stft_layout.addWidget(QLabel("Window Size:"), 0, 0)
        window_size_label = QLabel("1024 (fixed)")
        window_size_label.setStyleSheet("QLabel { color: gray; }")
        stft_layout.addWidget(window_size_label, 0, 1)
        
        # Kaiser beta
        stft_layout.addWidget(QLabel("Kaiser Beta:"), 1, 0)
        self.kaiser_beta_spin = QDoubleSpinBox()
        self.kaiser_beta_spin.setRange(0.0, 20.0)
        self.kaiser_beta_spin.setValue(5.0)
        self.kaiser_beta_spin.setSingleStep(0.5)
        stft_layout.addWidget(self.kaiser_beta_spin, 1, 1)
        
        # Noise floor percentile
        stft_layout.addWidget(QLabel("Noise Floor (%):"), 2, 0)
        self.noise_floor_spin = QDoubleSpinBox()
        self.noise_floor_spin.setRange(0.0, 99.0)
        self.noise_floor_spin.setValue(40.0)
        self.noise_floor_spin.setSingleStep(1.0)
        self.noise_floor_spin.setDecimals(0)
        stft_layout.addWidget(self.noise_floor_spin, 2, 1)
        
        # Max frequency
        stft_layout.addWidget(QLabel("Max Frequency (kHz):"), 3, 0)
        self.max_freq_spin = QDoubleSpinBox()
        self.max_freq_spin.setRange(0.1, 48.0)  # Will be limited to Nyquist
        self.max_freq_spin.setValue(20.0)
        self.max_freq_spin.setSingleStep(1.0)
        self.max_freq_spin.setSuffix(" kHz")
        stft_layout.addWidget(self.max_freq_spin, 3, 1)
        
        stft_group.setLayout(stft_layout)
        layout.addWidget(stft_group)
        
        # Control buttons
        control_group = QGroupBox("Controls")
        control_layout = QVBoxLayout()
        
        self.start_button = QPushButton("Start Recording")
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; }")
        control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Recording")
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        control_layout.addWidget(self.stop_button)
        
        self.process_file_button = QPushButton("Process File")
        self.process_file_button.setEnabled(False)
        self.process_file_button.setStyleSheet("QPushButton { background-color: #FF9800; color: white; }")
        control_layout.addWidget(self.process_file_button)
        
        self.stop_processing_button = QPushButton("Stop Processing")
        self.stop_processing_button.setEnabled(False)
        self.stop_processing_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; }")
        control_layout.addWidget(self.stop_processing_button)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Status
        status_group = QGroupBox("Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("QLabel { color: green; }")
        status_layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("QLabel { color: blue; }")
        status_layout.addWidget(self.progress_label)
        
        self.time_label = QLabel("Video Time: 0.0s")
        self.time_label.setStyleSheet("QLabel { color: purple; font-weight: bold; }")
        status_layout.addWidget(self.time_label)
        
        # Video display control
        self.show_video_checkbox = QCheckBox("Show Video Frames")
        self.show_video_checkbox.setChecked(True)  # Default to showing video
        self.show_video_checkbox.setStyleSheet("QCheckBox { color: black; font-weight: bold; }")
        self.show_video_checkbox.stateChanged.connect(self.toggle_video_display)
        status_layout.addWidget(self.show_video_checkbox)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def toggle_video_display(self, state):
        """Toggle video frame display visibility."""
        # This will be handled by the main window
        pass
        
    def populate_devices(self):
        """Populate audio device combo box."""
        try:
            devices = sd.query_devices()
            input_devices = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append((i, device['name']))
            
            for device_id, device_name in input_devices:
                self.device_combo.addItem(device_name, device_id)
                
        except Exception as e:
            print(f"Error querying audio devices: {e}")
            self.device_combo.addItem("Default Device", None)


class VideoFrameWidget(QWidget):
    """Widget for displaying video frames synchronized with audio."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.current_frame = None
        self.video_cap = None
        
    def setup_ui(self):
        """Setup the video frame display UI."""
        layout = QVBoxLayout()
        
        # Video frame label
        self.frame_label = QLabel("No Video")
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(320, 240)
        self.frame_label.setStyleSheet("""
            QLabel {
                border: 2px solid #333;
                background-color: #000;
                color: white;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.frame_label)
        
        # Time label
        self.time_label = QLabel("Time: 0.0s")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("QLabel { color: white; font-weight: bold; }")
        layout.addWidget(self.time_label)
        
        self.setLayout(layout)
        self.setMaximumWidth(400)
        
    def load_video(self, video_path):
        """Load video file for frame-by-frame processing."""
        try:
            import cv2
            self.video_cap = cv2.VideoCapture(video_path)
            if self.video_cap.isOpened():
                self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
                self.total_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.total_frames / self.fps
                
                # Initialize frame-by-frame processing
                self.current_frame = 0
                self.frame_duration = 1.0 / self.fps  # Duration of one frame
                self.current_frame_time = 0.0
                
                print(f"Video loaded: {self.total_frames} frames, {self.fps:.2f} FPS, {self.duration:.2f}s duration")
                print(f"Frame duration: {self.frame_duration:.4f}s")
                return True
            else:
                print("Error: Could not open video file")
                return False
        except ImportError:
            print("Error: OpenCV not available for video frame extraction")
            return False
        except Exception as e:
            print(f"Error loading video: {e}")
            return False
    
    def advance_frame(self):
        """Advance to the next video frame and return the frame time window."""
        if not hasattr(self, 'video_cap') or self.video_cap is None:
            return None, None
            
        try:
            import cv2
            # Check if we've reached the end
            if self.current_frame >= self.total_frames:
                return None, None
            
            # Seek to current frame
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.video_cap.read()
            
            if not ret:
                return None, None
            
            # Resize frame for display
            height, width = frame.shape[:2]
            if width > 200:
                scale = 200.0 / width
                new_width = 200
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate time window for this frame
            frame_start_time = self.current_frame_time
            frame_end_time = self.current_frame_time + self.frame_duration
            
            # Update for next frame
            self.current_frame += 1
            self.current_frame_time += self.frame_duration
            
            # Display the frame
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            
            from PyQt5.QtGui import QImage, QPixmap
            q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale to fit label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.frame_label.size(), 
                Qt.KeepAspectRatio, 
                Qt.FastTransformation
            )
            
            self.frame_label.setPixmap(scaled_pixmap)
            self.time_label.setText(f"Video Frame: {self.current_frame-1} ({frame_start_time:.3f}s - {frame_end_time:.3f}s)")
            self.time_label.setStyleSheet("QLabel { color: black; font-weight: bold; }")
                        
            return frame_start_time, frame_end_time
            
        except Exception as e:
            print(f"Error advancing video frame: {e}")
            return None, None
    
    def close_video(self):
        """Close video capture and clear cache."""
        if self.video_cap:
            self.video_cap.release()
            self.video_cap = None
        
        # Clear frame cache
        if hasattr(self, 'frame_cache'):
            self.frame_cache = None
            
        self.frame_label.setText("No Video")
        self.time_label.setText("Time: 0.0s")


class AudioWaterfallApp(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Initialize components
        self.audio_processor = None
        self.video_processor = None
        self.waterfall_widget = None
        self.controls_widget = None
        self.loaded_file_path = None
        
        # Timer for updating time display
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time_display)
        
        # Setup UI
        self.setup_ui()
        self.setup_connections()
        
        # Window properties
        self.setWindowTitle("Real-time Audio Waterfall Plot")
        self.setGeometry(100, 100, 1400, 800)
        
    def setup_ui(self):
        """Setup the main UI."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout()
        
        # Left panel - controls
        self.controls_widget = ControlsWidget()
        self.controls_widget.setMaximumWidth(300)
        main_layout.addWidget(self.controls_widget)
        
        # Right panel - waterfall plot and video frame
        right_panel = QWidget()
        right_layout = QHBoxLayout()
        
        # Waterfall plot
        self.waterfall_widget = WaterfallWidget()
        right_layout.addWidget(self.waterfall_widget)
        
        # Video frame display
        self.video_frame_widget = VideoFrameWidget()
        right_layout.addWidget(self.video_frame_widget)
        
        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel)
        
        central_widget.setLayout(main_layout)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready to start recording")
        
    def setup_connections(self):
        """Setup signal connections."""
        # Control buttons
        self.controls_widget.start_button.clicked.connect(self.start_recording)
        self.controls_widget.stop_button.clicked.connect(self.stop_recording)
        self.controls_widget.load_file_button.clicked.connect(self.load_file)
        self.controls_widget.process_file_button.clicked.connect(self.process_file)
        self.controls_widget.stop_processing_button.clicked.connect(self.stop_processing)
        
        # Parameter changes
        self.controls_widget.kaiser_beta_spin.valueChanged.connect(self.update_kaiser_beta)
        self.controls_widget.noise_floor_spin.valueChanged.connect(self.update_noise_floor)
        self.controls_widget.max_freq_spin.valueChanged.connect(self.update_max_frequency)
        
        # Video display control
        self.controls_widget.show_video_checkbox.stateChanged.connect(self.toggle_video_display)
        
    def start_recording(self):
        """Start audio recording."""
        try:
            # Stop any video processing first
            if self.video_processor and self.video_processor.isRunning():
                self.video_processor.stop_processing()
                self.video_processor.wait()
                self.video_processor = None
                # Close video display
                self.video_frame_widget.close_video()
            
            # Reset waterfall for new recording session
            self.waterfall_widget.reset_waterfall()
            
            # Get selected device
            device_id = self.controls_widget.device_combo.currentData()
            
            # Get device's default sample rate
            if device_id is not None:
                device_info = sd.query_devices(device_id)
                sample_rate = int(device_info['default_samplerate'])
            else:
                # Use default sample rate if device not specified
                sample_rate = 48000
            
            # Get parameters
            kaiser_beta = self.controls_widget.kaiser_beta_spin.value()
            max_freq_hz = self.controls_widget.max_freq_spin.value() * 1000.0  # Convert kHz to Hz
            window_size = 1024  # Hardcoded
            
            print(f"Starting recording with sample rate: {sample_rate} Hz")
            print(f"Using max frequency: {max_freq_hz:.0f} Hz")
            
            # Stop existing processor if running
            if self.audio_processor and self.audio_processor.is_recording:
                self.audio_processor.stop_recording()
                self.audio_processor = None
            
            # Create new audio processor with current parameters
            self.audio_processor = AudioProcessor(
                sample_rate=sample_rate,
                chunk_size=1024,
                stft_size=window_size,
                kaiser_beta=kaiser_beta,
                max_freq=max_freq_hz
            )
            
            # Update waterfall max frequency
            self.waterfall_widget.set_max_frequency(max_freq_hz)
            
            # Connect signals
            self.audio_processor.waterfall_ready.connect(self.waterfall_widget.update_waterfall)
            
            # Set correct time step based on hop length
            self.waterfall_widget.set_time_step(self.audio_processor.hop_length, self.audio_processor.sample_rate)
            
            # Start recording
            if self.audio_processor.start_recording(device=device_id):
                # Update UI
                self.controls_widget.start_button.setEnabled(False)
                self.controls_widget.stop_button.setEnabled(True)
                self.controls_widget.status_label.setText("Recording...")
                self.controls_widget.status_label.setStyleSheet("QLabel { color: red; }")
                self.status_bar.showMessage("Recording audio...")
            else:
                self.status_bar.showMessage("Failed to start recording")
                
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.status_bar.showMessage(f"Error: {str(e)}")
            
    def stop_recording(self):
        """Stop audio recording."""
        if self.audio_processor:
            self.audio_processor.stop_recording()
            self.audio_processor = None
            
            # Update UI
            self.controls_widget.start_button.setEnabled(True)
            self.controls_widget.stop_button.setEnabled(False)
            self.controls_widget.status_label.setText("Stopped")
            self.controls_widget.status_label.setStyleSheet("QLabel { color: orange; }")
            self.status_bar.showMessage("Recording stopped")
    
    def load_file(self):
        """Load MP4 file for processing."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Select MP4 File", 
            "", 
            "MP4 Files (*.mp4);;All Files (*)"
        )
        
        if file_path:
            self.loaded_file_path = file_path
            self.controls_widget.file_label.setText(f"Loaded: {os.path.basename(file_path)}")
            self.controls_widget.file_label.setStyleSheet("QLabel { color: green; }")
            self.controls_widget.process_file_button.setEnabled(True)
            self.status_bar.showMessage(f"File loaded: {os.path.basename(file_path)}")
    
    def process_file(self):
        """Process loaded MP4 file and display waterfall."""
        if not self.loaded_file_path:
            self.status_bar.showMessage("No file loaded")
            return
        
        try:
            # Stop any current recording
            if self.audio_processor and self.audio_processor.is_recording:
                self.stop_recording()
                # Small delay to ensure audio processor is fully stopped
                time.sleep(0.2)
            
            # Stop any existing video processing
            if self.video_processor and self.video_processor.isRunning():
                self.video_processor.stop_processing()
                self.video_processor.wait()
                self.video_processor = None
            
            # Reset waterfall display for new video processing
            self.waterfall_widget.reset_waterfall()
            
            # Get parameters
            kaiser_beta = self.controls_widget.kaiser_beta_spin.value()
            max_freq_hz = self.controls_widget.max_freq_spin.value() * 1000.0  # Convert kHz to Hz
            window_size = 1024  # Hardcoded
            
            # Create video processor
            self.video_processor = VideoProcessor(
                self.loaded_file_path,
                self.video_frame_widget,
                self.controls_widget.show_video_checkbox,
                stft_size=window_size,
                kaiser_beta=kaiser_beta,
                max_freq=max_freq_hz
            )
            
            # Connect signals
            self.video_processor.waterfall_ready.connect(self.waterfall_widget.update_waterfall)
            self.video_processor.progress_update.connect(self.update_progress)
            self.video_processor.processing_complete.connect(self.processing_complete)
            self.video_processor.video_duration_ready.connect(self.waterfall_widget.set_video_duration)
            self.video_processor.frame_time_update.connect(self.waterfall_widget.update_frame_time)
            self.video_processor.frame_time_update.connect(self.update_progress_time)
            
            # Video frames will be advanced by the VideoProcessor
            
            # Get actual sample rate from video file
            try:
                _, actual_sample_rate = load_video_audio(self.loaded_file_path)
                print(f"Video sample rate: {actual_sample_rate} Hz")
            except Exception as e:
                print(f"Could not determine video sample rate: {e}, using fallback")
                actual_sample_rate = 44100  # Fallback
            
            # Set correct time step for video processing
            hop_length = int(window_size * (1 - HOP_OVERLAP_RATIO))
            self.waterfall_widget.set_time_step(hop_length, actual_sample_rate)
            
            # Set max frequency for video processing
            self.waterfall_widget.set_max_frequency(max_freq_hz)
            
            # Load video for frame display
            self.video_frame_widget.load_video(self.loaded_file_path)
            
            # Update UI
            self.controls_widget.process_file_button.setEnabled(False)
            self.controls_widget.start_button.setEnabled(False)
            self.controls_widget.stop_processing_button.setEnabled(True)
            self.controls_widget.status_label.setText("Processing file...")
            self.controls_widget.status_label.setStyleSheet("QLabel { color: orange; }")
            self.controls_widget.progress_bar.setVisible(True)
            self.controls_widget.progress_bar.setValue(0)
            
            # Start processing
            self.video_processor.start()
            self.status_bar.showMessage("Processing video file...")
            
            # Update status to show processing has started
            self.controls_widget.status_label.setText("Processing video...")
            self.controls_widget.status_label.setStyleSheet("QLabel { color: orange; }")
            
            # Don't start time timer for frame-by-frame processing
            # The frame time will be updated via the frame_time_update signal
            
        except Exception as e:
            print(f"Error starting file processing: {e}")
            self.status_bar.showMessage(f"Error: {str(e)}")
    
    def update_progress(self, progress):
        """Update progress bar."""
        self.controls_widget.progress_bar.setValue(progress)
        self.controls_widget.progress_label.setText(f"Processing: {progress}%")
    
    def update_progress_time(self, frame_time):
        """Update the time display under the progress bar."""
        self.controls_widget.time_label.setText(f"Video Time: {frame_time:.1f}s")
        self.controls_widget.time_label.setStyleSheet("QLabel { color: black; font-weight: bold; }")
    
    def toggle_video_display(self, state):
        """Toggle video frame display visibility."""
        show_video = state == 2  # Qt.Checked = 2
        self.video_frame_widget.setVisible(show_video)
        
        if show_video:
            print("Video frame display enabled")
        else:
            print("Video frame display disabled - processing will be faster")
    
    def processing_complete(self):
        """Handle completion of file processing."""
        self.controls_widget.process_file_button.setEnabled(True)
        self.controls_widget.start_button.setEnabled(True)
        self.controls_widget.stop_processing_button.setEnabled(False)
        self.controls_widget.status_label.setText("Processing complete")
        self.controls_widget.status_label.setStyleSheet("QLabel { color: green; }")
        self.controls_widget.progress_bar.setVisible(False)
        self.controls_widget.progress_label.setText("")
        self.status_bar.showMessage("File processing complete")
        
        # Stop time display timer
        self.time_timer.stop()
        
        # Reset waterfall for potential new recording
        self.waterfall_widget.reset_waterfall()
        
        # Close video frame display
        self.video_frame_widget.close_video()
    
    def stop_processing(self):
        """Stop video processing."""
        if self.video_processor:
            self.video_processor.stop_processing()
            self.video_processor.wait()
            self.video_processor = None
            
            # Update UI
            self.controls_widget.process_file_button.setEnabled(True)
            self.controls_widget.start_button.setEnabled(True)
            self.controls_widget.stop_processing_button.setEnabled(False)
            self.controls_widget.status_label.setText("Processing stopped")
            self.controls_widget.status_label.setStyleSheet("QLabel { color: orange; }")
            self.controls_widget.progress_bar.setVisible(False)
            self.controls_widget.progress_label.setText("")
            self.status_bar.showMessage("Processing stopped by user")
            
            # Stop time display timer
            self.time_timer.stop()
            
            # Reset waterfall for potential new recording
            self.waterfall_widget.reset_waterfall()
            
            # Close video frame display
            self.video_frame_widget.close_video()
    
    def update_time_display(self):
        """Update the time display with current video time."""
        if self.waterfall_widget:
            current_time = self.waterfall_widget.get_current_video_time()
            self.controls_widget.time_label.setText(f"Video Time: {current_time:.1f}s")
            
    def update_kaiser_beta(self, beta):
        """Update Kaiser window beta."""
        if self.audio_processor and self.audio_processor.is_recording:
            print(f"Updating Kaiser beta to {beta}")
            # Stop current recording
            self.stop_recording()
            # Small delay to ensure clean shutdown
            import time
            time.sleep(0.1)
            # Start new recording with updated parameters
            self.start_recording()
    
    def update_noise_floor(self, percentile):
        """Update noise floor percentile."""
        if self.waterfall_widget:
            self.waterfall_widget.set_noise_floor_percentile(percentile)
    
    def update_max_frequency(self, max_freq_khz):
        """Update maximum frequency to display."""
        max_freq_hz = max_freq_khz * 1000.0
        
        # Update waterfall widget
        if self.waterfall_widget:
            self.waterfall_widget.set_max_frequency(max_freq_hz)
        
        # If recording is active, need to restart with new frequency
        if self.audio_processor and self.audio_processor.is_recording:
            print(f"Updating max frequency to {max_freq_hz:.0f} Hz during recording")
            # Stop current recording
            self.stop_recording()
            # Small delay to ensure clean shutdown
            time.sleep(0.1)
            # Start new recording with updated parameters
            self.start_recording()
        
        # If video processing is active, need to restart with new frequency
        elif self.video_processor and self.video_processor.isRunning():
            print(f"Updating max frequency to {max_freq_hz:.0f} Hz during video processing")
            # Stop current processing
            self.video_processor.stop_processing()
            self.video_processor.wait()
            self.video_processor = None
            # Reset waterfall for new processing
            self.waterfall_widget.reset_waterfall()
            # Small delay
            time.sleep(0.1)
            # Restart video processing with new frequency
            if self.loaded_file_path:
                self.process_file()
            
    def closeEvent(self, event):
        """Handle application close."""
        self.stop_recording()
        if self.video_processor:
            self.video_processor.stop_processing()
            self.video_processor.wait()
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Audio Waterfall Plot")
    app.setApplicationVersion("1.0")
    
    # Create and show main window
    window = AudioWaterfallApp()
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
