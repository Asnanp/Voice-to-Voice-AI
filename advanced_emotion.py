import threading
import requests
import pygame
import io
import asyncio
from edge_tts import Communicate
import os
from dotenv import load_dotenv
import re
import pyaudio
import numpy as np
import time
import json
import queue
import vosk
from typing import Optional, Dict, Any
import logging
from dataclasses import dataclass
from enum import Enum
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv('C:\\Users\\USER\\Documents\\project.env')
MISTRAL_API_KEY = os.getenv("MISTRAL")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL API key not found in environment variables")

# Constants
@dataclass(frozen=True)
class AudioConfig:
    SAMPLE_RATE: int = 16000
    CHUNK_SIZE: int = 4096
    FORMAT: int = pyaudio.paInt16
    CHANNELS: int = 1
    SILENCE_THRESHOLD: int = 500
    MAX_SILENCE_CHUNKS: int = 30

@dataclass(frozen=True)
class TTSConfig:
    FREQUENCY: int = 22050
    SIZE: int = -16
    CHANNELS: int = 2
    BUFFER: int = 512

class EmotionType(Enum):
    NEUTRAL = "neutral"
    EXCITED = "excited"
    ANGRY = "angry"
    SAD = "sad"
    CALM = "calm"
    STORY = "story"

# Voice mapping - Updated with more reliable voices
VOICE_MAPPING: Dict[str, str] = {
    EmotionType.NEUTRAL.value: "en-US-AriaNeural",
    EmotionType.EXCITED.value: "en-US-JennyNeural", 
    EmotionType.ANGRY.value: "en-US-GuyNeural",
    EmotionType.SAD.value: "en-US-SaraNeural",
    EmotionType.CALM.value: "en-US-DavisNeural",
    EmotionType.STORY.value: "en-US-NancyNeural"
}

# Enhanced emotion detection patterns
EMOTION_PATTERNS: Dict[str, list] = {
    EmotionType.EXCITED.value: [
        "awesome", "yay", "love", "superb", "epic", "amazing", "fantastic", "great", 
        "wow", "incredible", "brilliant", "perfect", "excellent", "outstanding",
        "thrilled", "excited", "happy", "joy", "wonderful", "marvelous"
    ],
    EmotionType.ANGRY.value: [
        "angry", "mad", "hate", "why", "annoyed", "damn", "frustrated", "terrible",
        "awful", "horrible", "disgusting", "ridiculous", "stupid", "idiot",
        "furious", "rage", "irritated", "pissed", "upset", "disappointed"
    ],
    EmotionType.SAD.value: [
        "sad", "hurt", "alone", "miss", "sorry", "depressed", "disappointed", "upset",
        "crying", "tears", "lonely", "heartbroken", "devastated", "miserable",
        "grief", "sorrow", "pain", "suffering", "hopeless", "despair"
    ],
    EmotionType.CALM.value: [
        "sure", "alright", "fine", "okay", "yes", "cool", "peaceful", "relaxed",
        "calm", "serene", "tranquil", "gentle", "soft", "quiet", "easy",
        "comfortable", "content", "satisfied", "pleased", "grateful"
    ],
    EmotionType.STORY.value: [
        "once upon", "long ago", "legend", "journey", "story", "tale", "narrative",
        "adventure", "quest", "hero", "princess", "dragon", "magic", "fantasy",
        "myth", "fable", "chronicle", "saga", "epic", "fairy tale"
    ]
}

# System prompt
SYSTEM_PROMPT = """You are a chill and caring male voice assistant. Your name is Asnan's Assistant. 
Be like human tone and only give response what the user asked. Don't say anything else. 
And Keep your words too short. Behave like what user says like human conversation"""

class ChatManager:
    """Manages chat history and API interactions"""
    
    def __init__(self, api_key: str, max_history: int = 20):
        self.api_key = api_key
        self.max_history = max_history
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        })
        self._request_timeout = 15
        self._max_retries = 3
    
    def add_message(self, role: str, content: str) -> None:
        """Add message to chat history"""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            # Keep system message and trim old messages
            self.history = [self.history[0]] + self.history[-(self.max_history-1):]
    
    def get_response(self, user_input: str) -> str:
        """Get response from Mistral API with retry logic"""
        self.add_message("user", user_input)
        
        for attempt in range(self._max_retries):
            try:
                response = self.session.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json={
                        "model": "mistral-small",
                        "messages": self.history[-self.max_history:],
                        "max_tokens": 150,
                        "temperature": 0.7
                    },
                    timeout=self._request_timeout
                )
                response.raise_for_status()
                reply = response.json()["choices"][0]["message"]["content"]
                self.add_message("assistant", reply)
                return reply
            except requests.exceptions.Timeout:
                logger.warning(f"API request timeout (attempt {attempt + 1}/{self._max_retries})")
                if attempt == self._max_retries - 1:
                    return "Sorry, the request timed out. Please try again."
            except requests.exceptions.RequestException as e:
                logger.error(f"Mistral API request error: {e}")
                if attempt == self._max_retries - 1:
                    return "Sorry, I'm having trouble connecting to my brain. Please check your internet connection."
            except Exception as e:
                logger.error(f"Mistral API error: {e}")
                if attempt == self._max_retries - 1:
                    return "Sorry, I'm having trouble processing that."
        
        return "Sorry, I'm having trouble processing that."

class AudioProcessor:
    """Handles audio processing and speech recognition"""
    
    def __init__(self, model_path: str, config: AudioConfig):
        self.config = config
        self.model_path = model_path
        self.audio_queue = queue.Queue()
        self.is_active = False
        self.audio_interface: Optional[pyaudio.PyAudio] = None
        self.audio_stream: Optional[pyaudio.Stream] = None
        self.vosk_model: Optional[vosk.Model] = None
        self.vosk_recognizer: Optional[vosk.KaldiRecognizer] = None
        self.current_sentence = ""
        self.silence_counter = 0
        self._processing_thread: Optional[threading.Thread] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False
        self._error_count = 0
        self._max_errors = 5
        self._is_speaking = False
    
    def set_speaking_state(self, is_speaking: bool) -> None:
        """Set whether the TTS system is currently speaking"""
        self._is_speaking = is_speaking
    
    def _validate_config(self) -> bool:
        """Validate audio configuration"""
        try:
            if self.config.SAMPLE_RATE <= 0:
                logger.error("Invalid sample rate")
                return False
            if self.config.CHUNK_SIZE <= 0:
                logger.error("Invalid chunk size")
                return False
            if self.config.SILENCE_THRESHOLD < 0:
                logger.error("Invalid silence threshold")
                return False
            return True
        except Exception as e:
            logger.error(f"Configuration validation error: {e}")
            return False
    
    def initialize(self, progress_callback=None) -> bool:
        """Initialize Vosk model and audio interface with progress tracking"""
        import time
        start_time = time.time()

        try:
            # Step 1: Validate configuration (5%)
            step_start = time.time()
            if progress_callback:
                progress_callback("Validating audio configuration...", 5)

            if not self._validate_config():
                if progress_callback:
                    progress_callback("Configuration validation failed", -1)
                return False

            step_duration = time.time() - step_start
            logger.info(f"⏱️ Audio configuration validation took: {step_duration:.2f} seconds")

            # Step 2: Check model existence (10%)
            step_start = time.time()
            if progress_callback:
                progress_callback("Checking Vosk model files...", 10)

            if not os.path.exists(self.model_path):
                logger.error(f"Model not found at {self.model_path}")
                if progress_callback:
                    progress_callback("Vosk model files not found", -1)
                return False

            step_duration = time.time() - step_start
            logger.info(f"⏱️ Model file check took: {step_duration:.2f} seconds")

            # Step 3: Load Vosk model (60% - this is the slow part)
            model_start_time = time.time()
            if progress_callback:
                progress_callback("Loading Vosk speech recognition model... (This may take a moment)", 15)

            logger.info("🔄 Starting Vosk model loading...")

            # Simulate progress during model loading (this is the slow part)
            if progress_callback:
                progress_callback("Loading neural network weights...", 25)
                time.sleep(0.5)  # Small delay to show progress
                progress_callback("Loading language model...", 35)
                time.sleep(0.5)
                progress_callback("Loading acoustic model...", 45)
                time.sleep(0.5)
                progress_callback("Initializing decoder...", 55)
                time.sleep(0.5)
                progress_callback("Finalizing model setup...", 65)

            # This is the actual slow operation
            logger.info("📥 Loading Vosk model files...")
            self.vosk_model = vosk.Model(self.model_path)

            model_duration = time.time() - model_start_time
            logger.info(f"⏱️ Vosk model loading took: {model_duration:.2f} seconds")

            if progress_callback:
                progress_callback("Initializing speech recognizer...", 75)

            self.vosk_recognizer = vosk.KaldiRecognizer(self.vosk_model, self.config.SAMPLE_RATE)
            self.vosk_recognizer.SetWords(True)

            # Step 4: Initialize audio system (20%)
            audio_start_time = time.time()
            if progress_callback:
                progress_callback("Setting up audio system...", 85)

            self.audio_interface = pyaudio.PyAudio()

            # Check available audio devices
            device_count = self.audio_interface.get_device_count()
            logger.info(f"🎤 Found {device_count} audio devices")

            # Find default input device
            default_input = self.audio_interface.get_default_input_device_info()
            logger.info(f"🎧 Using default input device: {default_input['name']}")

            audio_duration = time.time() - audio_start_time
            logger.info(f"⏱️ Audio system setup took: {audio_duration:.2f} seconds")

            # Step 5: Create audio stream (15%)
            stream_start_time = time.time()
            if progress_callback:
                progress_callback("Creating audio stream...", 95)

            self.audio_stream = self.audio_interface.open(
                format=self.config.FORMAT,
                channels=self.config.CHANNELS,
                rate=self.config.SAMPLE_RATE,
                input=True,
                frames_per_buffer=self.config.CHUNK_SIZE,
                stream_callback=self._audio_callback
            )

            stream_duration = time.time() - stream_start_time
            logger.info(f"⏱️ Audio stream creation took: {stream_duration:.2f} seconds")

            # Step 6: Finalization (5%)
            if progress_callback:
                progress_callback("Voice assistant ready!", 100)

            # Calculate total initialization time
            total_duration = time.time() - start_time
            logger.info(f"🎉 TOTAL INITIALIZATION TIME: {total_duration:.2f} seconds")
            logger.info(f"✅ Audio processor initialized successfully in {total_duration:.1f}s")

            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Audio processor initialization failed: {e}")
            if progress_callback:
                progress_callback(f"Initialization failed: {str(e)}", -1)
            return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        # Log unused parameters to avoid warnings
        _ = frame_count, time_info, status

        if self.is_active and not self._is_speaking:
            self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _is_silent(self, chunk: bytes) -> bool:
        """Check if audio chunk is silent"""
        audio_np = np.frombuffer(chunk, dtype=np.int16)
        return np.abs(audio_np).mean() < self.config.SILENCE_THRESHOLD
    
    def _process_audio_chunk(self, chunk: bytes, callback):
        """Process a single audio chunk"""
        if not self._initialized or not self.vosk_recognizer:
            logger.warning("Audio processor not properly initialized")
            return
            
        try:
            if self.vosk_recognizer.AcceptWaveform(chunk):
                result = json.loads(self.vosk_recognizer.Result())
                text = result.get('text', '').strip()
                if text:
                    logger.info(f"Complete transcription: {text}")
                    self._schedule_callback(callback, text)
                    self.current_sentence = ""
                    self.silence_counter = 0
            else:
                partial = json.loads(self.vosk_recognizer.PartialResult()).get('partial', '').strip()
                if partial != self.current_sentence:
                    self.current_sentence = partial
                    if partial:
                        logger.info(f"Partial transcription: {partial}")
                    self.silence_counter = 0
                elif self._is_silent(chunk):
                    self.silence_counter += 1
                    # Increased silence threshold to prevent random responses
                    if self.silence_counter > (self.config.MAX_SILENCE_CHUNKS * 2) and self.current_sentence and len(self.current_sentence.strip()) > 5:
                        logger.info(f"Complete transcription: {self.current_sentence}")
                        self._schedule_callback(callback, self.current_sentence)
                        self.current_sentence = ""
                        self.silence_counter = 0
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
    
    def _schedule_callback(self, callback, text: str):
        """Schedule callback in the event loop"""
        if self._event_loop and not self._event_loop.is_closed():
            try:
                asyncio.run_coroutine_threadsafe(callback(text), self._event_loop)
            except Exception as e:
                logger.error(f"Callback scheduling error: {e}")
        else:
            logger.warning("Event loop not available for callback")
    
    def start_listening(self, callback, event_loop: asyncio.AbstractEventLoop) -> None:
        """Start listening for audio input"""
        if not self._initialized or not self.audio_stream:
            logger.error("Audio processor not initialized")
            return
            
        self.is_active = True
        self._event_loop = event_loop
        self.audio_stream.start_stream()
        
        def processing_loop():
            while self.is_active:
                try:
                    chunk = self.audio_queue.get(timeout=0.1)
                    self._process_audio_chunk(chunk, callback)
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Audio processing loop error: {e}")
        
        self._processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self._processing_thread.start()
        logger.info("Audio listening started")
    
    def stop_listening(self) -> None:
        """Stop listening for audio input"""
        self.is_active = False
        if self.audio_stream:
            self.audio_stream.stop_stream()
        logger.info("Audio listening stopped")
    
    def cleanup(self) -> None:
        """Clean up audio resources"""
        try:
            self.stop_listening()
            if self.audio_stream:
                self.audio_stream.close()
            if self.audio_interface:
                self.audio_interface.terminate()
        except Exception as e:
            logger.error(f"Audio cleanup error: {e}")

class TextToSpeech:
    """Handles text-to-speech functionality with improved error handling"""
    
    def __init__(self, config: TTSConfig):
        self.config = config
        self.audio_processor = None  # Will be set by VoiceAssistant
        pygame.mixer.init(
            frequency=config.FREQUENCY,
            size=config.SIZE,
            channels=config.CHANNELS,
            buffer=config.BUFFER
        )
        self.emoji_pattern = re.compile(
            "[" "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF" "\U00002700-\U000027BF" "\U0001F900-\U0001F9FF"
            "\U0001FA70-\U0001FAFF" "\U00002600-\U000026FF" "]+", 
            flags=re.UNICODE
        )
        self._max_retries = 3
        # List of voices in order of preference
        self._voice_fallback_order = [
            "en-US-AriaNeural",
            "en-US-JennyNeural", 
            "en-US-GuyNeural",
            "en-US-SaraNeural",
            "en-US-DavisNeural",
            "en-US-NancyNeural"
        ]
        self._working_voice = None
        self._failed_voices = set()
    
    def set_audio_processor(self, audio_processor):
        """Set the audio processor reference"""
        self.audio_processor = audio_processor
    
    def _clean_text(self, text: str) -> str:
        """Remove emojis and clean text for TTS"""
        # Remove emojis
        clean_text = self.emoji_pattern.sub('', text).strip()
        
        # Remove special characters that might cause TTS issues
        clean_text = re.sub(r'[^\w\s.,!?-]', '', clean_text)
        
        # Replace problematic characters
        clean_text = clean_text.replace("'", "'").replace(""", '"').replace(""", '"')
        
        # Ensure text is not empty and has reasonable length
        if len(clean_text) > 200:  # Reduced for better reliability
            clean_text = clean_text[:200] + "..."
        
        return clean_text
    
    async def _test_voice_with_simple_text(self, voice: str) -> bool:
        """Test if a voice is working with simple text"""
        try:
            test_text = "Hello"
            communicate = Communicate(test_text, voice)
            audio_data = b""
            
            # Set a timeout for the streaming
            timeout_seconds = 10
            start_time = time.time()
            
            async for chunk in communicate.stream():
                if time.time() - start_time > timeout_seconds:
                    logger.warning(f"Voice test timeout for {voice}")
                    return False
                    
                if chunk["type"] == "audio":
                    chunk_data = chunk.get("data")
                    if chunk_data:
                        audio_data += chunk_data
                        # If we get some audio data, the voice is working
                        if len(audio_data) > 1000:  # Reasonable amount of audio
                            logger.info(f"Voice {voice} is working")
                            return True
            
            # Check if we got any audio data
            result = len(audio_data) > 100  # Minimum threshold
            if result:
                logger.info(f"Voice {voice} test passed")
            else:
                logger.warning(f"Voice {voice} test failed - no audio data")
            return result
            
        except Exception as e:
            logger.warning(f"Voice test failed for {voice}: {e}")
            return False
    
    async def _find_working_voice(self) -> str:
        """Find a working voice by testing available voices"""
        # If we already have a working voice, return it
        if self._working_voice and self._working_voice not in self._failed_voices:
            return self._working_voice
        
        # Test voices in order of preference
        for voice in self._voice_fallback_order:
            if voice not in self._failed_voices:
                logger.info(f"Testing voice: {voice}")
                if await self._test_voice_with_simple_text(voice):
                    self._working_voice = voice
                    return voice
                else:
                    self._failed_voices.add(voice)
        
        # If all voices failed, reset and try again with the first one
        logger.warning("All voices failed, resetting failed voices list")
        self._failed_voices.clear()
        self._working_voice = self._voice_fallback_order[0]
        return self._working_voice
    
    async def _generate_tts_audio(self, text: str, voice: str) -> bytes:
        """Generate TTS audio with timeout and error handling"""
        try:
            communicate = Communicate(text, voice)
            audio_data = b""
            
            # Set a reasonable timeout
            timeout_seconds = 15
            start_time = time.time()
            
            async for chunk in communicate.stream():
                if time.time() - start_time > timeout_seconds:
                    raise TimeoutError(f"TTS generation timeout for voice {voice}")
                    
                if chunk["type"] == "audio":
                    chunk_data = chunk.get("data")
                    if chunk_data:
                        audio_data += chunk_data
            
            if not audio_data:
                raise ValueError(f"No audio data generated for voice {voice}")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"TTS generation failed for voice {voice}: {e}")
            raise
    
    async def speak(self, text: str, emotion: str = EmotionType.NEUTRAL.value) -> None:
        """Convert text to speech with emotion and improved error handling"""
        try:
            clean_text = self._clean_text(text)
            if not clean_text:
                logger.warning("Empty text after cleaning, skipping TTS")
                return
            
            # Get voice for emotion (fallback to neutral if not found)
            preferred_voice = VOICE_MAPPING.get(emotion, VOICE_MAPPING[EmotionType.NEUTRAL.value])
            
            # Notify audio processor that we're about to speak
            if self.audio_processor:
                self.audio_processor.set_speaking_state(True)
            
            audio_data = None
            final_voice = None
            
            # Try with preferred voice first
            if preferred_voice not in self._failed_voices:
                try:
                    logger.info(f"Trying TTS with preferred voice [{preferred_voice}] for emotion [{emotion.upper()}]: {clean_text}")
                    audio_data = await self._generate_tts_audio(clean_text, preferred_voice)
                    final_voice = preferred_voice
                except Exception as e:
                    logger.warning(f"Preferred voice {preferred_voice} failed: {e}")
                    self._failed_voices.add(preferred_voice)
            
            # If preferred voice failed, find a working voice
            if not audio_data:
                logger.info("Finding working voice...")
                working_voice = await self._find_working_voice()
                try:
                    logger.info(f"Trying TTS with working voice [{working_voice}]: {clean_text}")
                    audio_data = await self._generate_tts_audio(clean_text, working_voice)
                    final_voice = working_voice
                except Exception as e:
                    logger.error(f"Working voice {working_voice} also failed: {e}")
                    self._failed_voices.add(working_voice)
            
            # Final fallback: try with the first voice in our list
            if not audio_data:
                fallback_voice = self._voice_fallback_order[0]
                try:
                    logger.info(f"Final fallback to voice [{fallback_voice}]: {clean_text}")
                    audio_data = await self._generate_tts_audio(clean_text, fallback_voice)
                    final_voice = fallback_voice
                except Exception as e:
                    logger.error(f"Final fallback voice {fallback_voice} failed: {e}")
                    # Reset speaking state and give up
                    if self.audio_processor:
                        self.audio_processor.set_speaking_state(False)
                    logger.error(f"All TTS attempts failed for text: {clean_text}")
                    return
            
            # Play the audio
            if audio_data:
                logger.info(f"Playing TTS audio with voice [{final_voice}]")
                def play_audio():
                    try:
                        stream = io.BytesIO(audio_data)
                        sound = pygame.mixer.Sound(stream)
                        sound.play()
                        while pygame.mixer.get_busy():
                            time.sleep(0.1)
                        # Add a small delay to prevent echo
                        time.sleep(0.5)
                    except Exception as e:
                        logger.error(f"Audio playback error: {e}")
                    finally:
                        # Always reset speaking state
                        if self.audio_processor:
                            self.audio_processor.set_speaking_state(False)
                
                threading.Thread(target=play_audio, daemon=True).start()
            else:
                # Reset speaking state
                if self.audio_processor:
                    self.audio_processor.set_speaking_state(False)
                logger.error("No audio data available for playback")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Reset speaking state on any error
            if self.audio_processor:
                self.audio_processor.set_speaking_state(False)
    
    async def speak_simple(self, text: str) -> None:
        """Simple TTS without emotion for error messages"""
        await self.speak(text, EmotionType.NEUTRAL.value)

class EmotionDetector:
    """Detects emotion from text input with enhanced pattern matching"""
    
    @staticmethod
    def detect(text: str) -> str:
        """Detect emotion from text with confidence scoring"""
        text_lower = text.lower()
        
        # Count matches for each emotion
        emotion_scores = {}
        for emotion, patterns in EMOTION_PATTERNS.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        # Return the emotion with the highest score, or neutral if no matches
        if emotion_scores:
            best_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
            logger.info(f"Detected emotion: {best_emotion} (score: {emotion_scores[best_emotion]})")
            return best_emotion
        
        return EmotionType.NEUTRAL.value
    
    @staticmethod
    def get_emotion_intensity(text: str) -> float:
        """Get emotion intensity (0.0 to 1.0)"""
        text_lower = text.lower()
        total_matches = 0
        
        for patterns in EMOTION_PATTERNS.values():
            total_matches += sum(1 for pattern in patterns if pattern in text_lower)
        
        # Normalize by text length and pattern count
        if total_matches == 0:
            return 0.0
        
        intensity = min(1.0, total_matches / (len(text.split()) * 0.1))
        return intensity

class ChatLogger:
    """Handles chat logging"""
    
    def __init__(self, log_file: str = "chat_log.txt"):
        self.log_file = log_file
    
    def log_interaction(self, user_input: str, bot_reply: str) -> None:
        """Log chat interaction"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"You: {user_input}\nBot: {bot_reply}\n" + "=" * 40 + "\n")
        except Exception as e:
            logger.error(f"Chat logging error: {e}")

class BackgroundMusic:
    """Handles background music playback"""
    
    def __init__(self, music_file: str, volume: float = 0.02):
        self.music_file = music_file
        self.volume = volume
    
    def play(self) -> None:
        """Play background music"""
        try:
            if os.path.exists(self.music_file):
                pygame.mixer.music.load(self.music_file)
                pygame.mixer.music.set_volume(self.volume)
                pygame.mixer.music.play(-1)
                logger.info("Background music started")
            else:
                logger.warning("Background music file not found")
        except Exception as e:
            logger.error(f"Background music error: {e}")

class VoiceAssistant:
    """Main voice assistant class"""
    
    def __init__(self):
        self.config = AudioConfig()
        self.tts_config = TTSConfig()
        
        # Validate environment
        if not MISTRAL_API_KEY:
            raise ValueError("MISTRAL API key is required")
            
        # Initialize components
        self.chat_manager = ChatManager(MISTRAL_API_KEY)
        self.audio_processor = AudioProcessor(
            "vosk-model-en-us-0.22",
            self.config
        )
        self.tts = TextToSpeech(self.tts_config)
        # Set the audio processor reference in TTS
        self.tts.set_audio_processor(self.audio_processor)
        
        self.emotion_detector = EmotionDetector()
        self.chat_logger = ChatLogger()
        self.background_music = BackgroundMusic(
            "Cartoon - On & On (feat. Daniel Levi) (Instrumental) [plnWPAYYjSw].mp3"
        )
        self.is_running = False
        self._startup_message_shown = False
        self._conversation_count = 0
    
    async def _show_startup_message(self) -> None:
        """Show startup message to user (only when explicitly requested)"""
        # Removed automatic startup message to prevent random talking
        # The assistant will only speak when user interacts
        self._startup_message_shown = True
        logger.info("Voice assistant ready and listening...")
    
    def _get_conversation_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        return {
            "total_conversations": self._conversation_count,
            "is_running": self.is_running,
            "startup_message_shown": self._startup_message_shown
        }
    
    async def _handle_user_input(self, text: str) -> None:
        """Handle user input asynchronously"""
        try:
            text = text.strip()
            if not text or len(text) < 3:
                return
            
            # Check for exit commands
            if text.lower() in ["exit", "quit", "stop", "goodbye", "bye"]:
                await self.tts.speak("Alright bro, peace out!", EmotionType.CALM.value)
                self.stop()
                return
            
            # Check for help commands
            if text.lower() in ["help", "what can you do", "commands"]:
                help_text = "I can chat with you, detect your emotions, and respond accordingly. Just speak naturally!"
                await self.tts.speak(help_text, EmotionType.NEUTRAL.value)
                return
            
            # Check for stats command
            if text.lower() in ["stats", "statistics", "info"]:
                stats = self._get_conversation_stats()
                stats_text = f"I've had {stats['total_conversations']} conversations with you so far!"
                await self.tts.speak(stats_text, EmotionType.NEUTRAL.value)
                return
            
            # Process the input
            emotion = self.emotion_detector.detect(text)
            emotion_intensity = self.emotion_detector.get_emotion_intensity(text)
            
            logger.info(f"Processing input: '{text}' (emotion: {emotion}, intensity: {emotion_intensity:.2f})")
            
            reply = self.chat_manager.get_response(text)
            self.chat_logger.log_interaction(text, reply)
            
            # Adjust emotion based on intensity
            if emotion_intensity < 0.3:
                emotion = EmotionType.NEUTRAL.value
            
            await self.tts.speak(reply, emotion)
            self._conversation_count += 1
        except Exception as e:
            logger.error(f"User input handling error: {e}")
            await self.tts.speak("Sorry, I'm having trouble understanding you. Can you please try again?", EmotionType.NEUTRAL.value)
    
    async def _listen_and_respond(self, progress_callback=None) -> None:
        """Listen for user input and respond asynchronously"""
        # Initialize audio processor with progress tracking
        if not self.audio_processor.initialize(progress_callback):
            logger.error("Failed to initialize audio processor")
            # Removed automatic error speech to prevent random talking
            return

        # Start listening with callback
        self.audio_processor.start_listening(self._handle_user_input, asyncio.get_event_loop())

        # Keep the event loop running with longer sleep to prevent random triggers
        try:
            while self.is_running:
                await asyncio.sleep(1.0)  # Increased sleep time to reduce CPU usage and prevent random triggers
        except Exception as e:
            logger.error(f"Listening error: {e}")
            # Removed automatic error speech to prevent random talking
        finally:
            self.audio_processor.cleanup()
    
    async def _run_assistant(self) -> None:
        """Run the assistant asynchronously"""
        import time
        assistant_start_time = time.time()

        self.is_running = True
        self.background_music.play()

        # Show startup message with greeting
        await self._show_startup_message()

        # Speak the greeting when fully ready
        await self.tts.speak("Hey! I'm Asnan, your AI assistant. I'm ready to help you!", EmotionType.EXCITED.value)

        assistant_ready_time = time.time() - assistant_start_time
        logger.info(f"🎉 ASSISTANT FULLY READY in {assistant_ready_time:.2f} seconds")

        await self._listen_and_respond()

    def start(self) -> None:
        """Start the voice assistant"""
        try:
            asyncio.run(self._run_assistant())
        except KeyboardInterrupt:
            logger.info("Voice assistant interrupted by user")
        except Exception as e:
            logger.error(f"Voice assistant error: {e}")
        finally:
            self.stop()
    
    def stop(self) -> None:
        """Stop the voice assistant"""
        self.is_running = False
        if hasattr(self, 'audio_processor'):
            self.audio_processor.cleanup()
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        logger.info("Voice assistant stopped")

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.start()
    