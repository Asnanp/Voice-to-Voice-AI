# 🎤 Advanced Voice To Voice with Sentiment Prediction

## Overview

A sophisticated, AI-powered voice assistant with real-time speech recognition, emotional intelligence, and a beautiful web interface. Built with Python, Flask, and modern web technologies, this assistant provides natural conversation capabilities with emotion detection and adaptive responses.

## ✨ Features

### 🎯 Core Features
- **Real-time Speech Recognition** - Powered by Vosk for accurate offline speech-to-text
- **Emotional Intelligence** - Detects and responds to user emotions with appropriate voice tones
- **Advanced Text-to-Speech** - Multiple neural voices with emotion-based selection
- **Web Dashboard** - Modern, responsive web interface for monitoring and control
- **Chat History** - Persistent conversation logging with emotion tracking
- **Background Music** - Ambient audio for enhanced user experience

### 🧠 AI & Intelligence
- **Mistral AI Integration** - Advanced language model for intelligent responses
- **Emotion Detection** - Analyzes text for emotional context (excited, angry, sad, calm, story)
- **Contextual Responses** - Maintains conversation history for coherent interactions
- **Adaptive Voice Selection** - Automatic voice switching based on detected emotions

### 🎨 Web Interface
- **Real-time Dashboard** - Live system stats and conversation monitoring
- **Interactive Chat** - Send messages directly through the web interface
- **Progress Tracking** - Visual feedback during system initialization
- **System Analytics** - Performance metrics and usage statistics
- **Settings Panel** - Customizable voice and system preferences

### 🔧 Technical Features
- **Async Processing** - Non-blocking audio processing and API calls
- **Error Handling** - Robust error recovery and fallback mechanisms
- **Thread Safety** - Proper synchronization for concurrent operations
- **Resource Management** - Efficient memory and CPU usage
- **Modular Architecture** - Clean, extensible code structure

## 📋 Requirements

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for models and dependencies
- **Audio**: Working microphone and speakers/headphones

### Hardware Requirements
- **Microphone**: Any USB or built-in microphone
- **Audio Output**: Speakers or headphones
- **Network**: Internet connection for AI API access

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AsnanP/voice-to-voice-ai.git
cd advanced-voice-assistant
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Vosk Model
Download the English Vosk model and extract it to the project directory:
```bash
# Download from: https://alphacephei.com/vosk/models
# Extract to: ./vosk-model-en-us-0.22/
```

### 5. Environment Setup
Create a `.env` file in the project root:
```env
MISTRAL=your_mistral_api_key_here
```

### 6. Audio Files (Optional)
Place your background music file in the project directory:
- `Cartoon - On & On (feat. Daniel Levi) (Instrumental) [plnWPAYYjSw].mp3`

## 🔑 API Keys

### Mistral AI API Key
1. Visit [Mistral AI](https://mistral.ai/)
2. Create an account and get your API key
3. Add it to your `.env` file as `MISTRAL=your_api_key_here`

## 📦 Dependencies

### Core Dependencies
```
flask==2.3.3
flask-socketio==5.3.6
requests==2.31.0
pygame==2.5.2
numpy==1.24.3
pyaudio==0.2.11
vosk==0.3.45
edge-tts==6.1.7
python-dotenv==1.0.0
psutil==5.9.5
```

### System Dependencies
- **Windows**: Microsoft Visual C++ Redistributable
- **macOS**: Xcode Command Line Tools
- **Linux**: `build-essential`, `libasound2-dev`, `portaudio19-dev`

## 🏃‍♂️ Usage

### 1. Start the Assistant
```bash
# Method 1: Direct execution
python advanced_emotion.py

# Method 2: Web interface
python app.py
```

### 2. Web Interface
Open your browser and navigate to:
```
http://localhost:5000
```

### 3. Voice Commands
- Say anything naturally - the assistant will respond with appropriate emotion
- The system automatically detects your emotional state and adapts its voice
- Use the web interface to monitor conversations and system status

### 4. Web Dashboard Features
- **Start/Stop** the voice assistant
- **Monitor** real-time conversations
- **View** system statistics and performance
- **Send** text messages directly
- **Customize** settings and preferences

## 🎛️ Configuration

### Audio Settings
```python
# In advanced_emotion.py
SAMPLE_RATE = 16000      # Audio sample rate
CHUNK_SIZE = 4096        # Audio buffer size
SILENCE_THRESHOLD = 500  # Silence detection threshold
```

### Voice Settings
```python
# Available voices for different emotions
VOICE_MAPPING = {
    "neutral": "en-US-AriaNeural",
    "excited": "en-US-JennyNeural",
    "angry": "en-US-GuyNeural",
    "sad": "en-US-SaraNeural",
    "calm": "en-US-DavisNeural",
    "story": "en-US-NancyNeural"
}
```

### Emotion Detection
```python
# Customize emotion keywords
EMOTION_PATTERNS = {
    "excited": ["awesome", "amazing", "great", "love"],
    "angry": ["angry", "mad", "hate", "frustrated"],
    "sad": ["sad", "hurt", "sorry", "lonely"],
    "calm": ["sure", "okay", "peaceful", "relaxed"]
}
```

## 🏗️ Architecture

### Core Components
- **VoiceAssistant**: Main orchestrator class
- **ChatManager**: Handles API communication with Mistral AI
- **AudioProcessor**: Manages speech recognition with Vosk
- **TextToSpeech**: Converts text to speech with emotion
- **EmotionDetector**: Analyzes text for emotional content
- **WebInterface**: Flask-based web dashboard

### File Structure
```
advanced-voice-assistant/
├── advanced_emotion.py    # Main voice assistant logic
├── app.py                # Web interface server
├── requirements.txt      # Python dependencies
├── .env                 # Environment variables
├── vosk-model-en-us-0.22/  # Speech recognition model
├── templates/           # HTML templates
├── static/             # CSS, JS, images
└── README.md           # This file
```

## 🔧 Troubleshooting

### Common Issues

**1. Audio not working**
```bash
# Check audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); print([p.get_device_info_by_index(i) for i in range(p.get_device_count())])"
```

**2. Vosk model not found**
- Ensure the model is extracted to `./vosk-model-en-us-0.22/`
- Check the folder structure and file permissions

**3. API key issues**
- Verify your Mistral API key is correct
- Check your internet connection
- Ensure the `.env` file is in the correct location

**4. Web interface not loading**
- Check if port 5000 is available
- Try running with `--host=0.0.0.0` flag
- Disable firewall temporarily for testing

### Performance Optimization

**1. Memory Usage**
```python
# Reduce chat history size
max_history = 10  # Instead of 20
```

**2. Audio Processing**
```python
# Adjust chunk size for better performance
CHUNK_SIZE = 2048  # Smaller chunks for lower latency
```

**3. Model Loading**
- Use SSD storage for faster model loading
- Increase system RAM for better performance

## 📊 API Reference

### REST Endpoints

**Start Assistant**
```http
POST /api/assistant/start
```

**Stop Assistant**
```http
POST /api/assistant/stop
```

**Send Message**
```http
POST /api/chat/send
Content-Type: application/json

{
  "message": "Hello, how are you?"
}
```

**Get System Status**
```http
GET /api/status
```

**Get Chat History**
```http
GET /api/chat/history
```

### WebSocket Events

**Connection**
```javascript
socket.on('connect', function() {
    console.log('Connected to voice assistant');
});
```

**Initialization Progress**
```javascript
socket.on('initialization_progress', function(data) {
    console.log(`Progress: ${data.progress}% - ${data.message}`);
});
```

**Assistant Events**
```javascript
socket.on('assistant_event', function(data) {
    console.log('Event:', data.type, data.data);
});
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Format code
black .
isort .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Vosk** - Offline speech recognition
- **Mistral AI** - Advanced language model
- **Microsoft Edge TTS** - Neural text-to-speech
- **Flask** - Web framework
- **Socket.IO** - Real-time communication

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Search existing issues on GitHub
3. Create a new issue with detailed information
4. Join our Discord server for community support

## 🔮 Future Features

- [ ] Multi-language support
- [ ] Voice cloning capabilities
- [ ] Smart home integration
- [ ] Mobile app companion
- [ ] Cloud synchronization
- [ ] Advanced analytics dashboard
- [ ] Plugin system for extensions

---

**Made with ❤️ by Asnan**
👉 [github.com/Asnanp/voice-to-voice-ai](https://github.com/Asnanp/voice-to-voice-ai)
