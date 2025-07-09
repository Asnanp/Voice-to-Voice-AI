# 🎤 Asnan's Advanced Voice Assistant with Web Interface

A sophisticated voice assistant with emotion detection, real-time web interface, and advanced analytics dashboard.

## 🚀 Features

### Core Voice Assistant
- **Advanced Emotion Detection**: Detects 6 different emotions (neutral, excited, angry, sad, calm, story)
- **Text-to-Speech**: High-quality voice synthesis with emotion-based voice selection
- **Speech Recognition**: Real-time voice input processing using Vosk
- **AI Chat**: Powered by Mistral AI for intelligent conversations
- **Background Music**: Ambient music during conversations

### Modern Web Interface
- **Real-time Dashboard**: Live monitoring and control of the voice assistant
- **Responsive Design**: Works perfectly on desktop, tablet, and mobile devices
- **Interactive Chat**: Send messages via web interface or voice
- **Voice Visualizer**: Animated voice activity indicators
- **Emotion Analytics**: Real-time emotion tracking with charts
- **System Statistics**: Conversation count, uptime, and current emotion display

### Technical Features
- **WebSocket Communication**: Real-time bidirectional communication
- **REST API**: Complete API for assistant control and data access
- **Modern UI/UX**: Glass morphism design with smooth animations
- **Particle Background**: Dynamic animated background effects
- **Notifications**: Toast notifications for system events
- **Loading States**: Smooth loading indicators and transitions

## 🚀 Quick Start

### Prerequisites
1. **Mistral API Key** - Get from [Mistral AI](https://mistral.ai/)
2. **Vosk Model** - Download from [Vosk Models](https://alphacephei.com/vosk/models)
3. **Python 3.8+**

### Installation

1. **Clone and setup**
```bash
cd "your-project-directory"
pip install -r requirements.txt
```

2. **Environment Setup**
Create `project.env` file:
```env
MISTRAL=your_mistral_api_key_here
```

3. **Download Vosk Model**
- Download `vosk-model-small-en-us-0.15` 
- Extract to `C:\Users\USER\Downloads\vosk-model-small-en-us-0.15\`
- Or update the path in `advanced_emotion.py` line 1078

4. **Optional: Background Music**
Place your music file as `Cartoon - On & On (feat. Daniel Levi) (Instrumental) [plnWPAYYjSw].mp3`

### Running

```bash
python advanced_emotion.py
```

The assistant will:
- Start the web dashboard at http://localhost:5000
- Load the Vosk model asynchronously
- Begin listening for voice commands
- Display feature status and instructions

## 🎮 Usage

### Voice Commands
- **"Be serious"** - Switch to professional mode
- **"Talk like a pirate"** - Switch to pirate personality
- **"Be funny"** - Switch to humorous mode
- **"My name is [Name]"** - Assistant learns your name
- **"I like [something]"** - Assistant remembers preferences
- **"Help"** - Get available commands
- **"Stats"** - Get conversation statistics
- **"Exit"** - Shutdown assistant

### Hotkeys
- **F12** - Toggle listening mode on/off

### Web Dashboard
Visit http://localhost:5000 to see:
- Current emotion and intensity
- Real-time microphone volume
- Last conversation
- Assistant status (listening/speaking)
- Chat statistics and uptime
- Recent emotion history

## 🔧 Configuration

### Audio Settings
Edit `AudioConfig` in `advanced_emotion.py`:
```python
SAMPLE_RATE: int = 16000
CHUNK_SIZE: int = 4096
SILENCE_THRESHOLD: int = 500
```

### TTS Settings
Edit `TTSConfig` for audio quality:
```python
FREQUENCY: int = 22050
CHANNELS: int = 2
BUFFER: int = 512
```

### Personality Prompts
Add custom personalities in `PERSONALITY_PROMPTS` dictionary.

## 🛠️ Architecture

### Core Components
- **VoiceAssistant** - Main orchestrator
- **AudioProcessor** - Handles speech recognition
- **TextToSpeech** - Enhanced TTS with emotion
- **ChatManager** - API interactions and streaming
- **PersonalityManager** - Dynamic personality switching
- **MemoryManager** - Context and preference storage
- **ErrorHandler** - Robust error handling
- **HotkeyManager** - Keyboard input handling
- **DashboardManager** - Real-time web interface

### Data Flow
1. Audio input → Vosk → Text transcription
2. Text → Memory extraction + Personality detection
3. Text → Mistral API (streaming) → Response chunks
4. Response → Emotion detection → Enhanced TTS
5. All events → Dashboard updates

## 🐛 Troubleshooting

### Common Issues

**"Model not found"**
- Verify Vosk model path in line 1078
- Download correct model version

**"API key not found"**
- Check `project.env` file exists
- Verify `MISTRAL` key is set correctly

**"Audio device error"**
- Check microphone permissions
- Verify PyAudio installation
- Try different audio device

**"Dashboard not loading"**
- Check port 5000 is available
- Install Flask dependencies
- Check firewall settings

### Dependencies Issues
```bash
# Windows PyAudio fix
pip install pipwin
pipwin install pyaudio

# Alternative sounddevice install
conda install -c conda-forge python-sounddevice
```

## 📈 Performance Tips

1. **SSD Storage** - Store Vosk model on SSD for faster loading
2. **RAM** - 8GB+ recommended for smooth operation
3. **Network** - Stable internet for Mistral API calls
4. **Audio** - Use quality microphone for better recognition

## 🔒 Security Notes

- API keys stored in environment files
- No data transmitted except to Mistral API
- Local speech processing with Vosk
- Dashboard runs on localhost only

## 📝 License

This project is for educational and personal use. Please respect API terms of service.

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

---

**Enjoy your enhanced voice assistant! 🎉**
