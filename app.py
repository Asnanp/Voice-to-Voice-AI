from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
import threading
import asyncio
import json
import uuid
import time
import psutil
import random
from datetime import datetime
from typing import Dict, Any, List, Optional
import logging
import os
from advanced_emotion import VoiceAssistant, EmotionDetector, ChatManager, MISTRAL_API_KEY

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
voice_assistant = None
assistant_thread = None
chat_history = []
start_time = None
system_stats = {
    'status': 'stopped',
    'conversations': 0,
    'uptime': None,
    'current_emotion': 'neutral',
    'last_interaction': None,
    'cpu_usage': 0,
    'memory_usage': 0,
    'network_activity': 0,
    'model_load': 0,
    'total_requests': 0,
    'success_rate': 98.5,
    'avg_response_time': 1200
}

class WebVoiceAssistant(VoiceAssistant):
    """Extended VoiceAssistant with web interface support"""

    def __init__(self):
        super().__init__()
        self.web_callbacks = []
        self.is_web_mode = True

    def add_web_callback(self, callback):
        """Add callback for web interface updates"""
        self.web_callbacks.append(callback)

    def notify_web_clients(self, event_type: str, data: Dict[str, Any]):
        """Notify web clients of events"""
        for callback in self.web_callbacks:
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Web callback error: {e}")

    async def _handle_user_input(self, text: str) -> None:
        """Override to add web notifications"""
        try:
            # Notify web clients of user input
            self.notify_web_clients('user_input', {
                'text': text,
                'timestamp': datetime.now().isoformat()
            })

            # Call parent method
            await super()._handle_user_input(text)

            # Update chat history and stats
            global chat_history, system_stats
            emotion = self.emotion_detector.detect(text)
            reply = self.chat_manager.history[-1]['content'] if self.chat_manager.history else "No response"

            chat_entry = {
                'id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'user_input': text,
                'bot_response': reply,
                'emotion': emotion,
                'emotion_intensity': self.emotion_detector.get_emotion_intensity(text)
            }

            chat_history.append(chat_entry)
            system_stats['conversations'] += 1
            system_stats['current_emotion'] = emotion
            system_stats['last_interaction'] = datetime.now().isoformat()

            # Notify web clients of response
            self.notify_web_clients('bot_response', {
                'text': reply,
                'emotion': emotion,
                'timestamp': datetime.now().isoformat(),
                'chat_entry': chat_entry
            })

        except Exception as e:
            logger.error(f"Web user input handling error: {e}")

    def start_with_progress(self, progress_callback=None):
        """Start the assistant with progress tracking"""
        try:
            if progress_callback:
                progress_callback("Initializing voice assistant...", 15)

            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Run the assistant with progress tracking
            loop.run_until_complete(self._run_with_progress(progress_callback))

        except Exception as e:
            logger.error(f"Assistant startup error: {e}")
            if progress_callback:
                progress_callback(f"Startup failed: {str(e)}", -1)
        finally:
            try:
                loop.close()
            except:
                pass

    async def _run_with_progress(self, progress_callback=None):
        """Run assistant with progress tracking"""
        import time
        total_start_time = time.time()

        try:
            if progress_callback:
                progress_callback("Setting up audio processor...", 20)

            # Initialize audio processor with progress
            if not self.audio_processor.initialize(progress_callback):
                if progress_callback:
                    progress_callback("Failed to initialize audio processor", -1)
                return

            if progress_callback:
                progress_callback("Starting background music...", 95)

            # Start background music (if available)
            try:
                if hasattr(self, 'background_music') and hasattr(self.background_music, 'start'):
                    self.background_music.start()
            except Exception as e:
                logger.warning(f"Background music not available: {e}")

            if progress_callback:
                progress_callback("Voice assistant ready!", 100)

            # Calculate total startup time
            total_startup_time = time.time() - total_start_time
            logger.info(f"🚀 COMPLETE STARTUP TIME: {total_startup_time:.2f} seconds")

            # Emit ready signal with timing
            socketio.emit('assistant_ready', {
                'message': f'Voice assistant ready! (Loaded in {total_startup_time:.1f}s)',
                'startup_time': total_startup_time,
                'timestamp': datetime.now().isoformat()
            })

            # Start the main assistant
            await self._run_assistant()

        except Exception as e:
            logger.error(f"Assistant run error: {e}")
            if progress_callback:
                progress_callback(f"Runtime error: {str(e)}", -1)

def web_event_callback(event_type: str, data: Dict[str, Any]):
    """Callback for web events"""
    socketio.emit('assistant_event', {
        'type': event_type,
        'data': data
    })

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    global system_stats
    return jsonify(system_stats)

@app.route('/api/chat/history')
def get_chat_history():
    """Get chat history"""
    global chat_history
    return jsonify(chat_history)

@app.route('/api/assistant/start', methods=['POST'])
def start_assistant():
    """Start the voice assistant"""
    global voice_assistant, assistant_thread, system_stats, start_time

    try:
        if voice_assistant and voice_assistant.is_running:
            return jsonify({'error': 'Assistant already running'}), 400

        voice_assistant = WebVoiceAssistant()
        voice_assistant.add_web_callback(web_event_callback)

        def run_assistant():
            try:
                # Create progress callback for initialization
                def progress_callback(message, progress):
                    logger.info(f"Progress: {progress}% - {message}")
                    socketio.emit('initialization_progress', {
                        'message': message,
                        'progress': progress,
                        'timestamp': datetime.now().isoformat()
                    })
                    # Force emit immediately
                    socketio.sleep(0)

                # Pass progress callback to assistant
                voice_assistant.start_with_progress(progress_callback)
            except Exception as e:
                logger.error(f"Assistant error: {e}")
                system_stats['status'] = 'error'
                socketio.emit('initialization_progress', {
                    'message': f'Error: {str(e)}',
                    'progress': -1,
                    'timestamp': datetime.now().isoformat()
                })

        assistant_thread = threading.Thread(target=run_assistant, daemon=True)
        assistant_thread.start()

        start_time = time.time()
        system_stats['status'] = 'running'
        system_stats['uptime'] = datetime.now().isoformat()

        return jsonify({'message': 'Assistant started successfully'})

    except Exception as e:
        logger.error(f"Failed to start assistant: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/assistant/stop', methods=['POST'])
def stop_assistant():
    """Stop the voice assistant"""
    global voice_assistant, system_stats

    try:
        if voice_assistant:
            voice_assistant.stop()
            voice_assistant = None

        system_stats['status'] = 'stopped'
        system_stats['uptime'] = None

        return jsonify({'message': 'Assistant stopped successfully'})

    except Exception as e:
        logger.error(f"Failed to stop assistant: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat/send', methods=['POST'])
def send_message():
    """Send message to assistant via web interface"""
    global voice_assistant

    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        if not voice_assistant or not voice_assistant.is_running:
            return jsonify({'error': 'Assistant is not running'}), 400

        # Process message asynchronously
        def process_message():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(voice_assistant._handle_user_input(message))
                loop.close()
            except Exception as e:
                logger.error(f"Message processing error: {e}")

        threading.Thread(target=process_message, daemon=True).start()

        return jsonify({'message': 'Message sent successfully'})

    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info('Client connected')
    emit('status', system_stats)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    logger.info('Client disconnected')

@socketio.on('request_status')
def handle_status_request():
    """Handle status request"""
    emit('status', system_stats)

@socketio.on('send_message')
def handle_send_message(data):
    """Handle message from web interface"""
    message = data.get('message', '').strip()
    if message:
        # Process message through the API endpoint
        def process_web_message():
            try:
                if voice_assistant and voice_assistant.is_running:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(voice_assistant._handle_user_input(message))
                    loop.close()
                else:
                    emit('error', {'message': 'Assistant is not running'})
            except Exception as e:
                logger.error(f"Web message processing error: {e}")
                emit('error', {'message': 'Failed to process message'})

        threading.Thread(target=process_web_message, daemon=True).start()

# Additional API Endpoints (removed duplicates)

@app.route('/api/chat/send', methods=['POST'])
def send_chat_message():
    """Send a chat message via API"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()

        if not message:
            return jsonify({'status': 'error', 'error': 'Message is required'}), 400

        if not voice_assistant or not voice_assistant.is_running:
            return jsonify({'status': 'error', 'error': 'Assistant is not running'}), 400

        # Process message in background
        def process_message():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(voice_assistant._handle_user_input(message))
                loop.close()
            except Exception as e:
                logger.error(f"Message processing error: {e}")

        threading.Thread(target=process_message, daemon=True).start()
        return jsonify({'status': 'success', 'message': 'Message sent successfully'})

    except Exception as e:
        logger.error(f"Chat API error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Removed duplicate route - using the existing one at line 125

@app.route('/api/system/stats', methods=['GET'])
def get_system_stats():
    """Get system statistics"""
    try:
        # Update system stats
        if start_time:
            uptime_seconds = time.time() - start_time
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            seconds = int(uptime_seconds % 60)
            system_stats['uptime'] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        # Simulate system metrics (in production, use real metrics)
        try:
            import psutil
            system_stats['cpu_usage'] = psutil.cpu_percent()
            system_stats['memory_usage'] = psutil.virtual_memory().percent
        except ImportError:
            system_stats['cpu_usage'] = random.randint(10, 80)
            system_stats['memory_usage'] = random.randint(30, 70)

        system_stats['network_activity'] = random.randint(50, 500)
        system_stats['model_load'] = random.randint(20, 90)
        system_stats['conversations'] = len(chat_history)

        return jsonify({
            'status': 'success',
            'stats': system_stats
        })
    except Exception as e:
        logger.error(f"System stats API error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    """Handle settings get/update"""
    try:
        if request.method == 'GET':
            # Return current settings (you can store these in a database or file)
            default_settings = {
                'theme': 'dark',
                'voiceSpeed': 1.0,
                'voicePitch': 1.0,
                'voiceVolume': 100,
                'voiceModel': 'neural',
                'voiceLanguage': 'en-US',
                'animationsEnabled': True,
                'particlesEnabled': True,
                'blurEffects': True,
                'accentColor': '#6366f1'
            }
            return jsonify({'status': 'success', 'settings': default_settings})

        elif request.method == 'POST':
            data = request.get_json()
            new_settings = data.get('settings', {})

            # Here you would save the settings to database or file
            # For now, just log and return success
            logger.info(f"Settings updated: {new_settings}")

            return jsonify({'status': 'success', 'message': 'Settings saved successfully'})

    except Exception as e:
        logger.error(f"Settings API error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get analytics data"""
    try:
        # Generate sample analytics data
        analytics_data = {
            'totalInteractions': len(chat_history),
            'avgSession': '12m 34s',
            'successRate': '98.5%',
            'satisfaction': '4.8/5',
            'usageData': {
                'labels': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                'data': [12, 19, 3, 5, 2, 3, 9]
            },
            'emotionDistribution': {
                'neutral': 45,
                'excited': 20,
                'calm': 15,
                'happy': 12,
                'sad': 5,
                'angry': 3
            },
            'responseTimeData': {
                'labels': ['< 1s', '1-2s', '2-3s', '3-4s', '> 4s'],
                'data': [45, 30, 15, 8, 2]
            },
            'featureUsage': {
                'voice': 80,
                'text': 65,
                'translation': 45,
                'codeGen': 30,
                'analysis': 55,
                'settings': 25
            }
        }

        return jsonify({
            'status': 'success',
            'analytics': analytics_data
        })
    except Exception as e:
        logger.error(f"Analytics API error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

# Update the web callback to emit more detailed events
def web_callback(event_type: str, data: Dict[str, Any]):
    """Enhanced callback for web interface updates"""
    try:
        if event_type == 'user_input':
            chat_history.append({
                'id': str(uuid.uuid4()),
                'type': 'user',
                'content': data.get('text', ''),
                'timestamp': datetime.now().isoformat(),
                'emotion': 'neutral'
            })
            system_stats['last_interaction'] = datetime.now().isoformat()
            system_stats['total_requests'] += 1

        elif event_type == 'bot_response':
            chat_history.append({
                'id': str(uuid.uuid4()),
                'type': 'bot',
                'content': data.get('text', ''),
                'timestamp': datetime.now().isoformat(),
                'emotion': data.get('emotion', 'neutral')
            })
            system_stats['current_emotion'] = data.get('emotion', 'neutral')

        # Emit to all connected clients
        socketio.emit('assistant_event', {
            'type': event_type,
            'data': data,
            'timestamp': datetime.now().isoformat()
        })

        # Emit updated stats
        socketio.emit('status', system_stats)

    except Exception as e:
        logger.error(f"Web callback error: {e}")

if __name__ == '__main__':
    # Create templates and static directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)

    logger.info("Starting Flask web application...")
    print("===================================================")
    print("🚀 Starting Asnan's Advanced Voice Assistant Server...")
    print("🌐 Open your browser and go to http://localhost:5000")
    print("===================================================")

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
