from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import json
import threading
import time
from datetime import datetime
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'voice_assistant_dashboard'
socketio = SocketIO(app, cors_allowed_origins="*")

class DashboardManager:
    """Manages dashboard data and real-time updates"""
    
    def __init__(self):
        self.current_emotion = "neutral"
        self.last_user_input = ""
        self.last_bot_response = ""
        self.mic_volume = 0.0
        self.chat_stats = {
            "total_conversations": 0,
            "current_personality": "default",
            "uptime": 0,
            "start_time": datetime.now()
        }
        self.is_listening = False
        self.is_speaking = False
        self.recent_emotions = []
        self.max_recent_emotions = 10
        
    def update_emotion(self, emotion: str, intensity: float = 0.5):
        """Update current emotion"""
        self.current_emotion = emotion
        self.recent_emotions.append({
            "emotion": emotion,
            "intensity": intensity,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.recent_emotions) > self.max_recent_emotions:
            self.recent_emotions.pop(0)
        self._emit_update()
    
    def update_conversation(self, user_input: str, bot_response: str):
        """Update conversation data"""
        self.last_user_input = user_input
        self.last_bot_response = bot_response
        self.chat_stats["total_conversations"] += 1
        self._emit_update()
    
    def update_mic_volume(self, volume: float):
        """Update microphone volume level"""
        self.mic_volume = volume
        self._emit_update()
    
    def update_personality(self, personality: str):
        """Update current personality"""
        self.chat_stats["current_personality"] = personality
        self._emit_update()
    
    def set_listening_state(self, is_listening: bool):
        """Update listening state"""
        self.is_listening = is_listening
        self._emit_update()
    
    def set_speaking_state(self, is_speaking: bool):
        """Update speaking state"""
        self.is_speaking = is_speaking
        self._emit_update()
    
    def get_dashboard_data(self):
        """Get all dashboard data"""
        uptime = (datetime.now() - self.chat_stats["start_time"]).total_seconds()
        return {
            "current_emotion": self.current_emotion,
            "last_user_input": self.last_user_input,
            "last_bot_response": self.last_bot_response,
            "mic_volume": self.mic_volume,
            "chat_stats": {
                "total_conversations": self.chat_stats["total_conversations"],
                "current_personality": self.chat_stats["current_personality"],
                "uptime": int(uptime)
            },
            "is_listening": self.is_listening,
            "is_speaking": self.is_speaking,
            "recent_emotions": self.recent_emotions
        }
    
    def _emit_update(self):
        """Emit real-time update to connected clients"""
        try:
            socketio.emit('dashboard_update', self.get_dashboard_data())
        except Exception as e:
            print(f"Dashboard emit error: {e}")

# Global dashboard manager instance
dashboard = DashboardManager()

@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """API endpoint to get current dashboard data"""
    return jsonify(dashboard.get_dashboard_data())

@socketio.on('connect')
def handle_connect(auth):
    """Handle client connection"""
    print('Client connected to dashboard')
    try:
        emit('dashboard_update', dashboard.get_dashboard_data())
    except Exception as e:
        print(f"Error sending dashboard data: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected from dashboard')

def start_dashboard_server(port=5000, debug=False):
    """Start the dashboard server"""
    try:
        print(f"Starting dashboard server on http://localhost:{port}")
        socketio.run(app, host='0.0.0.0', port=port, debug=debug)
    except Exception as e:
        print(f"Dashboard server error: {e}")

if __name__ == '__main__':
    start_dashboard_server(debug=True)
