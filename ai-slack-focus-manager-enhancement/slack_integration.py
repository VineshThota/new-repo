import os
import asyncio
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
import json

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_bolt import App
from slack_bolt.adapter.fastapi import SlackRequestHandler
from fastapi import FastAPI, Request
import sqlite3
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SlackUser:
    """Represents a Slack user"""
    id: str
    name: str
    real_name: str
    email: Optional[str]
    is_admin: bool = False
    is_bot: bool = False
    timezone: Optional[str] = None

@dataclass
class SlackChannel:
    """Represents a Slack channel"""
    id: str
    name: str
    is_private: bool
    member_count: int
    purpose: Optional[str] = None
    topic: Optional[str] = None

@dataclass
class SlackMessageData:
    """Raw Slack message data"""
    ts: str
    user: str
    text: str
    channel: str
    thread_ts: Optional[str] = None
    reactions: List[Dict] = None
    files: List[Dict] = None
    mentions: List[str] = None
    channel_type: str = "channel"
    subtype: Optional[str] = None

class SlackIntegration:
    """Handles all Slack API interactions and webhook events"""
    
    def __init__(self, bot_token: str, signing_secret: str, app_token: Optional[str] = None):
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.app_token = app_token
        
        # Initialize Slack clients
        self.client = WebClient(token=bot_token)
        self.app = App(
            token=bot_token,
            signing_secret=signing_secret,
            token_verification_enabled=True
        )
        
        # Cache for users and channels
        self.users_cache: Dict[str, SlackUser] = {}
        self.channels_cache: Dict[str, SlackChannel] = {}
        self.cache_expiry = datetime.now()
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables for Slack data"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slack_users (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    real_name TEXT,
                    email TEXT,
                    is_admin BOOLEAN,
                    is_bot BOOLEAN,
                    timezone TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Channels table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slack_channels (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    is_private BOOLEAN,
                    member_count INTEGER,
                    purpose TEXT,
                    topic TEXT,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Raw messages table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS slack_messages_raw (
                    ts TEXT PRIMARY KEY,
                    user_id TEXT,
                    text TEXT,
                    channel_id TEXT,
                    thread_ts TEXT,
                    channel_type TEXT,
                    subtype TEXT,
                    reactions TEXT,  -- JSON string
                    files TEXT,      -- JSON string
                    mentions TEXT,   -- JSON string
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES slack_users (id),
                    FOREIGN KEY (channel_id) REFERENCES slack_channels (id)
                )
            ''')
            
            # User preferences table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    focus_duration_default INTEGER DEFAULT 90,
                    allow_critical_during_focus BOOLEAN DEFAULT TRUE,
                    priority_threshold REAL DEFAULT 0.5,
                    notification_schedule TEXT,  -- JSON string
                    custom_keywords TEXT,         -- JSON string
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES slack_users (id)
                )
            ''')
            
            conn.commit()
    
    @contextmanager
    def _get_db_connection(self):
        """Get database connection with automatic cleanup"""
        conn = sqlite3.connect('slack_focus.db')
        try:
            yield conn
        finally:
            conn.close()
    
    def _setup_event_handlers(self):
        """Setup Slack event handlers"""
        
        @self.app.event("message")
        def handle_message_events(body, logger):
            """Handle incoming message events"""
            try:
                event = body["event"]
                
                # Skip bot messages and message changes
                if event.get("subtype") in ["bot_message", "message_changed", "message_deleted"]:
                    return
                
                # Store raw message
                message_data = SlackMessageData(
                    ts=event["ts"],
                    user=event.get("user", ""),
                    text=event.get("text", ""),
                    channel=event["channel"],
                    thread_ts=event.get("thread_ts"),
                    channel_type=event.get("channel_type", "channel"),
                    subtype=event.get("subtype")
                )
                
                self._store_message(message_data)
                logger.info(f"Stored message from {message_data.user} in {message_data.channel}")
                
            except Exception as e:
                logger.error(f"Error handling message event: {e}")
        
        @self.app.command("/focus")
        def handle_focus_command(ack, respond, command):
            """Handle /focus slash command"""
            ack()
            
            user_id = command["user_id"]
            text = command.get("text", "")
            
            if text.lower() == "start":
                # Start focus session
                respond({
                    "text": "🎯 Focus session started! I'll filter non-critical messages for you.",
                    "response_type": "ephemeral"
                })
            elif text.lower() == "stop":
                # Stop focus session
                respond({
                    "text": "Focus session ended. You'll now receive all messages.",
                    "response_type": "ephemeral"
                })
            else:
                # Show help
                respond({
                    "text": "Use `/focus start` to begin a focus session or `/focus stop` to end it.",
                    "response_type": "ephemeral"
                })
        
        @self.app.shortcut("priority_summary")
        def handle_priority_summary(ack, shortcut, client):
            """Handle priority summary shortcut"""
            ack()
            
            user_id = shortcut["user"]["id"]
            
            # Get priority messages for user
            priority_messages = self.get_priority_messages_for_user(user_id, hours=24)
            
            if not priority_messages:
                text = "No high-priority messages in the last 24 hours! 🎉"
            else:
                text = f"📊 *Priority Summary (Last 24h)*\n\n"
                for msg in priority_messages[:10]:  # Top 10
                    channel_name = self.get_channel_name(msg['channel_id'])
                    user_name = self.get_user_name(msg['user_id'])
                    text += f"• **{channel_name}** - {user_name}: {msg['text'][:100]}...\n"
            
            client.chat_postEphemeral(
                channel=shortcut["channel"]["id"],
                user=user_id,
                text=text
            )
    
    async def refresh_cache(self):
        """Refresh users and channels cache"""
        if datetime.now() - self.cache_expiry < timedelta(hours=1):
            return  # Cache still valid
        
        try:
            # Refresh users
            response = await self._async_api_call("users.list")
            if response["ok"]:
                for user_data in response["members"]:
                    if not user_data.get("deleted", False):
                        user = SlackUser(
                            id=user_data["id"],
                            name=user_data["name"],
                            real_name=user_data.get("real_name", ""),
                            email=user_data.get("profile", {}).get("email"),
                            is_admin=user_data.get("is_admin", False),
                            is_bot=user_data.get("is_bot", False),
                            timezone=user_data.get("tz")
                        )
                        self.users_cache[user.id] = user
                        self._store_user(user)
            
            # Refresh channels
            response = await self._async_api_call("conversations.list", types="public_channel,private_channel")
            if response["ok"]:
                for channel_data in response["channels"]:
                    if not channel_data.get("is_archived", False):
                        channel = SlackChannel(
                            id=channel_data["id"],
                            name=channel_data["name"],
                            is_private=channel_data.get("is_private", False),
                            member_count=channel_data.get("num_members", 0),
                            purpose=channel_data.get("purpose", {}).get("value"),
                            topic=channel_data.get("topic", {}).get("value")
                        )
                        self.channels_cache[channel.id] = channel
                        self._store_channel(channel)
            
            self.cache_expiry = datetime.now()
            logger.info("Cache refreshed successfully")
            
        except SlackApiError as e:
            logger.error(f"Error refreshing cache: {e}")
    
    async def _async_api_call(self, method: str, **kwargs) -> Dict[str, Any]:
        """Make async Slack API call"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.client.api_call(method, **kwargs))
    
    def _store_user(self, user: SlackUser):
        """Store user in database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO slack_users 
                (id, name, real_name, email, is_admin, is_bot, timezone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                user.id, user.name, user.real_name, user.email,
                user.is_admin, user.is_bot, user.timezone
            ))
            conn.commit()
    
    def _store_channel(self, channel: SlackChannel):
        """Store channel in database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO slack_channels 
                (id, name, is_private, member_count, purpose, topic)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                channel.id, channel.name, channel.is_private,
                channel.member_count, channel.purpose, channel.topic
            ))
            conn.commit()
    
    def _store_message(self, message: SlackMessageData):
        """Store raw message in database"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO slack_messages_raw 
                (ts, user_id, text, channel_id, thread_ts, channel_type, subtype, reactions, files, mentions)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                message.ts, message.user, message.text, message.channel,
                message.thread_ts, message.channel_type, message.subtype,
                json.dumps(message.reactions) if message.reactions else None,
                json.dumps(message.files) if message.files else None,
                json.dumps(message.mentions) if message.mentions else None
            ))
            conn.commit()
    
    def get_messages_for_user(self, user_id: str, hours: int = 24, limit: int = 100) -> List[Dict]:
        """Get recent messages for a user"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT m.*, u.name as user_name, c.name as channel_name
                FROM slack_messages_raw m
                LEFT JOIN slack_users u ON m.user_id = u.id
                LEFT JOIN slack_channels c ON m.channel_id = c.id
                WHERE m.created_at >= datetime('now', '-{} hours')
                AND (m.text LIKE '%<@{}>%' OR m.channel_id IN (
                    SELECT channel_id FROM user_channel_memberships WHERE user_id = ?
                ))
                ORDER BY m.created_at DESC
                LIMIT ?
            '''.format(hours, user_id), (user_id, limit))
            
            columns = [description[0] for description in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_priority_messages_for_user(self, user_id: str, hours: int = 24) -> List[Dict]:
        """Get high-priority messages for a user"""
        # This would integrate with the AI priority scoring system
        messages = self.get_messages_for_user(user_id, hours)
        
        # Filter for high-priority indicators
        priority_keywords = ['urgent', 'asap', 'critical', 'emergency', 'blocked', 'down', 'failed']
        priority_messages = []
        
        for msg in messages:
            text_lower = msg['text'].lower()
            if any(keyword in text_lower for keyword in priority_keywords):
                priority_messages.append(msg)
            elif msg['user_name'] and 'manager' in msg['user_name'].lower():
                priority_messages.append(msg)
            elif msg['channel_name'] and any(channel in msg['channel_name'] 
                                           for channel in ['alert', 'urgent', 'critical']):
                priority_messages.append(msg)
        
        return priority_messages
    
    def get_user_name(self, user_id: str) -> str:
        """Get user name from cache or database"""
        if user_id in self.users_cache:
            return self.users_cache[user_id].name
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM slack_users WHERE id = ?', (user_id,))
            result = cursor.fetchone()
            return result[0] if result else f"User-{user_id[:8]}"
    
    def get_channel_name(self, channel_id: str) -> str:
        """Get channel name from cache or database"""
        if channel_id in self.channels_cache:
            return f"#{self.channels_cache[channel_id].name}"
        
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name FROM slack_channels WHERE id = ?', (channel_id,))
            result = cursor.fetchone()
            return f"#{result[0]}" if result else f"#channel-{channel_id[:8]}"
    
    def send_message(self, channel: str, text: str, thread_ts: Optional[str] = None) -> bool:
        """Send a message to Slack"""
        try:
            response = self.client.chat_postMessage(
                channel=channel,
                text=text,
                thread_ts=thread_ts
            )
            return response["ok"]
        except SlackApiError as e:
            logger.error(f"Error sending message: {e}")
            return False
    
    def send_ephemeral_message(self, channel: str, user: str, text: str) -> bool:
        """Send an ephemeral message to a specific user"""
        try:
            response = self.client.chat_postEphemeral(
                channel=channel,
                user=user,
                text=text
            )
            return response["ok"]
        except SlackApiError as e:
            logger.error(f"Error sending ephemeral message: {e}")
            return False
    
    def get_request_handler(self) -> SlackRequestHandler:
        """Get FastAPI request handler for Slack events"""
        return SlackRequestHandler(self.app)
    
    def create_fastapi_app(self) -> FastAPI:
        """Create FastAPI app with Slack integration"""
        app = FastAPI(title="SlackFocus AI API")
        handler = self.get_request_handler()
        
        @app.post("/slack/events")
        async def endpoint(req: Request):
            return await handler.handle(req)
        
        @app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        return app

# Factory function for easy initialization
def create_slack_integration() -> SlackIntegration:
    """Create SlackIntegration instance from environment variables"""
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    signing_secret = os.getenv("SLACK_SIGNING_SECRET")
    app_token = os.getenv("SLACK_APP_TOKEN")
    
    if not bot_token or not signing_secret:
        raise ValueError("SLACK_BOT_TOKEN and SLACK_SIGNING_SECRET must be set")
    
    return SlackIntegration(bot_token, signing_secret, app_token)

if __name__ == "__main__":
    # Example usage
    import uvicorn
    
    # Create Slack integration
    slack = create_slack_integration()
    
    # Create FastAPI app
    app = slack.create_fastapi_app()
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=3000)