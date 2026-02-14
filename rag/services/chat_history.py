"""Chat history management service."""

import logging
from typing import Dict, List
from pydantic import BaseModel, Field
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, AIMessage


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In-memory implementation of chat message history."""
    
    messages: List[BaseMessage] = Field(default_factory=list)
    
    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store."""
        self.messages.extend(messages)
    
    def clear(self) -> None:
        """Clear all messages from history."""
        self.messages = []


class ChatHistoryService:
    """Service for managing chat history across sessions."""
    
    def __init__(self):
        """Initialize the chat history service."""
        self._store: Dict[str, InMemoryHistory] = {}
    
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get chat history for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Chat message history for the session
        """
        if session_id not in self._store:
            self._store[session_id] = InMemoryHistory()
        
        # Log existing messages
        for message in self._store[session_id].messages:
            prefix = "AI" if isinstance(message, AIMessage) else "User"
            logging.info("%s: %s", prefix, message.content)
        
        return self._store[session_id]
    
    def clear_session(self, session_id: str) -> None:
        """Clear history for a specific session."""
        if session_id in self._store:
            self._store[session_id].clear()
            logging.info("Cleared history for session: %s", session_id)
    
    def clear_all(self) -> None:
        """Clear all session histories."""
        self._store.clear()
        logging.info("Cleared all session histories")
    
    def get_all_sessions(self) -> List[str]:
        """Get list of all active session IDs."""
        return list(self._store.keys())
