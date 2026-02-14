"""Unit tests for ChatHistoryService."""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from rag.services.chat_history import InMemoryHistory, ChatHistoryService


def test_in_memory_history_init():
    """Test InMemoryHistory initialization."""
    history = InMemoryHistory()
    assert len(history.messages) == 0


def test_in_memory_history_add_messages():
    """Test adding messages to history."""
    history = InMemoryHistory()
    history.add_messages([
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there")
    ])
    assert len(history.messages) == 2


def test_in_memory_history_clear():
    """Test clearing history."""
    history = InMemoryHistory()
    history.add_messages([HumanMessage(content="Hello")])
    history.clear()
    assert len(history.messages) == 0


def test_chat_history_service_init():
    """Test ChatHistoryService initialization."""
    service = ChatHistoryService()
    assert service.get_all_sessions() == []


def test_get_session_history_creates_new():
    """Test that get_session_history creates new session."""
    service = ChatHistoryService()
    history = service.get_session_history('test-session')
    assert 'test-session' in service.get_all_sessions()
    assert len(history.messages) == 0


def test_get_session_history_returns_existing():
    """Test that get_session_history returns existing session."""
    service = ChatHistoryService()
    history1 = service.get_session_history('session-1')
    history1.add_messages([HumanMessage(content="Hello")])
    
    history2 = service.get_session_history('session-1')
    assert len(history2.messages) == 1


def test_clear_session():
    """Test clearing a specific session."""
    service = ChatHistoryService()
    history = service.get_session_history('session-1')
    history.add_messages([HumanMessage(content="Hello")])
    
    service.clear_session('session-1')
    history = service.get_session_history('session-1')
    assert len(history.messages) == 0


def test_clear_nonexistent_session():
    """Test clearing a nonexistent session doesn't raise."""
    service = ChatHistoryService()
    service.clear_session('nonexistent')  # Should not raise


def test_clear_all():
    """Test clearing all sessions."""
    service = ChatHistoryService()
    service.get_session_history('session-1')
    service.get_session_history('session-2')
    
    service.clear_all()
    assert service.get_all_sessions() == []


def test_multiple_sessions():
    """Test managing multiple sessions."""
    service = ChatHistoryService()
    service.get_session_history('a')
    service.get_session_history('b')
    service.get_session_history('c')
    
    sessions = service.get_all_sessions()
    assert len(sessions) == 3
    assert set(sessions) == {'a', 'b', 'c'}
