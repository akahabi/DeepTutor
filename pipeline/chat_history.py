"""Chat history management for DeepTutor.

Handles storage, retrieval, and formatting of conversation history
to provide context-aware responses in the QA pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class Message:
    """Represents a single message in the conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        """Convert message to dictionary format."""
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }


class ChatHistory:
    """Manages conversation history for a single session.

    Stores messages and provides utilities for formatting history
    into prompts compatible with LangChain QA chains.
    """

    def __init__(self, max_turns: int = 20):
        """Initialize chat history.

        Args:
            max_turns: Maximum number of conversation turns to retain.
                       Older messages are dropped to stay within context limits.
                       Increased from 10 to 20 to retain more context for
                       longer study sessions.
        """
        self._messages: List[Message] = []
        self.max_turns = max_turns

    def add_user_message(self, content: str) -> None:
        """Append a user message to the history."""
        self._messages.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the history."""
        self._messages.append(Message(role="assistant", content=content))
        self._trim()

    def _trim(self) -> None:
        """Remove oldest messages when history exceeds max_turns.

        Each turn consists of one user + one assistant message, so
        we keep at most max_turns * 2 messages.
        """
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]

    def get_langchain_history(self) -> List[Tuple[str, str]]:
        """Return history in LangChain's (human, ai) tuple format.

        Pairs are assembled from consecutive user/assistant messages.
        Unpaired trailing messages are ignored.

        Returns:
            List of (human_message, ai_message) tuples.
        """
        pairs: List[Tuple[str, str]] = []
        i = 0
        while i < len(self._messages) - 1:
            if (
                self._messages[i].role == "user"
                and self._messages[i + 1].role == "assistant"
            ):
                pairs.append(
                    (self._messages[i].content, self._messages[i + 1].content)
                )
                i += 2
            else:
                i += 1
        return pairs

    def get_messages(self) -> List[Message]:
        """Return a copy of all stored messages.

        Returns a shallow copy so callers cannot accidentally mutate
        the internal message list.
        """
        return list(self._messages)

    def clear(self) -> None:
        """Clear all messages from the history."""
        self._messages = []

    def __len__(self) -> int:
        """Return the number of messages currently stored."""
        return len(self._messages)
