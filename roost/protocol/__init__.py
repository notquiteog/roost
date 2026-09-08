"""Wire types shared by the agent runtime, the voice gateway and their clients.

These live in their own package because three different processes may need
them and none of them should own the definition: the runtime, the gateway,
and whatever front end is driving the pair. Keeping them here means a
protocol change breaks at import time rather than at three o'clock in the
morning on a websocket.
"""

from roost.protocol.agent import (
    AgentEvent,
    ClientCommand,
    ToolCall,
    ToolResult,
    ToolStatus,
    Turn,
)
from roost.protocol.voice import VoiceCommand, VoiceEvent

__all__ = [
    'AgentEvent',
    'ClientCommand',
    'ToolCall',
    'ToolResult',
    'ToolStatus',
    'Turn',
    'VoiceEvent',
    'VoiceCommand',
]
