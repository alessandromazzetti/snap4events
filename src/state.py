from typing import List, Optional, Dict, Any
from typing_extensions import TypedDict
from models import Event

class AgentState(TypedDict):
    user_query: str
    target_location: str
    latitude: Optional[float]
    longitude: Optional[float]
    discovered_sources: List[str]
    raw_page_contents: List[Dict[str, Any]]
    events: List[Event]
    errors: List[str]
    current_step: str
    mcp_client: Any
    llm_client: object