import os
import json
import asyncio
import logging
import requests

from dotenv import load_dotenv
from fastmcp import Client
from builder import create_agent_app
from token_manager import TokenManager

logging.basicConfig(level=logging.WARNING)


class ClearMLClient:
    """
    Client for calling the Snap4City ClearML OnDemand inference API directly.

    This does NOT use OpenAI or LangChain.
    """

    def __init__(
        self,
        access_token,
        endpoint,
        base_url,
        temperature=0
    ):
        self.access_token = access_token
        self.endpoint = endpoint
        self.base_url = base_url
        self.temperature = temperature

    def generate(self, prompt):
        """Send a prompt directly to the ClearML OnDemand API."""

        body = {
            "access_token": self.access_token,
            "endpoint": self.endpoint,
            "params": {
                "prompt": prompt,
                "temperature": self.temperature
            }
        }

        response = requests.post(
            self.base_url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json"
            },
            json=body,
            timeout=120
        )

        print(f"[CLEARML] Status code: {response.status_code}")

        if not response.ok:
            raise RuntimeError(
                f"ClearML API error {response.status_code}: "
                f"{response.text[:2000]}"
            )

        return response.json()


async def run_agent(question: str):
    print("=" * 50)
    print("SNAP4EVENTS AGENT - CLEARML MODE")
    print(f"👤 Query: {question}")
    print("=" * 50)

    # 1. Load clearml configuration
    if not os.path.exists("clearml_config.json"):
        raise FileNotFoundError("clearml_config.json missing!")

    with open("clearml_config.json", "r") as f:
        clearml_config = json.load(f)

    api_base_url = clearml_config["clearml_ondemand_api_base_url"]
    llm_endpoint = clearml_config["clearml_llm_endpoint"]
    temperature = clearml_config.get("temperature", 0)

    # 2. Loads credentials
    if not os.path.exists("user_credentials.json"):
        raise FileNotFoundError("user_credentials.json missing!")

    with open("user_credentials.json", "r") as f:
        creds = json.load(f)

    # 3. Obtain a valid token
    manager = TokenManager(username=creds["username"], password=creds["password"])
    auth_token = manager.get_token()

    # 4. Connection to the MCP server
    mcp_client = Client("http://localhost:8000/mcp")
    await mcp_client.__aenter__()

    try:
        # 5. Init ClearML client, pointing to the Snap4City LLM end-point
        llm_client = ClearMLClient(
            access_token=auth_token,
            endpoint=llm_endpoint,
            base_url=api_base_url,
            temperature=temperature
        )

        # 6. Build the graph (ClearML client and mcp_client are injected via the initial state below)
        app = create_agent_app()

        # 7. Run graph
        final_state = await app.ainvoke({
            "user_query": question,
            "target_location": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "discovered_sources": [],
            "raw_page_contents": [],
            "events": [],
            "errors": [],
            "current_step": "start",
            "mcp_client": mcp_client,
            "llm_client": llm_client
        })

        print("\n✅ [RESULT]")
        print(f"Events extracted: {len(final_state['events'])}")

        if final_state['events']:
            print("\n--- DETAILS ---")
            for i, ev in enumerate(final_state['events'], 1):
                print(f"\nEvent #{i}:")
                print(ev.model_dump_json(indent=2))

    finally:
        # Close MCP connection
        await mcp_client.__aexit__(None, None, None)


if __name__ == "__main__":
    load_dotenv()

    default_question = "I am looking for events near Florence this weekend"
    user_input = input(
        f"Enter your query (press Enter for default: '{default_question}'): "
    ).strip()
    question = user_input if user_input else default_question

    asyncio.run(run_agent(question))