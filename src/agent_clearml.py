import os
import json
import asyncio
import logging
from dotenv import load_dotenv
from fastmcp import Client
from langchain_openai import ChatOpenAI
from builder import create_agent_app
from token_manager import TokenManager

logging.basicConfig(level=logging.WARNING)


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
        # 5. Init model, pointing to the Snap4City LLM end-point
        model = ChatOpenAI(
            base_url=api_base_url,
            api_key=auth_token,
            model=llm_endpoint,
            temperature=temperature,
            default_headers={"Authorization": f"Bearer {auth_token}"}
        )

        # 6. Run langgraph app and assigns model and mcp client to it
        app = create_agent_app(model=model, mcp_client=mcp_client)

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
            "mcp_client": mcp_client
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

    # Test query
    asyncio.run(run_agent(
        "I am looking for electronic music near Florence this weekend"
    ))