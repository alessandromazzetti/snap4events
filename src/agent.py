import os
import asyncio
import logging
from dotenv import load_dotenv
from fastmcp import Client
from langchain_google_genai import ChatGoogleGenerativeAI
from builder import create_agent_app

logging.basicConfig(level=logging.WARNING)


async def run_agent(question: str):
    print("=" * 75)
    print("SNAP4EVENTS AGENT - START")
    print(f"👤 Query: {question}")
    print("=" * 75)

    # Connection to FastMCP server on port 8000
    mcp_client = Client("http://localhost:8000/mcp")
    await mcp_client.__aenter__()

    try:
    # LLM init
        model = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash",
            temperature=0,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Creates app and assign the LLM model to it
        app = create_agent_app(model=model, mcp_client=None)

        # Run graph
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
        print(f"Event extracted: {len(final_state['events'])}")

        if final_state['events']:
            print("\n--- DETAILS ---")
            for i, ev in enumerate(final_state['events'], 1):
                print(f"\nEvento #{i}:")
                # .model_dump_json(indent=2) is a Pydantic command to convert a Python object in a JSON format
                print(ev.model_dump_json(indent=2))
    finally:
        await mcp_client.__aexit__(None, None, None)


if __name__ == "__main__":
    load_dotenv()

    asyncio.run(run_agent(
        "I am looking for electronic music near Florence this weekend"
    ))