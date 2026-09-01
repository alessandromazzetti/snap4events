import asyncio
import json
from fastmcp import Client


async def main():

    client = Client("http://localhost:8000/mcp")

    async with client:

        result = await client.call_tool(
            "get_events",
            {
                "city": "Firenze",
                "limit": 50
            }
        )

        print("\n=== EVENTS ===")

        if hasattr(result, "data") and result.data:
            events = json.loads(result.data)

            print(f"Found {len(events)} events")

            for event in events:
                print(
                    f"- {event['title']} | "
                    f"{event['category']} | "
                    f"{event['start_datetime']}"
                )


if __name__ == "__main__":
    asyncio.run(main())