import json
import os
from datetime import datetime
from token_manager import TokenManager
from models import Event, EventCategory, Coordinates
from km4city_client import KM4CityClient

# Loads credentials
def load_credentials(path="user_credentials.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Credentials '{path}'not found.")
    with open(path, "r") as f:
        return json.load(f)

# Calls endpoint /events/ of Km4City/Snap4City Advanced Smart City API for a specific area (default:
# Firenze) and validates obtained results with Pydantic Event model
def fetch_and_validate_events(token_manager=None, lat=45.4642, lon=9.1900, range_="month"):

    print(f"\n[KM4CITY] Retrieving events around (lat={lat}, lon={lon}), range='{range_}'...")
    client = KM4CityClient(token_manager=token_manager)

    try:
        raw_data = client.search_events(lat=lat, lon=lon, range_=range_, max_dists=15, max_results=50)
    except Exception as e:
        print(f"[KM4CITY ERROR] Failed request: {e}")
        return []

    features = raw_data.get("features", [])
    print(f"[KM4CITY] Found {len(features)} events (fullCount: {raw_data.get('fullCount', '?')})")

    validated_events = []
    for feature in features:
        # DEBUG: prints raw structure for veriyfing
        print("\n--- Dettaglio Feature Grezza (per confronto campi) ---")
        print(json.dumps(feature, indent=2))
        print("------------------------------------------------------\n")

        try:
            event = Event.from_km4city_feature(feature)
            validated_events.append(event)
        except Exception as e:
            print(f"[KM4CITY WARNING] Invalid event: {e}")

    print(f"[KM4CITY] {len(validated_events)} events correctly validated.")
    return validated_events

def main():
    print("=== Local test snap4events ===")

    # Loads credentials and init token manager
    token_manager = None
    try:
        creds = load_credentials("user_credentials.json")
        username = creds.get("username")
        password = creds.get("password")

        print(f"[AUTH] Authentication for user: {username}")
        token_manager = TokenManager(
            username=username,
            password=password,
            client_id="clearml-apis",
            store_path="token_stored.json"
        )

        # Test retrieve token from Snap4City
        access_token = token_manager.get_token()
        print(f"[AUTH] Token successively obtained! (Length: {len(access_token)})")

    except Exception as e:
        print(f"[AUTH ERROR] Failed authentication: {e}")

    # Retrieve events
    km4city_events = fetch_and_validate_events(token_manager=token_manager)
    for ev in km4city_events[:3]:
        print(ev.model_dump_json(indent=2))


if __name__ == "__main__":
    main()