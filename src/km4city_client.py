import requests
from typing import Optional, Dict, Any

from token_manager import TokenManager

class KM4CityClient:
    BASE_URL = "https://www.snap4city.org/superservicemap/api/v1"

    def __init__(self, token_manager: Optional[TokenManager] = None, timeout: int = 15):
        self.token_manager = token_manager
        self.timeout = timeout

    def _auth_headers(self) -> Dict[str, str]:
        if not self.token_manager:
            return {}
        try:
            token = self.token_manager.get_token()
            return {"Authorization": f"Bearer {token}"}
        except Exception as e:
            print(f"[KM4CityClient] - Cannot obtain token, proceeding without it: {e}")
            return {}

    # Calls GET /events/ (Snap4City API) to retrieve geo-localized events within a certain range
    # Return a raw GeoJSON obtained from the API
    def search_events(
        self,
        lat: float,
        lon: float,
        range_: str = "week",
        max_dists: float = 10.0,
        max_results: int = 100,
    ) -> Dict[str, Any]:

        if range_ not in ("day", "week", "month"):
            raise ValueError("range_ must be 'day', 'week' o 'month'")

        params = {
            "range": range_,
            "selection": f"{lat};{lon}",
            "maxDists": max_dists,
            "maxResults": max_results,
            "format": "json",
        }
        return self._get("/events/", params)

    # This method make possible to search events in a square are
    def search_events_in_area(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
        range_: str = "week",
        max_results: int = 100,
    ) -> Dict[str, Any]:

        params = {
            "range": range_,
            "selection": f"{lat1};{lon1};{lat2};{lon2}",
            "maxResults": max_results,
            "format": "json",
        }
        return self._get("/events/", params)

    # Search for services in a given area
    def search_services(
        self,
        lat: float,
        lon: float,
        categories: Optional[str] = None,
        max_dists: float = 1.0,
        max_results: int = 50,
        text: Optional[str] = None,
    ) -> Dict[str, Any]:

        params = {
            "selection": f"{lat};{lon}",
            "maxDists": max_dists,
            "maxResults": max_results,
            "format": "json",
        }
        if categories:
            params["categories"] = categories
        if text:
            params["text"] = text
        return self._get("/", params)

    def _get(self, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE_URL}{path}"
        headers = self._auth_headers()
        print(f"[KM4CityClient] - GET {url} params={params}")
        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
