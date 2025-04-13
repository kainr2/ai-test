import requests

from langchain.agents import Tool


class DuckDuckGoSearch:
    DUCKDUCKGO_API_URL = "https://api.duckduckgo.com/"

    # Custom function to query DuckDuckGo API
    def search(self, query: str) -> str:
        params = {
            "q": query,
            "format": "json",  # Return results in JSON format
            "no_html": 1,      # Remove HTML from results
            "skip_disambig": 1 # Skip disambiguation (if any)
        }
        response = requests.get(self.DUCKDUCKGO_API_URL, params=params).json()
        # Retrieve and return the abstract text or description
        return response.get("AbstractText", "No information available.")

    def create_tool(self) -> Tool:
        return Tool(
            name="DuckDuckGoSearch",
            func=self.search,
            description="Use this tool to search the web using DuckDuckGo API"
        )
