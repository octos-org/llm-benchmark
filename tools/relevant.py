"""The single tool that the benchmark prompt should select."""

from typing import Any, Dict

RELEVANT_TOOL: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Get current weather conditions for a given city, including "
            "temperature, humidity, wind speed, and a forecast for the "
            "next 24 hours from the live weather service."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city name, e.g. 'Tokyo', 'New York'.",
                },
                "units": {
                    "type": "string",
                    "enum": ["metric", "imperial"],
                    "description": "Temperature units to use.",
                },
            },
            "required": ["city"],
        },
    },
}
