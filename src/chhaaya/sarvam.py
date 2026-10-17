"""The Sarvam client: the only code that talks to api.sarvam.ai."""

import httpx2

API_URL = "https://api.sarvam.ai"
LANGUAGES = {"hi-IN": "hi", "en-IN": "en"}


class SarvamClient:
    def __init__(self, api_key: str, http: httpx2.Client | None = None) -> None:
        self.http = http or httpx2.Client(timeout=10)
        self.headers = {"api-subscription-key": api_key}

    def identify_language(self, text: str) -> tuple[str, str]:
        """The language ("hi" or "en") and script ("Deva" or "Latn") of typed text."""
        response = self.http.post(
            f"{API_URL}/text-lid", headers=self.headers, json={"input": text[:1000]}
        )
        response.raise_for_status()
        found = response.json()
        language = LANGUAGES.get(found["language_code"])
        if language is None or found["script_code"] not in ("Deva", "Latn"):
            raise ValueError(
                f"unsupported language {found['language_code']} "
                f"in script {found['script_code']}"
            )
        return language, found["script_code"]
