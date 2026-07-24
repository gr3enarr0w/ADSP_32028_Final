from .base import SearchResult, BaseProvider
from .exa_provider import ExaProvider
from .brave_provider import BraveProvider
from .tavily_provider import TavilyProvider
from .gemini_provider import GeminiProvider
from .linkup_provider import LinkupProvider
from .newsdata_provider import NewsdataProvider
from .apify_provider import ApifyProvider

ALL_PROVIDERS = {
    "exa": ExaProvider,
    "brave": BraveProvider,
    "tavily": TavilyProvider,
    "gemini": GeminiProvider,
    "linkup": LinkupProvider,
    "newsdata": NewsdataProvider,
}
