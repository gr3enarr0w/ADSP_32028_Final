"""Provider registry with optional SDKs loaded independently.

One configured HTTP-only provider (such as Tavily) must not be disabled just
because an unrelated optional SDK (such as ``exa_py``) is not installed.
"""

from .base import BaseProvider, SearchResult

ALL_PROVIDERS = {}

try:
    from .exa_provider import ExaProvider
    ALL_PROVIDERS["exa"] = ExaProvider
except ImportError:
    pass

try:
    from .brave_provider import BraveProvider
    ALL_PROVIDERS["brave"] = BraveProvider
except ImportError:
    pass

try:
    from .tavily_provider import TavilyProvider
    ALL_PROVIDERS["tavily"] = TavilyProvider
except ImportError:
    pass

try:
    from .gemini_provider import GeminiProvider
    ALL_PROVIDERS["gemini"] = GeminiProvider
except ImportError:
    pass

try:
    from .linkup_provider import LinkupProvider
    ALL_PROVIDERS["linkup"] = LinkupProvider
except ImportError:
    pass

try:
    from .newsdata_provider import NewsdataProvider
    ALL_PROVIDERS["newsdata"] = NewsdataProvider
except ImportError:
    pass
