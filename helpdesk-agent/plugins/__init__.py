"""Plugin discovery and registration."""

import importlib
import logging
from pathlib import Path

from plugins._protocol import Plugin

log = logging.getLogger(__name__)

_PLUGIN_PACKAGES = [
    "plugins.ingest",
    "plugins.analysis",
    "plugins.feedback",
    "plugins.faq",
    "plugins.responder",
    "plugins.export",
    "plugins.alerting",
]

_registry: list[Plugin] = []
_discovered = False


def discover_plugins(enabled: set[str] | None = None) -> list[Plugin]:
    """Import plugin packages and collect their exported `plugin` instances."""
    global _discovered
    _registry.clear()
    for pkg_name in _PLUGIN_PACKAGES:
        short = pkg_name.rsplit(".", 1)[-1]
        if enabled is not None and short not in enabled:
            log.info("Plugin '%s' disabled — skipping", short)
            continue
        try:
            mod = importlib.import_module(pkg_name)
            plugin: Plugin = getattr(mod, "plugin", None)
            if plugin is None:
                log.warning("Plugin package %s has no 'plugin' attribute", pkg_name)
                continue
            _registry.append(plugin)
            log.info("Loaded plugin '%s'", plugin.name)
        except Exception as e:
            log.error("Failed to load plugin %s: %s", pkg_name, e)
    _discovered = True
    return _registry


def get_plugins() -> list[Plugin]:
    global _discovered
    if not _discovered:
        discover_plugins()
    return list(_registry)


def register_plugins(app, enabled: set[str] | None = None) -> None:
    """Call register() on all discovered, enabled plugins."""
    from core.pipeline import get_plugin_config, is_plugin_enabled

    for plugin in discover_plugins(enabled):
        if not is_plugin_enabled(plugin.name):
            log.info("Plugin '%s' disabled — skipping register()", plugin.name)
            continue
        try:
            plugin.register(app, get_plugin_config(plugin.name))
            log.info("Registered plugin '%s'", plugin.name)
        except Exception:
            log.exception("Plugin '%s' register() failed", plugin.name)


def dispatch_on_ticket(ticket_key: str, event: str, payload: dict | None = None) -> None:
    """Notify all loaded plugins of a ticket event."""
    payload = payload or {}
    for plugin in get_plugins():
        if plugin.name == "analysis" and event == "classified":
            continue
        try:
            plugin.on_ticket(ticket_key, event, payload)
        except Exception as exc:
            log.error("Plugin '%s' on_ticket failed for %s: %s", plugin.name, ticket_key, exc)
