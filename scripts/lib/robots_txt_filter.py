"""Access the vendored ``apply_robots_txt_filter`` submodule.

The submodule lives at ``modules/apply_robots_txt_filter`` (added via
``git submodule add``). It is a flat collection of scripts, not a pip-installed
package, so we load the file we need directly by absolute path with importlib.
This is independent of the current working directory and avoids polluting
``sys.path`` / clashing with same-named modules elsewhere.

Usage
-----
    from scripts.lib.robots_txt_filter import PIIFormatter

    formatter = PIIFormatter(remove_emails=True, remove_ips=True)

Requires ``datatrove`` to be installed in the environment (imported by the
submodule file). If the submodule is missing (cloned without
``--recurse-submodules``), a clear error explains how to fix it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

# Repo root = two levels up from this file (scripts/lib/ -> scripts/ -> repo/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_ROOT = _REPO_ROOT / "modules" / "apply_robots_txt_filter"
_PII_FILE = SUBMODULE_ROOT / "pii_formatter_simple.py"


def _load_module():
    if not _PII_FILE.exists():
        raise ModuleNotFoundError(
            f"Expected {_PII_FILE} but it was not found. "
            "Initialise the submodule with:\n"
            "    git submodule update --init --recursive"
        )
    spec = importlib.util.spec_from_file_location(
        "apply_robots_txt_filter.pii_formatter_simple", _PII_FILE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # may raise ImportError if datatrove missing
    return module


_pii_module = _load_module()

# Re-export the public symbol so callers just do:
#   from scripts.lib.robots_txt_filter import PIIFormatter
PIIFormatter = _pii_module.PIIFormatter

__all__ = ["PIIFormatter", "SUBMODULE_ROOT"]
