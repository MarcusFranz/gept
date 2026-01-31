"""Regression test: verify all engine modules import without errors.

Guards against broken imports after dead code removal (e.g. issue #37)
and catches circular or missing dependency issues early.
"""

import importlib
import pkgutil

import pytest

import src

# Entrypoint modules with side effects (file handlers, server startup)
# that aren't safe to import in a test environment.
SKIP_MODULES = {"src.main"}


def _all_module_names():
    """Yield dotted module names for every sub-module under src/."""
    prefix = src.__name__ + "."
    for info in pkgutil.walk_packages(src.__path__, prefix=prefix):
        if info.name not in SKIP_MODULES:
            yield info.name


@pytest.mark.parametrize("module_name", list(_all_module_names()))
def test_import(module_name):
    importlib.import_module(module_name)
