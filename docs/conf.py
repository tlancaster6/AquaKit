"""Sphinx configuration for project documentation."""

import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

# Add project source to path for autodoc
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Project information
project = "Aquacore"
copyright = f"{datetime.now().year}, Tucker Lancaster"
author = "Tucker Lancaster"
try:
    release = version("aquacore")
except PackageNotFoundError:
    release = "0.1.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "nbsphinx",
    "sphinxcontrib.mermaid",
]

# Templates and static files
templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# nbsphinx configuration
nbsphinx_execute = "never"  # Use committed outputs, don't re-execute
nbsphinx_allow_errors = False
nbsphinx_requirejs_path = ""  # Avoid RequireJS conflicts

# HTML output
html_theme = "furo"
html_title = "Aquacore"
html_static_path = []

# Furo theme options
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]

# Napoleon configuration
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Autodoc configuration
autodoc_member_order = "bysource"
autodoc_typehints = "description"
