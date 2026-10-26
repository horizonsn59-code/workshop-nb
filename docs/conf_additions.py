# Add to docs/source/conf.py.   pip install myst-nb sphinx-thebe
extensions += ["myst_nb", "sphinx_thebe"]

# Render the notebook from its stored outputs; the docs build never executes it.
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]

thebe_config = {
    "repository_url": "https://github.com/<GITHUB_USER>/autouq-workshop-binder",
    "repository_branch": "main",
    "codemirror-theme": "default",  # light editor; sphinx-thebe's own default is dark
}
