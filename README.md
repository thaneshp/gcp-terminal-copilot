# GCP Terminal Copilot

A Python-based GCP CLI assistant that provides natural language processing capabilities for `gcloud` commands leveraging [GCP MCP Server](https://github.com/eniayomi/gcp-mcp).

## Prerequisites

- Python 3.11+ (as specified in pyproject.toml)
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and configured
- [Ollama](https://ollama.com) with a model downloaded
- [uv](https://github.com/astral-sh/uv) for Python package management
- [GCP MCP server installed and running](https://github.com/eniayomi/gcp-mcp)

## Installation

1. Clone the repo

    ```bash
    git clone https://github.com/thaneshp/gcp-terminal-copilot
    cd gcp-terminal-copilot
    ```

1. Open the terminal and Start a virtual env with uv

    ```bash
    uv venv
    ```

1. Install packages using uv

    ```bash
    uv pip install .
    ```

1. Run Ollama and make note of its local address
1. Run GCP MCP server and make note of its local path
1. Rename `.env-sample` to `.env`
1. Update with the values that correspond to your locally running Ollama, GCP MCP executable path, and model you want to use
1. Run `gcloud auth application-default login` to authenticate to you Google Cloud account
1. You can now run `python client.py`

## Troubleshooting

- Make sure your `gcloud` cli is logged in, GCP MCP uses that as auth

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
