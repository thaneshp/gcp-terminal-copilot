from typing import Any
from jinja2 import Environment, FileSystemLoader, select_autoescape

def process_template(template_file: str, data: dict[str, Any]) -> str:
    """Process a Jinja2 template with the given data.

    Args:
        template_file (str): The name of the template file to process.
        data (dict): The data to render the template with.
    
    Returns:
        str: The rendered template as a string.
    """
    jinja_env = Environment(
        loader=FileSystemLoader(searchpath="../prompts/system"), autoescape=select_autoescape()
    )
    template = jinja_env.get_template(template_file)
    return template.render(**data)