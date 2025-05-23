from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from client import MCPClient
from adapter import ModelAdapter, OllamaAdapter, OpenAIAdapter
import pytest
from constants import COMMANDS
import os

SYSTEM_PROMPTS = ["01.jinja", "02.jinja", "03.jinja"]

MODEL_ADAPTERS = [
    (OllamaAdapter, {"host": "http://localhost:11434", "model_name": "gemma3:4b"}),
    (OpenAIAdapter, {"api_key": os.environ.get("OPENAI_API_KEY"), "model_name": "gpt-4-turbo"}),
]

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.95, # MCP Client responses should be close to 100% correct
)

@pytest.mark.asyncio
@pytest.mark.parametrize("system_prompt", SYSTEM_PROMPTS)
@pytest.mark.parametrize("adapter_cls,adapter_kwargs", MODEL_ADAPTERS)
@pytest.mark.parametrize(
    "user_input,expected_output",
    [
        ("List all projects I have access to", "list-projects"),
        ("Show billing info for my current project", "get-billing-info"),
        ("List all GKE clusters in europe-west1", "list-gke-clusters --location europe-west1"),
        ("List all sql instances", "list-sql-instances"),
        (
            "Use project abc-123 for remaining requests",
            "select-project --projectId abc-123",
        ),
        ("Use project xyz-456", "select-project --projectId xyz-456"),
    ],
)
async def test_correctness(system_prompt, adapter_cls, adapter_kwargs, user_input, expected_output):
    client = MCPClient(system_prompt_template=system_prompt)
    model = ModelAdapter(adapter_cls(**adapter_kwargs))
    output = await client.translate_to_gcpmcp_command(user_input, COMMANDS, model)
    test_case = LLMTestCase(
        input=user_input,
        actual_output=output,
        expected_output=expected_output,
    )
    assert_test(test_case, [correctness_metric])
    # print(
    #     f"Prompt: {system_prompt}, Model: {adapter_cls.__name__}, Input: '{user_input}'\n"
    #     f"Expected: '{expected_output}', Actual: '{output}', Result: {result}\n"
    # )
