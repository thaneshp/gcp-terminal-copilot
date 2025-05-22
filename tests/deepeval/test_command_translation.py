from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from client import MCPClient
from adapter import ModelAdapter, OllamaAdapter
import pytest
from constants import COMMANDS

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=1.0,  # MCP commands are deterministic
)

@pytest.fixture(scope="module")
def client():
    return MCPClient(system_prompt_template="01.jinja")


@pytest.fixture(scope="module")
def model():
    return ModelAdapter(OllamaAdapter("http://localhost:11434", "gemma3:4b"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "user_input,expected_output",
    [
        ("List all projects I have access to", "list-projects"),
        ("Show billing info for my current project", "get-billing-info"),
        ("List all GKE clusters in europe-west1", "list-gke-clusters"),
        (
            "Use project abc-123 for remaining requests",
            "select-project --projectId abc-123",
        ),
        # Add more (user_input, expected_output) pairs as needed
    ],
)
async def test_correctness(client, model, user_input, expected_output):
    output = await client.translate_to_gcpmcp_command(user_input, COMMANDS, model)

    test_case = LLMTestCase(
        input=user_input,
        actual_output=output,
        expected_output=expected_output,
    )
    assert_test(test_case, [correctness_metric])
