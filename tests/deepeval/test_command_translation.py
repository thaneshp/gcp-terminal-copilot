from typing import List
from deepeval import evaluate
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from client import MCPClient
from adapter import ModelAdapter
from constants import COMMANDS
from config import SYSTEM_PROMPTS, MODEL_ADAPTERS
import asyncio

TEST_INPUTS = [
    ("List all projects I have access to", "list-projects"),
    ("What projects can I access?", "list-projects"),
    ("Use project abc-123 for remaining requests", "select-project --projectId abc-123"),
    ("Use project xyz-456", "select-project --projectId xyz-456 --region us-central1"),
    ("Use project abc-123 in australia-southeast1", "select-project --projectId abc-123 --region australia-southeast1"),
    ("Show billing info for my current project", "get-billing-info"),
    ("Get billing info for project abc-123", "get-billing-info --projectId abc-123"),
    ("What is the cost forecast for the next 3 months?", "get-cost-forecast"),
    ("Get cost forecast for the next 6 months", "get-cost-forecast --months 6"),
    ("Get billing budget for my current project", "get-billing-budget"),
    ("List all GKE clusters in europe-west1", "list-gke-clusters --location europe-west1"),
    ("List all sql instances", "list-sql-instances"),
    ("Show me the last 10 log entries from my project", "get-logs --pageSize 10"),
]

correctness_metric = GEval(
    name="Correctness",
    criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    threshold=0.95,  # MCP Client responses should be close to 100% correct
)


def construct_test_cases(adapter_cls, adapter_kwargs, system_prompt) -> List[LLMTestCase]:
    """Construct test cases for testing command translation."""
    client = MCPClient(system_prompt_template=system_prompt)
    model = ModelAdapter(adapter_cls(**adapter_kwargs))
    test_cases = []
    for user_input, expected_output in TEST_INPUTS:
        actual_output = asyncio.run(client.translate_to_gcpmcp_command(user_input, COMMANDS, model))
        test_cases.append(
            LLMTestCase(
                input=user_input,
                actual_output=actual_output,
                expected_output=expected_output,
            )
        )
    return test_cases


def test_command_translation_batch():
    """Test command translation using different models and system prompts."""
    for adapter_cls, adapter_kwargs in MODEL_ADAPTERS:
        for system_prompt in SYSTEM_PROMPTS:
            evaluate(
                test_cases=construct_test_cases(adapter_cls, adapter_kwargs, system_prompt),
                metrics=[correctness_metric],
                hyperparameters={
                    "model": adapter_kwargs.get("model_name"),
                    "prompt template": system_prompt,
                },
            )
