"""The LLM project's registry entry, logged as "models from code" (MLflow's recommended form for custom models).

What is versioned is the assistant's configuration: llm_pipeline.py passes it as `model_config`, and
`predict` simply returns it. In production this file would hold the RAG pipeline itself
(search + prompt + model call); the registry, aliases and tags would work exactly the same way.
"""
import mlflow
from mlflow.models import set_model


class AssistantConfig(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.config = context.model_config

    def predict(self, context, model_input: list[str], params=None) -> list[dict]:
        return [self.config for _ in model_input]


set_model(AssistantConfig())
