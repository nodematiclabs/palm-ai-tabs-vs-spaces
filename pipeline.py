import kfp
import kfp.dsl as dsl

from kfp import compiler
from kfp.dsl import Dataset, Input, Output

from typing import List

MODEL = "text-bison@001"
TEMPERATURE = 0.2
TOKEN_LIMIT = 16
TOP_P = 0.8
TOP_K = 40

prompt = """Answer questions as if you are a {0}

input: {1}
output:
"""


@dsl.component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform", "appengine-python-standard"],
)
def respond_simulacrum(
    project_id: str,
    model_name: str,
    temperature: float,
    max_output_tokens: int,
    top_p: float,
    top_k: int,
    content: str,
    output: Output[Dataset],
    location: str = "us-central1"
):
    import json
    import vertexai
    from vertexai.preview.language_models import TextGenerationModel

    vertexai.init(project=project_id, location=location)
    model = TextGenerationModel.from_pretrained(model_name)
    response = model.predict(
        content,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_k=top_k,
        top_p=top_p,
    )

    with open(output.path, 'w') as f:
        f.write(response.text)

@dsl.component(base_image="python:3.11")
def count_word_instances(text: Input[Dataset], word: str) -> int:
    with open(text.path, 'r') as f:
        word = word.lower()
        words = f.read().lower().split()
        return words.count(word)

@dsl.component(base_image="python:3.11")
def compile_results(tabs_count: List[int], spaces_count: List[int]) -> str:
    if sum(tabs_count) > sum(spaces_count):
        return "tabs"
    else:
        return "spaces"

@dsl.pipeline(name="simulacra-consensus")
def transcript_extraction(project_id: str, question: str):
    with dsl.ParallelFor(
        name="simulacra-responses",
        items=[
            "senior software engineer",
            "junior software engineer",
            "software engineer",
            "lead software engineer",
            "devops engineer",
            "software engineering manager",
            "software architect",
            "frontend engineer",
            "backend engineer",
            "full stack engineer",
            "software development engineer in test",
            "quality assurance engineer",
            "devops engineer",
            "system engineer",
            "data engineer",
            "database administrator",
            "security engineer",
            "product manager",
            "project manager",
            "scrum master",
            "ui/ux designer",
            "data scientist",
            "machine learning engineer",
            "site reliability engineer",
            "network engineer",
            "cloud engineer",
            "mobile developer",
            "web developer",
            "web designer",
            "principal engineer",
            "director of engineering",
            "vp of engineering",
            "chief technology officer"
        ],
    ) as simulacrum:
        response = respond_simulacrum(
            project_id=project_id,
            model_name=MODEL,
            temperature=TEMPERATURE,
            max_output_tokens=TOKEN_LIMIT,
            top_p=TOP_P,
            top_k=TOP_K,
            content=prompt.format(simulacrum, question),
            location="us-central1"
        )
        tabs_count = count_word_instances(
            text=response.output,
            word="tabs",
        )
        spaces_count = count_word_instances(
            text=response.output,
            word="spaces",
        )
    compile_results(
        tabs_count=dsl.Collected(tabs_count.output),
        spaces_count=dsl.Collected(spaces_count.output),
    )

compiler.Compiler().compile(transcript_extraction, "pipeline.yaml")
