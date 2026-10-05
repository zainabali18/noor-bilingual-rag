"""Optional LLM-as-judge answer evaluation (faithfulness + relevancy).

Retrieval metrics (evaluate.py) run fully offline. Judging the *generated
answer* needs a model, so this module only runs when LLM_PROVIDER is set to
openai. For each question it generates an answer from the retrieved context,
then asks the model to score, on a 1-5 scale:

  * faithfulness - is every claim supported by the provided context?
  * relevancy    - does the answer actually address the question?

This is a lightweight, dependency-free take on the idea behind RAGAS.

Run:  LLM_PROVIDER=openai python -m eval.answer_eval
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from noor.config import config
from noor.generator import _format_context
from noor.pipeline import RAGPipeline

EVAL_PATH = Path(__file__).parent / "eval_set.yaml"

JUDGE_PROMPT = """You are grading a question-answering system.

Question:
{question}

Context the system was given:
{context}

System answer:
{answer}

Score the answer from 1 to 5 on two axes and reply with STRICT JSON only:
{{"faithfulness": <1-5>, "relevancy": <1-5>, "reason": "<one short sentence>"}}

faithfulness: 5 = every claim is supported by the context, 1 = largely unsupported.
relevancy: 5 = directly answers the question, 1 = off-topic."""


def _judge(question, context, answer):
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    provider = config.llm_provider.lower()
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=config.openai_model, temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content
    else:
        raise SystemExit("Set LLM_PROVIDER=openai to run answer evaluation.")
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    return json.loads(match.group(0)) if match else {"faithfulness": 0, "relevancy": 0, "reason": "unparseable"}


def main():
    if config.llm_provider.lower() == "none":
        raise SystemExit("Answer evaluation needs an LLM. Set LLM_PROVIDER and a key.")
    questions = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))["questions"]
    pipe = RAGPipeline()

    faith, rel = 0.0, 0.0
    for item in questions:
        ans = pipe.answer(item["q"])
        context = _format_context(ans.passages)
        scores = _judge(item["q"], context, ans.text)
        faith += scores["faithfulness"]
        rel += scores["relevancy"]
        print(f"[{item['lang']}] faith={scores['faithfulness']} "
              f"rel={scores['relevancy']}  {item['q'][:55]}")

    n = len(questions)
    print(f"\nMean faithfulness: {faith / n:.2f} / 5")
    print(f"Mean relevancy:    {rel / n:.2f} / 5")


if __name__ == "__main__":
    main()
