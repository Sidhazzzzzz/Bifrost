import asyncio
import json
import os
import sys

from app.config import load_settings
from app.processor import run_batch

def _evaluate_correctness(answer: str, ground_truth: str, category: str) -> bool:
    if not ground_truth:
        return True
    a = str(answer).lower()
    g = str(ground_truth).lower()
    
    if category in ['math', 'mathematical', 'factual', 'sentiment', 'logic', 'logical_reasoning', 'summarization']:
        return g in a
    elif category in ["code_gen", "code_generation", "code_debug", "code_debugging"]:
        return g in a
    elif category == "ner":
        entities = [e.strip() for e in g.split(",")]
        return all(e in a for e in entities)
    return False

async def main():
    settings = load_settings()
    settings.input_path = "tests/gradingsets/very_easy.json"
    settings.output_path = "temp_results.json"
    
    with open(settings.input_path) as f:
        dataset = json.load(f)
        eval_data = {item["task_id"]: item for item in dataset}

    print("Running evaluation on fresh dataset...")
    report = await run_batch(settings)
    
    correct = 0
    total = len(dataset)

    # `report.results` contains `TaskResult` objects
    for res in report.results:
        task_id = res.task_id
        if task_id not in eval_data:
            continue
        item = eval_data[task_id]
        ground_truth = item["ground_truth"]
        category = item["category"]
        
        is_correct = _evaluate_correctness(res.response, ground_truth, category)
        if is_correct:
            correct += 1
        else:
            print(f"FAILED: {task_id}")
            print(f"  Prompt: {item['prompt']}")
            print(f"  Expected: {ground_truth}")
            print(f"  Got: {res.response}")
            print(f"  Model Used: {res.model_used}")
            print(f"  Routed To: {res.routed_to}")
            
    print(f"\nFresh-set Baseline Accuracy: {correct}/{total} ({correct/total:.1%})")

if __name__ == "__main__":
    asyncio.run(main())
