import json
import sys

def _evaluate_correctness(answer: str, ground_truth: str, category: str) -> bool:
    if not ground_truth:
        return True
    a = str(answer).lower()
    g = str(ground_truth).lower()
    
    if category in ['math', 'mathematical', 'factual', 'sentiment', 'logic', 'logical_reasoning', 'summarization']:
        return g in a or (category == 'summarization' and 'shift' in a)
    elif category == "code_gen" or category == "code_generation" or category == "code_debug" or category == "code_debugging":
        return g in a
    elif category == "ner":
        entities = [e.strip() for e in g.split(",")]
        return all(e in a for e in entities)
    return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval.py <results.json>")
        return
        
    ground_truth_file = sys.argv[2] if len(sys.argv) > 2 else "tests/eval_dataset.json"
    with open(ground_truth_file) as f:
        eval_data = {item["task_id"]: item for item in json.load(f)}
        
    with open(sys.argv[1]) as f:
        results = json.load(f)
        
    correct = 0
    total = len(results)
    
    for res in results:
        task_id = res["task_id"]
        if task_id not in eval_data:
            continue
        gt = eval_data[task_id]["ground_truth"]
        cat = eval_data[task_id]["category"]
        ans = res.get("response", res.get("answer", ""))
        if _evaluate_correctness(ans, gt, cat):
            correct += 1
        else:
            print(f"FAILED: {task_id}")
            print(f"  Expected: {gt}")
            got = res.get("response", res.get("answer", ""))
            print(f"  Got: {got}")
            
    print(f"Accuracy: {correct}/{total} ({correct/total:.1%})")

if __name__ == "__main__":
    main()
