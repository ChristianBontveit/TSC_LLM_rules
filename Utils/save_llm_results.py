import json
import os

def save_results(args, rule_txt, acc, preds, test_ts_labels, rand_ts_idx):
    rules_dict = {}
    curr_cls = None
    lines = rule_txt.split("\n")

    for line in lines:
        if line.startswith("Class"):
            curr_cls = line.replace(":", "").strip().lower().replace(" ", "_")
            rules_dict[curr_cls] = ""
        elif curr_cls:
            rules_dict[curr_cls] += line + "\n"

    run_data = {
        "dataset": args.dataset,
        "classifier": args.classifier,
        "llm": args.llm,
        "k": args.k,
        "num_rules": args.rules,
        "accuracy": acc,
        "extracted_rules": rules_dict,
        "instance": []
    }

    for idx, (true_l, pred_l, ts_idx) in enumerate(zip(test_ts_labels, preds, rand_ts_idx)):
        run_data["instance"].append({
            "instance_id": idx + 1,
            "ts_idx": int(ts_idx),
            "true_label": int(true_l),
            "predicted_label": int(pred_l),
            "status": "MATCH" if true_l == pred_l else "MISMATCH"
        })

    os.makedirs("results", exist_ok=True)

    with open("results/llm_rules_results.jsonl", "a") as f:
        f.write(json.dumps(run_data) + "\n")