import json
import os

def save_results(args, rule_txt, acc, preds, test_ts_labels, rand_ts_idx, repetition=None, support_examples=None):
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
        "mode": args.mode,
        "classifier": args.classifier,
        "llm": args.llm,
        "k": args.k,
        "num_rules": args.rules,
        "accuracy": acc,
        "extracted_rules": rules_dict,
        "instance": []
    }

    if repetition is not None:
        run_data["repetition"] = repetition

    if support_examples is not None:
        run_data["support_examples"] = support_examples

    for idx, (true_l, pred_l, ts_idx) in enumerate(zip(test_ts_labels, preds, rand_ts_idx)):
        run_data["instance"].append({
            "instance_id": idx + 1,
            "ts_idx": int(ts_idx),
            "true_label": int(true_l),
            "predicted_label": int(pred_l),
            "status": "MATCH" if true_l == pred_l else "MISMATCH"
        })

    output_dir = os.path.join("results", "llm_results")
    
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir,
        f"{args.dataset}_{args.mode}_{args.k}_{args.rules}_llm_results.jsonl",
    )

    with open(filename, "a") as f:
        f.write(json.dumps(run_data) + "\n")


def save_raw_outputs_txt(args, raw_outputs, repetition: int):
    output_dir = os.path.join("results", "llm_raw_outputs")
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir,
        f"{args.dataset}_{args.mode}_{args.k}_{args.rules}_run_{repetition}.txt",
    )

    with open(filename, "w") as f:
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"mode: {args.mode}\n")
        f.write(f"classifier: {args.classifier}\n")
        f.write(f"llm: {args.llm}\n")
        f.write(f"k: {args.k}\n")
        f.write(f"rules: {args.rules}\n")
        f.write(f"repetition: {repetition}\n\n")

        baseline_response = raw_outputs.get("baseline_response")
        if baseline_response is not None:
            f.write("=== Baseline Response ===\n")
            f.write(baseline_response)
            f.write("\n")

        rule_generation_response = raw_outputs.get("rule_generation_response")
        if rule_generation_response is not None:
            f.write("=== Rule Generation Response ===\n")
            f.write(rule_generation_response)
            f.write("\n\n")

        classification_batch_responses = raw_outputs.get("classification_batch_responses")
        if classification_batch_responses is not None:
            for batch in classification_batch_responses:
                f.write(
                    f"=== Classification Batch start={batch['batch_start']} size={batch['batch_size']} ===\n"
                )
                f.write(batch["response"])
                f.write("\n\n")
