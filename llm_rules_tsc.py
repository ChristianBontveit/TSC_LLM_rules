import argparse
import io
import os
import re
import numpy as np
import base64
from dotenv import load_dotenv, dotenv_values
from Utils.load_models import model_batch_classify
from Utils.selectPrototypes import select_prototypes
from Utils.load_data import load_dataset, load_dataset_labels, normalize_data
import openai
from openai import OpenAI
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

load_dotenv()

api_key = os.getenv("API_KEY")
if api_key is None:
    raise ValueError("API_KEY not found, add it to .env file")

client = OpenAI(api_key=api_key)

DEBUG = False

def get_response(prompt: list[dict], model:str):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}], #type: ignore
        # extra_body={"reasoning": {"enabled": True}}
    )
    return response.choices[0].message.content

### NEW PROMPT BUILDERS FOR RULE EXTRACTION AND CLASSIFICATION WITH RULES
def build_rule_prompt(images: list[str], num_labels: int, n_rules: int):
    labels = range(0, num_labels)
    classes = ", ".join(map(str, labels))

    prompt = [
        {"type": "text",
        "text": f"""

        You are a time-series classification expert. You are given labeled examples for classes {classes}.

        There are three prototypes for each of the two classes (R, B).
        I want you to provide me , a human-understandable rule for each class.
        Output only the rules.
        Structure each rule so that it is composed of sub-rules enumerated as R1, R2.
        Each sub-rule only covers one condition.
        Use at most {n_rules} sub-rules per class.

        Use the following format for your answer:

        Class <class_label>:
        R1: ...
        R2: ...
        ...
        
        Class <class_label>:
        R1: ...
        R2: ...
        ...
        
        """}
        ]

    num_images = len(images)
    num_images_per_label = int(num_images / num_labels)

    for label in labels:
        images_label = images[:num_images_per_label]
        images = images[num_images_per_label:]

        prompt.append({"type": "text", "text": f"Class {label} examples:"})
        prompt.extend([
            {"type": "image_url", "image_url": {"url": img}}
            for img in images_label
        ])

    return prompt

def build_classification_prompt(rule: str, test_samples: list[str]):

    prompt = [
        {
        "type": "text",
        "text": f"""
        You are given the following decision rules:
        
        {rule}
        
        Apply this rule to classify the 10 new time-series plots below.
        
        Instructions:
        - Follow the rule strictly.
        - Do not invent new criteria.
        - Provide the final classification for each of the 10 instances.
        - For each prediction, use the exact format: 'Predicted class: X'
        """}
        ]

    prompt.extend([
        {"type": "image_url", "image_url": {"url": img}}
        for img in test_samples
    ])

    return prompt

def get_idx_per_cls(labels_ts: np.ndarray, k_cls:int) -> dict[str,list[int]]:
    labels = np.unique(labels_ts)
    idx_labels = {}
    for label in labels:
        labels_idx = np.where(labels_ts == label)[0]
        rand_idx = np.asanyarray(np.random.randint(labels_idx.shape[0], size=(k_cls)))
        idx_labels[label] = rand_idx

    return idx_labels

def get_k_examples(dataset_ts: np.ndarray, k_idx:dict) -> np.ndarray:
    labels = k_idx.keys()
    k_examples = []
    for label in labels:
        idx = k_idx[label]
        k_examples_label = dataset_ts[idx]
        k_examples.append(k_examples_label)

    return np.array(k_examples)

def ts_to_image(ts: np.ndarray, show_fig: bool = False, name: str = ""):
    plt.figure(figsize=(4,3))
    plt.plot(ts); plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf)
    if show_fig:
        plt.savefig(f"./llm_tests/{name}")
        plt.pause(1)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_b64}"


def simp_ts_to_img(dataset_ts: np.ndarray, dataset_ts_labels: list[int], test_ts: np.ndarray) -> tuple[list[str], list[str]]:
    dataset_ts = dataset_ts
    dataset_ts_labels = dataset_ts_labels
    test_ts = test_ts
    
    k_img = [ts_to_image(ts, show_fig=DEBUG, name=f"train_{i}") for i, ts in enumerate(dataset_ts)]
    test_sample = [ts_to_image(ts, show_fig=DEBUG, name=f"test_{i}") for i, ts in enumerate(test_ts)]

    return k_img, test_sample

### NEW FUNCTIONS FOR RULE EXTRACTION AND CLASSIFICATION WITH RULES
def extract_rule(llm_model: str, k_img: list[str], labels: int, n_rules: int):
    prompt = build_rule_prompt(k_img, labels, n_rules)
    rule = get_response(prompt, llm_model)
    return rule

def classify_with_rule(llm_model: str, rule: str, test_img: list[str], test_labels: list[int], labels: int):
    prompt = build_classification_prompt(rule, test_img)
    response = get_response(prompt, llm_model)

    pattern = r"Predicted class:\s+(\d+)"
    predicted_labels = [int(x) for x in re.findall(pattern, response)]

    acc = sum(
        1 for i in range(len(test_labels))
        if i < len(predicted_labels) and predicted_labels[i] == test_labels[i]
    ) / len(test_labels)

    return acc, predicted_labels, response

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="cnn", help="Classifier to compare with." )
    parser.add_argument('--llm', type=str, default="google/gemma-3-27b-it:free",help="LLM within the OpenAI API. Models supported: gpt4o, o4-mini, gpt-4.1 and o3")
    parser.add_argument('--k', type=int, default=3, help="Number of total examples to use.")
    parser.add_argument('--rules', type=int, default=2, help="Number of LLM generated classification rules")
    parser.add_argument('--interactive', action='store_true', help='Make code interactive')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = argparser()
    assert args.llm in ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "o3", "gpt-5.1", ], "Invalid model type"
    print(f"Testing {args.dataset } for classifier {args.classifier} on LLM {args.llm} using {args.rules} LLM rules")
    global INTERACTIVE
    INTERACTIVE = True if args.interactive else False
    steps = 1
        
    train_ts_norm = load_dataset(args.dataset, data_type="TRAIN_normalized")
    prototipes_ts_norm = select_prototypes(args.dataset, num_instances=args.k, data_type="TRAIN_normalized") 
    prot_labels = np.array(load_dataset_labels(args.dataset, data_type='TEST_normalized'))
    test_ts_norm = load_dataset(args.dataset, data_type="TEST_normalized")
    rand_ts_idx = np.random.randint(0, test_ts_norm.shape[0], size=(10))
    test_ts_norm = test_ts_norm[rand_ts_idx]
    
    classifier_file = f"{args.classifier}_norm.pth" if args.classifier == "cnn" else f"{args.classifier}_norm.pkl"
    dataset_ts_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", prototipes_ts_norm, len(set(prot_labels)))   #type: ignore
    test_ts_labels = model_batch_classify(f"./models/{args.dataset}/{classifier_file}", test_ts_norm, len(set(prot_labels)))  #type: ignore
    
    step_results = []
    for i in range(steps):
        prot_img_simp, test_img_simp = simp_ts_to_img(prototipes_ts_norm, dataset_ts_labels, test_ts_norm)
        # extract rule
        rule = extract_rule(args.llm, prot_img_simp, len(set(prot_labels)), args.rules)
        
        print("Extracted Rule:\n", rule)
        
        # classify using rule
        accuracy, preds, raw_output = classify_with_rule(
            args.llm,
            rule,
            test_img_simp,
            test_ts_labels,
            len(set(prot_labels))
        )

        print("\n")
        
        # table
        print(f"{'Instance':<10} | {'True Label':<10} | {'Predicted':<10} | {'Status':<10} | {'TS Idx':<10}")
        print("-" * 70)
        for idx, (true_l, pred_l, ts_idx) in enumerate(zip(test_ts_labels, preds, rand_ts_idx)):
            status = "MATCH" if true_l == pred_l else "MISMATCH"
            print(f"{idx+1:<10} | {true_l:<10} | {pred_l:<10} | {status:<10} | {ts_idx:<10}")
        
        print("-" * 70)
        print(f"Final Accuracy: {accuracy * 100}%")