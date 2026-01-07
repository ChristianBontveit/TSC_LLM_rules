import argparse
from dotenv import load_dotenv, dotenv_values
import numpy as np
from Utils.load_data import load_dataset, load_dataset_labels, normalize_data
import matplotlib.pyplot as plt
import base64
import io
import re
from Utils.load_models import model_batch_classify
from Utils.selectPrototypes import select_prototypes
import openai
from openai import AzureOpenAI
from openai import OpenAI 

openai.api_key = dotenv_values(".env")["API_KEY"]
openai.api_base = dotenv_values(".env")["API_BASE"]

DEBUG = False

def get_response(prompt: list[dict], model:str):
    client = OpenAI(
        base_url=dotenv_values(".env")["API_BASE"], 
        api_key=dotenv_values(".env")["API_KEY"],
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}     #type: ignore
        ],
          extra_body={"reasoning": {"enabled": True}}
    )
    return response.choices[0].message.content
    
def build_prompt(images: list[str], test_samples: list[str], num_labels: int, print_prompt:bool=False) -> list[dict]:
    labels = range(0, num_labels)
    num_images = len(images)
    num_images_per_label = int(num_images/2)
    
    classes = ", ".join(map(str, labels))
    prompt = [
        {"type": "text", "text": f"""You are a time-series classification expert. Your goal is to learn from a small set of labeled examples (classes {classes}) and then assign the correct class to a new, unlabeled time series. 
         Follow these steps:
         1.Carefully examine the {classes} examples of classes {classes} and identify their common patterns.
         2.Compare the new instance to your learned characteristics of classes {classes}.
         3.Provide a brief rationale for your decision.
         4.Conclude with one line stating only the predicted class for each one of the unlabeled examples.
         5. Only use the pattern (Predicted class: {classes}) for each one of the unlabeled examples exclusively in the final guess.
         Remember that I will always provide you with 10 unlabeled examples. Therefore, you need to perform exactly 10 predictions.
         The examples:"""}
    ]
    for label in labels:
        images_label = images[:num_images_per_label]
        images = images[num_images_per_label:]
        images_dict = [{"type": "text", "text": f"Class {label} examples ({num_images_per_label} time-series plots labeled “{label}”) "}] + \
                      [{"type": "image_url", "image_url": {"url": data_url}} for data_url in images_label]
        prompt.extend(images_dict) #type:ignore

    test = [{"type": "text", "text": "New instances to classify (unlabeled time-series plot)"}] + [{"type": "image_url", "image_url": {"url": data_url}} for data_url in test_samples] + [{"type": "text", "text": "Now, classify the new instances."}]
    prompt.extend(test) #type: ignore
    
    if INTERACTIVE:
        print("----------------------------------------------")
        for text in prompt:
            if text["type"] == "text":
                print(text["text"])
            elif text["type"] == "image_url":
                print("$IMAGE$")
        print("----------------------------------------------")
        input("Press Enter to continue...")
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

def prompt_model(llm_model: str, k_img: list[str], test_sample: list[str], test_ts_label: list[int], labels: int) -> float:
    prompt = build_prompt(k_img, test_samples= test_sample, num_labels=labels, print_prompt=DEBUG)

    response = get_response(prompt, llm_model)
    #print(response)

    pattern = r"Predicted class:\s+(\d+)"
    predicted_labels_str = re.findall(pattern, str(response))
    predicted_labels_int = [int(label) for label in predicted_labels_str]
    if len(predicted_labels_int) == 1:
        predicted_labels_int = np.full(10, predicted_labels_int[0])
    if len(predicted_labels_int) == 20:
        predicted_labels_int = predicted_labels_int[0:10]
    acc_length = min(len(test_ts_label), len(predicted_labels_int))
    accuracy = sum([1 if test_ts_label[i] == predicted_labels_int[i] else 0 for i in range(acc_length)])/acc_length
    
    if INTERACTIVE:
        print(f"------------------------------------------\n")
        print(response)
        print(f"Real label {test_ts_label}, predicted label {predicted_labels_int} = {accuracy}")
        print("\n------------------------------------------")
        input("Press Enter to continue...")
    #print(f"Real label {test_ts_label}, predicted label {predicted_labels_int} = {accuracy}")
    return accuracy

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Dataset to feed samples from.")
    parser.add_argument('--classifier', type=str, default="cnn", help="Classifier to compare with." )
    parser.add_argument('--llm', type=str, default="google/gemma-3-27b-it:free",help="LLM within the OpenAI API. Models supported: gpt4o, o4-mini, gpt-4.1 and o3")
    parser.add_argument('--k', type=int, default=3, help="Number of total examples to use.")
    parser.add_argument('--interactive', action='store_true', help='Make code interactive')
    return parser.parse_args()
    
if __name__ == '__main__':
    args = argparser()
    assert args.llm in ["gpt4o", "o4-mini", "gpt-4.1","o3", "google/gemma-3-27b-it:free"], "Invalid model type"
    print(f"Testing {args.dataset } for classifier {args.classifier} on LLM {args.llm}")

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
        out = prompt_model(args.llm, prot_img_simp, test_img_simp, test_ts_labels, len(set(prot_labels)))
        step_results.append(out)






