import os
import sys
import random

import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Dict
import numpy as np
from tqdm import tqdm

import openai
from openai.types.chat.chat_completion import Choice

def set_openai_api_key():
    api_key = "API_KEY"
    openai.api_key = api_key
    return api_key


class OpenAIModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.Client(api_key=set_openai_api_key())

    def evaluate_test_oracle(self, target_sample: dict, fewshot_examples: List[dict],
                             fewshot_rubric, template: str):
        # Build the prompt
        prompt_blocks = [template]

        # Add few-shot examples
        prompt_blocks.append("### LEARN FROM THESE EXAMPLES")
        prompt_blocks.append("A software developer has labeled these cases. Study their judgment patterns:\n")

        for idx, ex in enumerate(fewshot_examples, 1):
            example_data = {
                "test_oracle": ex["test_oracle"],
                "actual_outputs": ex["actual_outputs"],
            }
            # This correctly uses the integer label from the data
            prompt_blocks.append(
                dedent(f"""
                ### EXAMPLE {idx}
                {json.dumps(example_data, indent=2)}
                Developer's judgment:
                {{"label": {ex['label']}}}
                """).strip()
            )

        # Add target case
        target_data = {
            "test_oracle": target_sample["test_oracle"],
            "actual_outputs": target_sample["actual_outputs"]
        }
        prompt_blocks.append(
            dedent(f"""
            ### YOUR TASK
            Now apply the same judgment style you learned from the examples above.
            Focus on `actual_outputs` as the primary indicator, and reason like the
            developer would. Return *only* the JSON object – nothing else.
            {json.dumps(target_data, indent=2)}
            """).strip()
        )

        full_prompt = "\n\n".join(prompt_blocks)

        # Call the model
        prompt_config = {
            "model": self.model_name,
            "max_completion_tokens": 1024,
            "logprobs": True,
            "temperature": 0,
            "top_logprobs": 20,
            "response_format": {"type": "json_object"}
        }

        output = self.prompt_generation(
            "You are a helpful assistant that outputs valid JSON.",
            full_prompt,
            prompt_config
        )

        if output is None:
            return None, (None, None), None

        try:
            response_text = output.message.content.strip()
            response_json = json.loads(response_text)
            predicted_label = response_json.get("label", None) # This will be 0 or 1
            reason = response_json.get("reason", "")

            prob_fail, prob_pass = self._extract_label_probs(output)

            return predicted_label, (prob_fail, prob_pass), reason

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error parsing model output: {e}")
            print(f"Response: {output.message.content if output else 'None'}")
            return None, (None, None), None

    def _extract_label_probs(self, output: Choice) -> Tuple[float, float]:
        if not output.logprobs or not output.logprobs.content:
            return (0.5, 0.5) # Fallback

        FAIL_TOKENS = ['0', 'FAIL']
        PASS_TOKENS = ['1', 'PASS']
        ALL_LABEL_TOKENS = FAIL_TOKENS + PASS_TOKENS

        for idx, token_logprob in enumerate(output.logprobs.content):
            token = token_logprob.token.strip()

            if token in ALL_LABEL_TOKENS:
                top_logprobs = token_logprob.top_logprobs

                if not top_logprobs:
                    continue

                prob_pass = 0.0
                prob_fail = 0.0

                for top_lp in top_logprobs:
                    top_token_str = top_lp.token.strip()

                    if top_token_str in PASS_TOKENS:
                        prob_pass += math.exp(top_lp.logprob)
                    elif top_token_str in FAIL_TOKENS:
                        prob_fail += math.exp(top_lp.logprob)

                return (prob_fail, prob_pass)

        # Fallback
        return (0.5, 0.5)

    def simulate_test_oracle_annotators(self, target_sample: dict,
                                        fewshot_examples_list: List[List[dict]],
                                        fewshot_rubric_list,
                                        template: str):
        probs_pass_list = []
        probs_fail_list = []
        labels = []
        reasons = []

        for fewshot_examples, fewshot_rubric in zip(fewshot_examples_list, fewshot_rubric_list):
            label, (prob_fail, prob_pass), reason = self.evaluate_test_oracle(
                target_sample, fewshot_examples, fewshot_rubric, template
            )

            if label is not None and prob_pass is not None and prob_fail is not None:
                labels.append(label)
                # This correctly averages the raw probabilities for "pass"
                # and "fail" signals, as per the paper's method.
                probs_pass_list.append(prob_pass)
                probs_fail_list.append(prob_fail)
                reasons.append(reason)

        if len(probs_pass_list) == 0:
            return None, None, []

        avg_prob_pass = np.mean(probs_pass_list)
        avg_prob_fail = np.mean(probs_fail_list)

        predicted_label = 1 if avg_prob_pass > avg_prob_fail else 0

        return [avg_prob_fail, avg_prob_pass], predicted_label, reasons

    def _request_generation(self, system_prompt: str, prompt: str, prompt_config: Dict) -> Choice:
        completion = self.client.chat.completions.create(
            **prompt_config,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0]

    def prompt_generation(self, system_prompt: str, prompt: str, prompt_config: Dict) -> Choice | None:
        """
        Send prompt to OpenAI API with retry logic.
        """
        try:
            inferred_answer = self._request_generation(system_prompt, prompt, prompt_config)
            return inferred_answer

        except StopIteration as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 1
            print(f"Server Time out. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            return self._request_generation(system_prompt, prompt, prompt_config)

        except openai.BadRequestError as e:
            print(f"Bad Request Error: {e}")
            return None

        except openai.OpenAIError as e:
            retry_time = e.retry_after if hasattr(e, 'retry_after') else 5
            print(f"OpenAI Error. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            try:
                return self._request_generation(system_prompt, prompt, prompt_config)
            except:
                return None

        except OSError as e:
            retry_time = getattr(e, 'retry_after', 5)
            print(f"OS Error. Retrying in {retry_time} seconds...")
            time.sleep(retry_time)
            try:
                return self._request_generation(system_prompt, prompt, prompt_config)
            except:
                return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


def save_to_file(sample_list: List[dict], out_filename: str | Path, save_mode: str = 'w'):
    """Save samples to a JSONL file."""
    assert save_mode in ['w', 'a'], "Save mode should be either `w` or `a`."
    if len(sample_list) == 0:
        return
    sample_str_list = [json.dumps(sample) for sample in sample_list]
    with open(out_filename, save_mode) as f:
        f.write("\n".join(sample_str_list) + "\n")


def load_and_merge_annotators(annotator_files: List[Path]) -> List[Dict]:
    """
    Load 3 annotator files and merge using majority vote.

    Args:
        annotator_files: List of 3 paths to annotator files

    Returns:
        List of 100 samples with majority vote labels
    """
    print("Loading annotator files...")

    # Load all 3 annotator files
    all_data = []
    for file_path in annotator_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        all_data.append(data)
        print(f"  Loaded {len(data)} samples from {file_path.stem}")

    # Verify all have same number of samples
    if len(set(len(d) for d in all_data)) > 1:
        raise ValueError("Annotator files have different numbers of samples!")

    # Merge using majority vote
    print("\nMerging with majority vote...")
    merged_data = []

    for anno1, anno2, anno3 in zip(all_data[0], all_data[1], all_data[2]):
        # Compute majority vote
        votes = [anno1['label'], anno2['label'], anno3['label']]
        majority_label = 1 if sum(votes) >= 2 else 0

        # Create merged sample
        merged_sample = {
            'actual_input': anno1['actual_input'],
            'actual_outputs': anno1['actual_outputs'],
            'test_oracle': anno1['test_oracle'],
            'label': majority_label
        }
        merged_data.append(merged_sample)

    print(f"Merged into {len(merged_data)} samples with majority vote labels")

    # Print label distribution
    num_pass = sum(1 for s in merged_data if s['label'] == 1)
    print(f"  Pass (label=1): {num_pass} ({num_pass/len(merged_data)*100:.1f}%)")
    print(f"  Fail (label=0): {len(merged_data)-num_pass} ({(len(merged_data)-num_pass)/len(merged_data)*100:.1f}%)")

    return merged_data


def create_simulated_annotators_fewshot(calibration_data: List[Dict],
                                        K: int = 5,
                                        N: int = 5) -> List[List[Dict]]:
    """
    Create N simulated annotators by sampling K examples from calibration data.

    Following the paper: Each simulated annotator gets K random samples.

    Args:
        calibration_data: List of 100 calibration samples (with majority vote labels)
        K: Number of few-shot examples per simulated annotator
        N: Number of simulated annotators to create

    Returns:
        List of N few-shot example lists
    """
    print(f"\nCreating {N} simulated annotators with K={K} examples each...")

    fewshot_examples_list = []

    for i in range(N):
        # Sample K examples randomly from calibration data
        K_actual = min(K, len(calibration_data))
        fewshot_examples = random.sample(calibration_data, K_actual)
        fewshot_examples_list.append(fewshot_examples)

        print(f"  Simulated annotator {i+1}: {K_actual} examples sampled")

    return fewshot_examples_list


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-4.1",
                        choices=["LLM1", "LLM2"],
                        help="Model name")
    parser.add_argument("--annotator_files", type=str, nargs="+", default=[
        "ANNOTATION_RESULTS1",
        "ANNOTATION_RESULTS2",
        "ANNOTATION_RESULTS3"
    ], help="List of 3 annotator calibration files")
    
    parser.add_argument("--K", type=int, default=5,
                        help="Number of few-shot examples per simulated annotator")
    parser.add_argument("--N", type=int, default=5,
                        help="Number of simulated annotators")
    parser.add_argument("--output_dir", type=str,
                        default="/home/fanyuw/fanyu/LLM_acc_testing/dataset/cali_out",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    # Map to full model names
    model_map = {
        "gpt-4.1-nano": "gpt-4.1-nano-2025-04-14",
        "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
        "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
        "gpt-4.1": "gpt-4.1-2025-04-14"
    }
    args.openai_model_name = model_map[args.model_name]

    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    args.out_filename = args.output_dir / f"{args.model_name}.evaluation_prob2.jsonl"

    print(f"{'='*60}")
    print(f"Calibration Prediction Generation")
    print(f"{'='*60}")
    print(f"Model: {args.openai_model_name}")
    # print(f"Annotator files: {len(args.annotator_files)}")
    print(f"K (examples per annotator): {args.K}")
    print(f"N (simulated annotators): {args.N}")
    print(f"Output: {args.out_filename}")
    print(f"{'='*60}\n")

    return args


if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Initialize model
    print("Initializing model...")
    model = OpenAIModel(model_name=args.openai_model_name)

    # Load and merge annotator files using majority vote
    annotator_files = [Path(f) for f in args.annotator_files]
    calibration_data = load_and_merge_annotators(annotator_files)

    # Create simulated annotators (for few-shot prompting)
    fewshot_examples_list = create_simulated_annotators_fewshot(
        calibration_data,
        K=args.K,
        N=args.N
    )
    fewshot_rubric_list = [[] for _ in range(len(fewshot_examples_list))]

    # Remove few-shot examples from calibration samples to avoid leakage
    fewshot_samples_set = set()
    for fewshot_list in fewshot_examples_list:
        for sample in fewshot_list:
            # Use a unique identifier (e.g., hash of sample content)
            sample_id = json.dumps(sample['ID'], sort_keys=True)
            fewshot_samples_set.add(sample_id)

    calibration_samples = []
    for sample in calibration_data:
        sample_id = json.dumps(sample['actual_input'], sort_keys=True)
        if sample_id not in fewshot_samples_set:
            calibration_samples.append(sample)

    print(f"\n{'='*60}")
    print(f"Evaluating {len(calibration_samples)} calibration samples...")
    print(f"(Excluded {len(fewshot_samples_set)} samples used in few-shot examples)")
    print(f"{'='*60}\n")

    results = []
    skipped = 0

    for sample in tqdm(calibration_samples, desc=f"Evaluating with {args.model_name}"):
        # Get model predictions using simulated annotators
        probs, predicted_label, reasons = model.simulate_test_oracle_annotators(
            sample, fewshot_examples_list, fewshot_rubric_list, TEMPLATE
        )

        if probs is not None and len(probs) > 0 and not np.any(np.isnan(probs)):
            # Store result
            result = {
                "actual_input": sample["actual_input"],
                "test_oracle": sample["test_oracle"],
                "actual_outputs": sample["actual_outputs"],
                "probs": probs,  # [prob(fail), prob(pass)]
                "predicted_label": predicted_label,
                "confidence": max(probs),
                "label": sample['label'],  # Ground truth = majority vote
                "model_name": args.model_name,
                "num_simulated_annotators": args.N
            }

            save_to_file([result], args.out_filename, save_mode="a")
            results.append(result)
        else:
            print(f"\nWarning: Skipped sample due to invalid prediction")
            skipped += 1

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Calibration Prediction Complete!")
    print(f"{'='*60}")
    print(f"Total calibration samples: {len(calibration_samples)}")
    print(f"Successfully evaluated: {len(results)}")
    print(f"Skipped: {skipped}")
    print(f"Results saved to: {args.out_filename}")

    if results:
        confidences = [r["confidence"] for r in results]
        predictions = [r["predicted_label"] for r in results]
        labels = [r["label"] for r in results]

        print(f"\nStatistics:")
        print(f"  Average confidence: {np.mean(confidences):.3f}")
        print(f"  Std confidence: {np.std(confidences):.3f}")
        print(f"  Predicted Pass (label=1): {sum(predictions)} ({sum(predictions)/len(predictions)*100:.1f}%)")
        print(f"  Ground truth Pass: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")

        # Accuracy
        correct = sum(1 for r in results if r['predicted_label'] == r['label'])
        accuracy = correct / len(results)
        print(f"  Accuracy: {accuracy*100:.1f}%")

        # Confidence calibration
        correct_confs = [r['confidence'] for r in results if r['predicted_label'] == r['label']]
        incorrect_confs = [r['confidence'] for r in results if r['predicted_label'] != r['label']]
        if correct_confs and incorrect_confs:
            print(f"  Avg confidence when correct: {np.mean(correct_confs):.3f}")
            print(f"  Avg confidence when incorrect: {np.mean(incorrect_confs):.3f}")
