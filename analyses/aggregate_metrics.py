import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional
import statistics


# ==========================================
# PAPER TYPE MAPPING
# ==========================================
PAPER_TYPES = {
    "1272": "method",
    "2124": "both",
    "1687": "both",
    "4366": "both",
    "0421": "method",
    "0896": "method",
    "2666": "method",
    "4160": "method",
    "2515": "method",
    "0666": "method",
}


# ==========================================
# IMPORT FUNCTIONS FROM metrics_cont.py
# ==========================================
def get_confusion_counts(y_true, y_pred):
    """Calculate confusion matrix counts from boolean predictions."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    return tp, tn, fp, fn


def mcc_from_counts(tp, tn, fp, fn):
    """Calculate Matthews Correlation Coefficient from confusion matrix counts."""
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / math.sqrt(denom)


def accuracy_from_counts(tp, tn, fp, fn):
    """Calculate accuracy (percent agreement) from confusion matrix counts."""
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0


def extract_analysis_booleans(data, section_name):
    """Extracts criterion_id -> present (bool) for Paper and Dataset arrays."""
    return {
        item["criterion_id"]: item["present"] for item in data.get(section_name, [])
    }


def extract_analysis_booleans_by_category(data, section_name):
    """Extracts booleans grouped by category for Paper/Dataset analysis."""
    by_category = {}
    for item in data.get(section_name, []):
        category = item.get("category", "unknown")
        if category not in by_category:
            by_category[category] = {}
        by_category[category][item["criterion_id"]] = item["present"]
    return by_category


def extract_code_booleans(data):
    """
    Extracts satisfaction booleans from code_analysis using same evaluation
    logic as compute_reproducibility_score.
    """
    bools = {}
    code_sec = data.get("code_analysis", {})

    if not code_sec:
        return bools

    # Extract methodology info for context
    methodology = code_sec.get("methodology", {})
    requires_training = methodology.get("requires_training", True)
    requires_datasets = methodology.get("requires_datasets", True)

    # 1. Structure - has_requirements
    structure = code_sec.get("structure", {})
    if "has_requirements" in structure:
        bools["structure.has_requirements"] = structure["has_requirements"]
    if "requirements_match_imports" in structure:
        match = structure["requirements_match_imports"]
        bools["structure.requirements_match_imports"] = match is not False

    # 2. Components - code availability
    components = code_sec.get("components", {})
    if "has_training_code" in components:
        bools["components.has_training_code"] = components["has_training_code"]
    if "has_evaluation_code" in components:
        bools["components.has_evaluation_code"] = components["has_evaluation_code"]
    if "has_documented_commands" in components:
        bools["components.has_documented_commands"] = components[
            "has_documented_commands"
        ]

    # 3. Artifacts
    artifacts = code_sec.get("artifacts", {})
    if "has_checkpoints" in artifacts:
        bools["artifacts.has_checkpoints"] = artifacts["has_checkpoints"]
    if "has_dataset_links" in artifacts:
        bools["artifacts.has_dataset_links"] = artifacts["has_dataset_links"]
    if "dataset_coverage" in artifacts and requires_datasets:
        coverage = artifacts["dataset_coverage"]
        bools["artifacts.dataset_coverage_full"] = coverage == "full"

    # 4. Dataset Splits
    splits = code_sec.get("dataset_splits", {})
    if "splits_specified" in splits:
        bools["splits.splits_specified"] = splits["splits_specified"]
    if "splits_provided" in splits:
        bools["splits.splits_provided"] = splits["splits_provided"]
    if "random_seeds_documented" in splits:
        bools["splits.random_seeds_documented"] = splits["random_seeds_documented"]

    # 5. Documentation
    documentation = code_sec.get("documentation", {})
    if "has_readme" in documentation:
        bools["documentation.has_readme"] = documentation["has_readme"]
    if "has_results_table" in documentation:
        bools["documentation.has_results_table"] = documentation["has_results_table"]
    if "has_reproduction_commands" in documentation:
        bools["documentation.has_reproduction_commands"] = documentation[
            "has_reproduction_commands"
        ]

    return bools


def compute_paper_metrics(paper_id: str, paper_type: str) -> Optional[Dict]:
    """Compute metrics for a single paper."""

    # Define weights based on paper type
    if paper_type == "both":
        WEIGHTS = {"models": 0.35, "datasets": 0.35, "experiments": 0.35}
    else:  # method
        WEIGHTS = {"models": 0.45, "datasets": 0.20, "experiments": 0.35}

    # Define dataset weights
    DATASET_WEIGHTS = {
        "data_collection": 0.35,
        "annotation": 0.40,
        "ethics_availability": 0.25,
    }

    # Load data files 78k:x=1M:2.5 20 cent in 3 cent out = 23 cent gpt4o 33+8=41 cent

    try:
        with open(f"{paper_id}_human_5.json", "r") as f:
            human_data = json.load(f)
        with open(f"{paper_id}_ps_5.json", "r") as f:
            ps_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Warning: Files not found for paper {paper_id}: {e}")
        return None

    # Separate Paper analysis by category
    paper_by_category_human = extract_analysis_booleans_by_category(
        human_data, "paper_analysis"
    )
    paper_by_category_ps = extract_analysis_booleans_by_category(
        ps_data, "paper_analysis"
    )

    # Separate Dataset analysis by category
    dataset_by_category_human = extract_analysis_booleans_by_category(
        human_data, "dataset_analysis"
    )
    dataset_by_category_ps = extract_analysis_booleans_by_category(
        ps_data, "dataset_analysis"
    )

    # Store results for weighted global computation
    category_results = {}

    # Compute Code metrics
    code_h_dict = extract_code_booleans(human_data)
    code_ps_dict = extract_code_booleans(ps_data)
    code_common_keys = set(code_h_dict.keys()).intersection(set(code_ps_dict.keys()))

    if code_common_keys:
        h_list = [code_h_dict[k] for k in code_common_keys]
        ps_list = [code_ps_dict[k] for k in code_common_keys]
        tp, tn, fp, fn = get_confusion_counts(h_list, ps_list)
        acc = accuracy_from_counts(tp, tn, fp, fn) * 100
        mcc = mcc_from_counts(tp, tn, fp, fn)
        category_results["Code"] = {"accuracy": acc, "mcc": mcc}

    # Compute Dataset metrics (weighted by category)
    dataset_results = {}

    for category in ["data_collection", "annotation", "ethics_availability"]:
        if category in dataset_by_category_human or category in dataset_by_category_ps:
            h_dict = dataset_by_category_human.get(category, {})
            ps_dict = dataset_by_category_ps.get(category, {})
            common_keys = set(h_dict.keys()).intersection(set(ps_dict.keys()))

            if common_keys:
                h_list = [h_dict[k] for k in common_keys]
                ps_list = [ps_dict[k] for k in common_keys]
                tp, tn, fp, fn = get_confusion_counts(h_list, ps_list)
                acc = accuracy_from_counts(tp, tn, fp, fn) * 100
                mcc = mcc_from_counts(tp, tn, fp, fn)
                dataset_results[category] = {"accuracy": acc, "mcc": mcc}

    # Calculate weighted Dataset metrics
    if dataset_results:
        weighted_acc = 0.0
        weighted_mcc_numerator = 0.0
        total_weight = 0.0

        for category in ["data_collection", "annotation", "ethics_availability"]:
            if category in dataset_results:
                weight = DATASET_WEIGHTS[category]
                total_weight += weight
                res = dataset_results[category]
                weighted_acc += res["accuracy"] * weight

                if not math.isnan(res["mcc"]):
                    weighted_mcc_numerator += res["mcc"] * weight

        if total_weight > 0:
            weighted_acc /= total_weight
            weighted_mcc = (
                weighted_mcc_numerator / total_weight
                if total_weight > 0
                else float("nan")
            )
            category_results["Dataset"] = {
                "accuracy": weighted_acc,
                "mcc": weighted_mcc,
            }

    # Compute Paper metrics (weighted by category)
    paper_results = {}

    for category in ["models", "datasets", "experiments"]:
        if category in paper_by_category_human or category in paper_by_category_ps:
            h_dict = paper_by_category_human.get(category, {})
            ps_dict = paper_by_category_ps.get(category, {})
            common_keys = set(h_dict.keys()).intersection(set(ps_dict.keys()))

            if common_keys:
                h_list = [h_dict[k] for k in common_keys]
                ps_list = [ps_dict[k] for k in common_keys]
                tp, tn, fp, fn = get_confusion_counts(h_list, ps_list)
                acc = accuracy_from_counts(tp, tn, fp, fn) * 100
                mcc = mcc_from_counts(tp, tn, fp, fn)
                paper_results[category] = {"accuracy": acc, "mcc": mcc}

    # Calculate weighted Paper metrics
    if paper_results:
        weighted_acc = 0.0
        weighted_mcc_numerator = 0.0
        total_weight = 0.0

        for category in ["models", "datasets", "experiments"]:
            if category in paper_results:
                weight = WEIGHTS[category]
                total_weight += weight
                res = paper_results[category]
                weighted_acc += res["accuracy"] * weight

                if not math.isnan(res["mcc"]):
                    weighted_mcc_numerator += res["mcc"] * weight

        if total_weight > 0:
            weighted_acc /= total_weight
            weighted_mcc = (
                weighted_mcc_numerator / total_weight
                if total_weight > 0
                else float("nan")
            )
            category_results["Paper"] = {"accuracy": weighted_acc, "mcc": weighted_mcc}

    # Calculate Global metrics with adaptive weights
    has_code = "Code" in category_results
    has_dataset = "Dataset" in category_results
    has_paper = "Paper" in category_results

    # Determine weights based on availability
    if has_code and has_dataset and has_paper:
        global_weights = {"Paper": 0.5, "Code": 0.2, "Dataset": 0.3}
    elif has_paper and has_code:
        global_weights = {"Paper": 0.6, "Code": 0.4}
    elif has_paper and has_dataset:
        global_weights = {"Paper": 0.6, "Dataset": 0.4}
    elif has_paper:
        global_weights = {"Paper": 1.0}
    else:
        available = [k for k in category_results.keys()]
        global_weights = {k: 1.0 / len(available) for k in available}

    # Compute weighted global accuracy and MCC
    global_weighted_acc = 0.0
    global_weighted_mcc = 0.0
    total_weight = 0.0
    valid_mcc_weight = 0.0

    for category, weight in global_weights.items():
        if category in category_results:
            res = category_results[category]
            total_weight += weight
            global_weighted_acc += res["accuracy"] * weight

            if not math.isnan(res["mcc"]):
                global_weighted_mcc += res["mcc"] * weight
                valid_mcc_weight += weight

    if total_weight > 0:
        global_weighted_acc /= total_weight
    if valid_mcc_weight > 0:
        global_weighted_mcc /= valid_mcc_weight
    else:
        global_weighted_mcc = float("nan")

    category_results["Global"] = {
        "accuracy": global_weighted_acc,
        "mcc": global_weighted_mcc,
    }

    return category_results


def compute_mean_excluding_nan(values: List[float]) -> float:
    """Compute mean excluding NaN values."""
    valid_values = [v for v in values if not math.isnan(v)]
    if not valid_values:
        return float("nan")
    return statistics.mean(valid_values)


def main():
    """Main aggregation function."""

    print("=" * 60)
    print("AGGREGATED METRICS ACROSS ALL PAPERS")
    print("=" * 60)

    # Store all paper results
    all_results = {}

    # Process each paper
    for paper_id, paper_type in sorted(PAPER_TYPES.items()):
        print(f"\nProcessing paper {paper_id} ({paper_type})...")
        results = compute_paper_metrics(paper_id, paper_type)
        if results:
            all_results[paper_id] = results
            print(f"  ✓ Successfully processed")
        else:
            print(f"  ✗ Failed to process")

    if not all_results:
        print("\nError: No papers were successfully processed")
        sys.exit(1)

    print(f"\n{len(all_results)} papers processed successfully\n")

    # Aggregate results by category
    categories = ["Code", "Paper", "Dataset", "Global"]
    aggregated = {}

    for category in categories:
        accuracies = []
        mccs = []

        for paper_id, results in all_results.items():
            if category in results:
                accuracies.append(results[category]["accuracy"])
                mccs.append(results[category]["mcc"])

        if accuracies:
            mean_acc = compute_mean_excluding_nan(accuracies)
            mean_mcc = compute_mean_excluding_nan(mccs)
            count = len(accuracies)

            aggregated[category] = {
                "mean_accuracy": mean_acc,
                "mean_mcc": mean_mcc,
                "count": count,
            }

    # Display results
    print("=" * 60)
    print("MEAN METRICS ACROSS ALL PAPERS")
    print("=" * 60)
    print(f"\n{'Analysis Category':<20} | {'% Agr.':<8} | {'MCC':<8} | {'N':<4}")
    print("-" * 60)

    for category in categories:
        if category in aggregated:
            res = aggregated[category]
            acc = res["mean_accuracy"]
            mcc = res["mean_mcc"]
            count = res["count"]

            acc_str = f"{acc:.2f}%" if not math.isnan(acc) else "NaN"
            mcc_str = f"{mcc:.4f}" if not math.isnan(mcc) else "NaN"

            print(f"{category:<20} | {acc_str:>8} | {mcc_str:>8} | {count:<4}")

    print("=" * 60)
    print(f"Total papers: {len(all_results)}")
    print("=" * 60)

    # Save detailed results to JSON
    output_file = "aggregated_metrics_results_ps5.json"
    output_data = {
        "summary": {
            category: {
                "mean_accuracy": res["mean_accuracy"],
                "mean_mcc": res["mean_mcc"],
                "count": res["count"],
            }
            for category, res in aggregated.items()
        },
        "per_paper": {
            paper_id: {
                category: {
                    "accuracy": results[category]["accuracy"],
                    "mcc": results[category]["mcc"],
                }
                for category in results.keys()
            }
            for paper_id, results in all_results.items()
        },
        "paper_types": PAPER_TYPES,
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
