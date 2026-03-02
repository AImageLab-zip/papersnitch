import json
import math
import sys


# ==========================================
# 1. METRICS FUNCTIONS
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


# ==========================================
# 2. DATA EXTRACTION FUNCTIONS
# ==========================================
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

    Returns a dict of criterion_id -> satisfied (bool)
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
        # True or None counts as satisfied
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

    # 3. Artifacts - use compute_score logic
    artifacts = code_sec.get("artifacts", {})
    if "has_checkpoints" in artifacts:
        bools["artifacts.has_checkpoints"] = artifacts["has_checkpoints"]
    if "has_dataset_links" in artifacts:
        bools["artifacts.has_dataset_links"] = artifacts["has_dataset_links"]
    if "dataset_coverage" in artifacts and requires_datasets:
        # Full coverage = satisfied, partial/none = not satisfied
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


# ==========================================
# 3. LOAD & PROCESS DATA
# ==========================================
if len(sys.argv) < 2:
    print("Usage: python metrics_cont.py <paper_id> [paper_type]")
    print("Example: python metrics_cont.py 0421 method")
    print("         python metrics_cont.py 2124 both")
    print("paper_type: 'method' or 'both' (default: 'method')")
    sys.exit(1)

paper_id = sys.argv[1]
paper_type = sys.argv[2] if len(sys.argv) > 2 else "method"

if paper_type not in ["method", "both"]:
    print(f"Error: paper_type must be 'method' or 'both', got '{paper_type}'")
    sys.exit(1)

# Define weights based on paper type
if paper_type == "both":
    # Both papers: equal weights across all categories
    WEIGHTS = {"models": 0.35, "datasets": 0.35, "experiments": 0.35}
else:  # method
    # Method papers: emphasis on models
    WEIGHTS = {"models": 0.45, "datasets": 0.20, "experiments": 0.35}

with open(f"{paper_id}_human.json", "r") as f:
    human_data = json.load(f)

with open(f"{paper_id}_ps.json", "r") as f:
    ps_data = json.load(f)

# Map the data to separate Paper analysis by category for internal weighted computation
paper_by_category_human = extract_analysis_booleans_by_category(
    human_data, "paper_analysis"
)
paper_by_category_ps = extract_analysis_booleans_by_category(ps_data, "paper_analysis")

# Map Dataset analysis by category for weighted computation
dataset_by_category_human = extract_analysis_booleans_by_category(
    human_data, "dataset_analysis"
)
dataset_by_category_ps = extract_analysis_booleans_by_category(
    ps_data, "dataset_analysis"
)

# Define dataset weights
DATASET_WEIGHTS = {
    "data_collection": 0.35,
    "annotation": 0.40,
    "ethics_availability": 0.25,
}

# ==========================================
# 4. COMPUTE METRICS & PRINT TABLE
# ==========================================
print(f"\n{'Analysis Category':<20} | {'% Agr.':<8} | {'MCC':<8}")
print("-" * 42)

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
    mcc_str = f"{mcc:.4f}" if not math.isnan(mcc) else "NaN"

    category_results["Code"] = {
        "accuracy": acc,
        "mcc": mcc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    print(
        f"{'Code':<20} | {acc:>5.2f}%   | {mcc_str:>6}  [TP={tp}, TN={tn}, FP={fp}, FN={fn}]"
    )

# Compute Dataset metrics (weighted by category)
dataset_results = {}
dataset_total_tp, dataset_total_tn, dataset_total_fp, dataset_total_fn = 0, 0, 0, 0

for category in ["data_collection", "annotation", "ethics_availability"]:
    if category in dataset_by_category_human or category in dataset_by_category_ps:
        h_dict = dataset_by_category_human.get(category, {})
        ps_dict = dataset_by_category_ps.get(category, {})
        common_keys = set(h_dict.keys()).intersection(set(ps_dict.keys()))

        if common_keys:
            h_list = [h_dict[k] for k in common_keys]
            ps_list = [ps_dict[k] for k in common_keys]
            tp, tn, fp, fn = get_confusion_counts(h_list, ps_list)

            dataset_total_tp += tp
            dataset_total_tn += tn
            dataset_total_fp += fp
            dataset_total_fn += fn

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
            weighted_mcc_numerator / total_weight if total_weight > 0 else float("nan")
        )
        weighted_mcc_str = (
            f"{weighted_mcc:.4f}" if not math.isnan(weighted_mcc) else "NaN"
        )

        category_results["Dataset"] = {
            "accuracy": weighted_acc,
            "mcc": weighted_mcc,
            "tp": dataset_total_tp,
            "tn": dataset_total_tn,
            "fp": dataset_total_fp,
            "fn": dataset_total_fn,
        }
        print(
            f"{'Dataset':<20} | {weighted_acc:>5.2f}%   | {weighted_mcc_str:>6}  [TP={dataset_total_tp}, TN={dataset_total_tn}, FP={dataset_total_fp}, FN={dataset_total_fn}]"
        )

# Compute Paper metrics (weighted by category)
paper_results = {}
paper_total_tp, paper_total_tn, paper_total_fp, paper_total_fn = 0, 0, 0, 0

for category in ["models", "datasets", "experiments"]:
    if category in paper_by_category_human or category in paper_by_category_ps:
        h_dict = paper_by_category_human.get(category, {})
        ps_dict = paper_by_category_ps.get(category, {})

        # Find common keys to compare
        common_keys = set(h_dict.keys()).intersection(set(ps_dict.keys()))

        if common_keys:
            # Build parallel lists of boolean values
            h_list = [h_dict[k] for k in common_keys]
            ps_list = [ps_dict[k] for k in common_keys]

            # Calculate confusion matrix
            tp, tn, fp, fn = get_confusion_counts(h_list, ps_list)

            # Accumulate for paper total
            paper_total_tp += tp
            paper_total_tn += tn
            paper_total_fp += fp
            paper_total_fn += fn

            # Calculate metrics for weighted average
            acc = accuracy_from_counts(tp, tn, fp, fn) * 100
            mcc = mcc_from_counts(tp, tn, fp, fn)

            paper_results[category] = {
                "accuracy": acc,
                "mcc": mcc,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
            }

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

            # Only include in MCC average if not NaN
            if not math.isnan(res["mcc"]):
                weighted_mcc_numerator += res["mcc"] * weight

    if total_weight > 0:
        weighted_acc /= total_weight
        weighted_mcc = (
            weighted_mcc_numerator / total_weight if total_weight > 0 else float("nan")
        )
        weighted_mcc_str = (
            f"{weighted_mcc:.4f}" if not math.isnan(weighted_mcc) else "NaN"
        )

        category_results["Paper"] = {
            "accuracy": weighted_acc,
            "mcc": weighted_mcc,
            "tp": paper_total_tp,
            "tn": paper_total_tn,
            "fp": paper_total_fp,
            "fn": paper_total_fn,
        }
        print(
            f"{'Paper':<20} | {weighted_acc:>5.2f}%   | {weighted_mcc_str:>6}  [TP={paper_total_tp}, TN={paper_total_tn}, FP={paper_total_fp}, FN={paper_total_fn}]"
        )

# Calculate Global metrics with adaptive weights
print("-" * 42)

has_code = "Code" in category_results
has_dataset = "Dataset" in category_results
has_paper = "Paper" in category_results

# Determine weights based on availability
if has_code and has_dataset and has_paper:
    # All three: 50% paper, 20% code, 30% dataset
    global_weights = {"Paper": 0.5, "Code": 0.2, "Dataset": 0.3}
elif has_paper and has_code:
    # Paper + code: 60% paper, 40% code
    global_weights = {"Paper": 0.6, "Code": 0.4}
elif has_paper and has_dataset:
    # Paper + dataset: 60% paper, 40% dataset
    global_weights = {"Paper": 0.6, "Dataset": 0.4}
elif has_paper:
    # Only paper: 100%
    global_weights = {"Paper": 1.0}
else:
    # Fallback: equal weights for whatever is available
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

        # Only include in MCC average if not NaN
        if not math.isnan(res["mcc"]):
            global_weighted_mcc += res["mcc"] * weight
            valid_mcc_weight += weight

# Normalize
if total_weight > 0:
    global_weighted_acc /= total_weight
if valid_mcc_weight > 0:
    global_weighted_mcc /= valid_mcc_weight
else:
    global_weighted_mcc = float("nan")

global_mcc_str = (
    f"{global_weighted_mcc:.4f}" if not math.isnan(global_weighted_mcc) else "NaN"
)

print(f"{'Global':<20} | {global_weighted_acc:>5.2f}%   | {global_mcc_str:>6}")

# Show weight breakdown
weight_info = ", ".join([f"{k}: {v:.0%}" for k, v in global_weights.items()])
print(
    f"\nPaper type: {paper_type} | Paper weights - models: {WEIGHTS['models']}, datasets: {WEIGHTS['datasets']}, experiments: {WEIGHTS['experiments']}"
)
print(
    f"Dataset weights - data_collection: {DATASET_WEIGHTS['data_collection']}, annotation: {DATASET_WEIGHTS['annotation']}, ethics_availability: {DATASET_WEIGHTS['ethics_availability']}"
)
print(f"Global weights: {weight_info}")
