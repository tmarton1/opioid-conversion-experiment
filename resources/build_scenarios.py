"""
Validate and augment reworded_vignettes.json.

1. Validates that all vignettes within a scenario share identical
   conversion parameters (everything except vignette_id, text,
   and has_vignette_text). Raises an error with details if the
   assumption is violated.

2. Adds ``has_vignette_text: true`` to every existing vignette
   that lacks the field.

3. For each scenario that doesn't already have an empty-text
   vignette, adds one (with has_vignette_text=false and text="").

4. Writes the updated vignettes back to the JSON file.

5. Returns the list of empty-text vignette IDs.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

# Fields that are unique per vignette (not part of the scenario definition)
VIGNETTE_ONLY_FIELDS = {"vignette_id", "text", "has_vignette_text"}

VIGNETTES_PATH = Path(__file__).parent / "reworded_vignettes.json"


def build_scenarios(vignettes_path: str = None, verbose: bool = True):
    """Validate scenario consistency, add has_vignette_text field, and
    create empty-text vignettes for each scenario.

    Returns:
        list[str]: Vignette IDs of all empty-text vignettes.
    """
    if vignettes_path is None:
        vignettes_path = str(VIGNETTES_PATH)

    with open(vignettes_path, "r") as f:
        vignettes = json.load(f)

    # --- Step 1: Group by scenario_key and validate ---
    groups = defaultdict(list)
    for vignette_id, vignette in vignettes.items():
        scenario_key = vignette.get("scenario_key")
        if not scenario_key:
            raise ValueError(f"Vignette {vignette_id} has no scenario_key")
        groups[scenario_key].append(vignette)

    errors = []
    for scenario_key, vignette_list in sorted(groups.items()):
        # Use the first vignette with text as the reference
        text_vignettes = [v for v in vignette_list if v.get("text", "") != ""]
        if not text_vignettes:
            continue  # All empty — nothing to validate against
        reference = text_vignettes[0]
        scenario_fields = {
            k: v for k, v in reference.items() if k not in VIGNETTE_ONLY_FIELDS
        }

        for vignette in vignette_list:
            if vignette["vignette_id"] == reference["vignette_id"]:
                continue
            other_fields = {
                k: v for k, v in vignette.items() if k not in VIGNETTE_ONLY_FIELDS
            }
            if other_fields != scenario_fields:
                all_keys = set(scenario_fields.keys()) | set(other_fields.keys())
                mismatches = []
                for k in sorted(all_keys):
                    ref_val = scenario_fields.get(k)
                    other_val = other_fields.get(k)
                    if ref_val != other_val:
                        mismatches.append(
                            f"    {k}: {ref_val!r} != {other_val!r}"
                        )
                errors.append(
                    f"Scenario '{scenario_key}': vignette {vignette['vignette_id']} "
                    f"differs from {reference['vignette_id']}:\n"
                    + "\n".join(mismatches)
                )

    if errors:
        raise ValueError(
            f"Found {len(errors)} inconsistencies across scenarios:\n\n"
            + "\n\n".join(errors)
        )

    if verbose:
        print(f"✓ Validated {len(groups)} scenarios — all consistent")

    # --- Step 2: Add has_vignette_text to existing vignettes ---
    added_field_count = 0
    for vignette in vignettes.values():
        if "has_vignette_text" not in vignette:
            has_text = vignette.get("text", "") != ""
            vignette["has_vignette_text"] = has_text
            added_field_count += 1

    if verbose and added_field_count > 0:
        print(f"✓ Added has_vignette_text field to {added_field_count} vignettes")

    # --- Step 3: Add empty-text vignettes for scenarios that need one ---
    # Find which scenarios already have an empty-text vignette
    scenarios_with_empty = set()
    existing_empty_ids = []
    for vignette in vignettes.values():
        if not vignette.get("has_vignette_text", True):
            scenarios_with_empty.add(vignette["scenario_key"])
            existing_empty_ids.append(vignette["vignette_id"])

    # Determine next vignette ID
    max_id = max(int(vid[1:]) for vid in vignettes.keys())
    next_id = max_id + 1

    new_empty_ids = []
    for scenario_key in sorted(groups.keys()):
        if scenario_key in scenarios_with_empty:
            continue

        # Use any text-vignette as template for scenario-level fields
        template = groups[scenario_key][0]
        new_vid = f"v{next_id}"
        next_id += 1

        new_vignette = {}
        for k, v in template.items():
            if k in VIGNETTE_ONLY_FIELDS:
                continue
            new_vignette[k] = v
        new_vignette["vignette_id"] = new_vid
        new_vignette["text"] = ""
        new_vignette["has_vignette_text"] = False

        vignettes[new_vid] = new_vignette
        new_empty_ids.append(new_vid)

    if verbose and new_empty_ids:
        print(f"✓ Added {len(new_empty_ids)} empty-text vignettes: {', '.join(new_empty_ids)}")

    all_empty_ids = sorted(existing_empty_ids + new_empty_ids, key=lambda x: int(x[1:]))

    # --- Step 4: Write back ---
    sorted_vignettes = dict(
        sorted(vignettes.items(), key=lambda x: int(x[0][1:]))
    )
    with open(vignettes_path, "w") as f:
        json.dump(sorted_vignettes, f, indent=2)

    if verbose:
        total = len(sorted_vignettes)
        text_count = sum(1 for v in sorted_vignettes.values() if v.get("has_vignette_text", True))
        print(f"✓ Wrote {total} vignettes ({text_count} with text, {total - text_count} without) to {vignettes_path}")
        print(f"\nEmpty-text vignette IDs: {all_empty_ids}")

    return all_empty_ids


if __name__ == "__main__":
    vignettes_path = sys.argv[1] if len(sys.argv) > 1 else str(VIGNETTES_PATH)
    build_scenarios(vignettes_path)
