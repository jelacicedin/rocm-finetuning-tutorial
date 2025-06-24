import os
import random
import json
from datetime import datetime
from generators.scenario_templates import (
    generate_straight_road,
    generate_clothoid_scenario,
    generate_line_spiral_combo,
    generate_adjustable_planview_junction,
    generate_t_junction,
    generate_arc_road,
    generate_composite_road,
    generate_cross_junction,
)
import subprocess

# Set up output folder with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"xodr_generated_scenarios_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# List of all scenario generators
generators = [
    lambda: generate_straight_road(
        length=random.randint(50, 300),
        road_id=random.randint(1, 100),
        lanes_left=random.choice([1, 2, 3]),
        lanes_right=random.choice([1, 2, 3]),
    ),
    lambda: generate_clothoid_scenario(
        start_x=0,
        start_y=0,
        end_x=random.randint(50, 300),
        end_y=random.randint(-100, 100),
        lanes_left=random.choice([1, 2, 3]),
        lanes_right=random.choice([1, 2, 3]),
    ),
    generate_line_spiral_combo,
    generate_adjustable_planview_junction,
    generate_t_junction,
    lambda: generate_arc_road(
        length=random.randint(50, 150),
        curvature=random.uniform(0.005, 0.03),
        road_id=random.randint(101, 200),
    ),
    generate_composite_road,
    generate_cross_junction,
]

# Generate N examples
num_examples = 20
metadata = []

for i in range(num_examples):
    gen_fn = random.choice(generators)

    try:
        example = gen_fn()
        scenario_name = f"scenario_{i:03d}"
        script_filename = f"{scenario_name}.py"
        script_path = os.path.join(output_dir, script_filename)

        # Write scenario Python script
        with open(script_path, "w") as f:
            f.write(example["response"])

        # Save metadata
        metadata.append(
            {
                "name": scenario_name,
                "prompt": example["prompt"],
                "script_path": script_path,
            }
        )

    except Exception as e:
        print(f"[!] Error in generator {gen_fn.__name__}: {e}")

# Write all metadata to JSONL file
metadata_path = os.path.join(output_dir, "scenario_metadata.jsonl")
with open(metadata_path, "w") as f:
    for item in metadata:
        f.write(json.dumps(item) + "\n")

print(f"✅ Generated {len(metadata)} scenarios in '{output_dir}'")
print(f"🗂️  Metadata saved to '{metadata_path}'")

# Load metadata
with open(metadata_path, "r") as f:
    metadata = [json.loads(line.strip()) for line in f.readlines()]

successes = []
failures = []

# Run each scenario script
for item in metadata:
    name = item["name"]
    script_path = item["script_path"]
    print(f"▶ Running {name}...")

    try:
        result = subprocess.run(
            ["python", script_path], capture_output=True, text=True, check=True
        )
        print(f"✅ Success: {name}")

        # Move any generated .xodr or .xosc files to output_dir
        for file in os.listdir("."):
            if file.endswith(".xodr") or file.endswith(".xosc"):
                new_name = f"{name}_{file}"
                os.rename(file, os.path.join(output_dir, new_name))

        successes.append(name)

    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {name}\n{e.stderr}")
        failures.append({"name": name, "error": e.stderr})

# Summary
print("\n--- Execution Summary ---")
print(f"✅ Successful: {len(successes)}")
print(f"❌ Failed: {len(failures)}")
if failures:
    for fail in failures:
        print(f"- {fail['name']}: {fail['error'][:100]}...")
