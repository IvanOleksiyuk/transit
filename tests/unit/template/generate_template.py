import sys
from pathlib import Path

current_dir = Path.cwd()
src_path = current_dir / "src"
sys.path.append(str(src_path))

from transit.src.data.generate_dummy import make_gaussian

output_data = Path(snakemake.output[0])  # noqa: F821
output_template = Path(snakemake.output[1])  # noqa: F821

# Create the directory if it doesn't exist
output_dir = output_data.parent
output_dir.mkdir(parents=True, exist_ok=True)

make_gaussian(int(1e5), "feature_0,feature_1", output_data, 1)
make_gaussian(int(1e5), "feature_0,feature_1", output_template, -1)
