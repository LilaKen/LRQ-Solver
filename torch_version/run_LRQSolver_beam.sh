#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Train
# uv run python "${SCRIPT_DIR}/main_beam.py" \
#     --config-name lrqsolver_beam.yaml \
#     mode="train" \
#     lr=0.002 \
#     val_freq=1 \
#     model.in_dim=24  \
#     model.hidden_channel=20 \
#     num_epochs=5000 \
#     data_module.scaler_dir=./data/beam/scalers \
#     data_module.input_param_path=./data/beam/input_params.npy \
#     data_module.output_npz_path=./data/beam/processed_data/Outputs_rpt3_N5000.npz

# Test
uv run python "${SCRIPT_DIR}/main_beam.py" \
    --config-name lrqsolver_beam.yaml \
    mode="test" \
    test_batch_size=1 \
    model.in_dim=24  \
    model.hidden_channel=20 \
    data_module.scaler_dir=./data/beam/scalers \
    data_module.input_param_path=./data/beam/input_params.npy \
    data_module.output_npz_path=./data/beam/processed_data/Outputs_rpt3_N60000.npz \
    checkpoint=./output/lrqsolver_beam/best_model.pt


# uv run python "${SCRIPT_DIR}/visual_beam.py" \
#     --config-name lrqsolver_beam.yaml \
#     batch_size=16 \
#     model.in_dim=24  \
#     model.hidden_channel=20
