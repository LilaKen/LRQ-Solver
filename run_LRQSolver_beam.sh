# Train
python main_beam.py \
    --config-name lrqsolver_beam.yaml \
    mode="train" \
    lr=0.002 \
    val_freq=1 \
    model.in_dim=24  \
    model.hidden_channel=20 \
    num_epochs=5000 \
    data_module.scaler_dir=./data/beam/scalers \
    data_module.input_param_path=./data/beam/input_params.npy \
    data_module.output_npz_path=./data/beam/processed_data/Outputs_rpt3_N5000.npz

# Test
python main_beam.py \
    --config-name lrqsolver_beam.yaml \
    mode="test" \
    test_batch_size=1 \
    model.in_dim=24  \
    model.hidden_channel=20 \
    data_module.scaler_dir=./data/beam/scalers \
    data_module.input_param_path=./data/beam/input_params.npy \
    data_module.output_npz_path=./data/beam/processed_data/Outputs_rpt3_N60000.npz \
    checkpoint=/ssd2/wangguan12/repo/output/lrqsolver_beam/train/2025-09-16-13-06-17/best_model.pdparams


# python visual_beam.py \
#     --config-name lrqsolver_beam.yaml \
#     batch_size=16 \
#     model.in_dim=24  \
#     model.hidden_channel=20
