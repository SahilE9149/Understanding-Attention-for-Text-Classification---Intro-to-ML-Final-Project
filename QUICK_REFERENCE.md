Goal: Implement “Extension 1: Modern Encoder Comparison” for attention‑based text classification.

You will compare four encoders: LSTM, Affine, GRU, and CNN, and show how accuracy, interpretability (correlation r), and training time change across them.​

Files to have in the project
Create models/:
models/lstm_encoder.py – baseline LSTM encoder
models/affine_encoder.py – affine (linear) encoder
models/gru_encoder.py – GRU encoder (new)
models/cnn_encoder.py – CNN encoder (new)
models/__init__.py – imports all encoder classes

Project root:
improved_attn_model_extended.py – attention model that can use any encoder
extension_grid_sweep.py – runs all experiments
generate_tables.py – converts CSV results into tables
extension_analysis.ipynb – notebook for plots
EXTENSION_1_README.md – full guide
OPTION_2_CHECKLIST.md – checklist and troubleshooting​

How to run
1. Copy the files above into your repo and ensure models/ exists.
2. Run all experiments: python extension_grid_sweep.py (This creates extension_results.csv with about 24 rows (encoder × dataset × scaling))
3. Analyze results: python generate_tables.py && jupyter notebook extension_analysis.ipynb (You will get five tables and three plots (accuracy vs scaling, correlation vs scaling, training time))

What to report
a. Accuracy for each encoder and scaling value
b. Correlation r for each encoder
c. Training time and speedup relative to LSTM
d. A short discussion explaining which encoder is fastest, which gives the best accuracy, and whether interpretability is preserved.

​


