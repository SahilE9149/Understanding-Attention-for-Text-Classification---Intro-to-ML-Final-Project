Overview: Extension 1 adds about ten new files and a few plots. Together they implement and evaluate LSTM, Affine, GRU, and CNN encoders under a single attention model.​

New encoder files (in models/)
1. lstm_encoder.py – LSTM baseline encoder
2. affine_encoder.py – affine baseline encoder
3. gru_encoder.py – GRU encoder (fewer parameters than LSTM, faster)
4. cnn_encoder.py – 1D CNN encoder (parallel, fastest)
 5. __init__.py – exposes all four encoders as a package​

Main model and runner (root)
1. improved_attn_model_extended.py – defines ExtendedAttentionModel with an encoder_type argument ("lstm", "affine", "gru", "cnn").
2. extension_grid_sweep.py – runs a grid of experiments over encoders, datasets, and scaling factors, saving results to extension_results.csv.​

Analysis tools
1. generate_tables.py – reads extension_results.csv and prints tables: accuracy, correlation, training time, speedup, and best configurations.
2. extension_analysis.ipynb – notebook that loads the CSV and produces three plots: accuracy vs scaling, correlation vs scaling, and training time per encoder. It also saves PNG files such as extension_accuracy_vs_tau.png and extension_training_time.png.​

Documentation   
1. EXTENSION_1_README.md – long guide describing setup, experiments, and how to write the new section in the report.

2. OPTION_2_CHECKLIST.md – implementation checklist and common issues.​

Minimal quick start
1. Ensure models/ exists and copy all encoder files plus __init__.py.
2. Place the main scripts and notebook in the project root.
3. Test that ExtendedAttentionModel imports correctly.
4. Run extension_grid_sweep.py to generate results.
5. Run generate_tables.py or the notebook to get tables and plots for the report.​