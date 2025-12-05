Files required for Extension 1: 

New files to add:
1. Encoders in models/: lstm_encoder.py, affine_encoder.py, gru_encoder.py, cnn_encoder.py, __init__.py
2. Main model: improved_attn_model_extended.py
3. Runner: extension_grid_sweep.py
4. Analysis: generate_tables.py, extension_analysis.ipynb
5. Documentation: EXTENSION_1_README.md​

Existing files to keep:
1. Data under data/ for SST and IMDB
2. run_experiments.py, helper.py, data_processor.py, and improved_attn_model.py as references.​

Recommended implementation order
1. Create models/ and add all encoder files plus __init__.py.
2. Copy improved_attn_model_extended.py, extension_grid_sweep.py, generate_tables.py, extension_analysis.ipynb, and EXTENSION_1_README.md into the project root.
3. Verify the structure using ls and test that importing ExtendedAttentionModel works.
4. Run a small sanity‑check forward pass with one encoder.
5. Run the full grid with extension_grid_sweep.py to produce extension_results.csv.
6. Generate tables and plots using generate_tables.py and the notebook.​

Common problems and fixes
1. Module import errors: ensure models/__init__.py exists and encoders are in models/.
2. Data not found: check that SST and IMDB sequences are under data/sst/ and data/imdb/ and that paths in extension_grid_sweep.py match.
3. Out‑of‑memory or very slow runs: reduce batch size or test fewer scaling values or datasets.​

What you should have at the end
1. After everything runs successfully, you should see:
2. extension_results.csv with one row per configuration
3. Three PNG plots for accuracy, correlation, and training time
4. extension_results.txt if you redirected generate_tables.py output
5. Enough numbers and plots to write a new “Extension 1: Modern Encoder Architectures” section in your report.