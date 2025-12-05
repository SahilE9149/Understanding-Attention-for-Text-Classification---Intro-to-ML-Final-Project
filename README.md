# 🚀 UAFTC: Understanding Attention for Text Classification

**73% Faster • Same Accuracy • Same Interpretability**

Reproduction + improvement of [Sun & Lu ACL 2020](https://aclanthology.org/2020.acl-main.312/)

## 🎯 Results Summary

| Dataset    | LSTM     | Affine (Ours) | Speedup |
|------------|----------|---------------|---------|
| **SST**    | 82.2%    | **83.4%**     | **3.75x** |
| **IMDB**   | 89.8%    | 89.2%         | 3.75x    |
| **20News** | 94.2%    | **94.9%**     | 3.75x    |
| **Average**| 88.1%    | **87.4%**     | **73%**  |

**Key Findings:**
- ✅ **73% time reduction** (54 → 14 GPU hours)
- ✅ **Correlation preserved** (r=0.714 vs 0.726) 
- ✅ **L2=0.001** boosts small datasets +1.5%

## 🚀 Quick Start (2 minutes)
git clone https://github.com/YOUR_USERNAME/UAFTC
cd UAFTC

Install
pip install -r requirements.txt

Test SST (2min)
python test_sst.py

Full experiments (1hr)
python run_experiments.py

# 📁 What's Inside

UAFTC/
├── data/ # Preprocessed SST - ( Links of IMDB and News 20 Dataset available below)
├── improved_attn_model.py # 🔥 LSTM + Affine encoders
├── run_experiments.py # 112 hyperparameter configs
├── notebooks/ # Jupyter analysis
│ ├── attention_analysis.ipynb
│ └── synthetic_analysis.ipynb
├── CSV tables + plots

## 🧠 Model Architecture
Your improved model (drop-in replacement)
model = ImprovedAttentionModel(
encoder_type='affine', # 3.75x faster than LSTM
scaling_factor=10.0, # Optimal d=√10
l2_lambda=0.001, # Small dataset boost
embed_dim=100
)

Attention: aⱼ = hⱼᵀV/√d
Polarity: sⱼ = hⱼᵀW

## 🔬 Reproduce Figures

Correlation plot (H3)
jupyter notebook notebooks/attention_analysis.ipynb

Synthetic patterns (VI.B)
jupyter notebook notebooks/synthetic_analysis.ipynb 

**Generated:**
✅ H3_correlation.png # r=0.71 scatter
✅ synthetic_patterns.png # Pos>Neutral>Neg
✅ results_table.csv # Full 112 configs 

## 📊 Full Hyperparameter Sweep
Tested **112 configurations:**
7 scales × 2 encoders × 4 datasets
[0.001, 1, 10, 20, 50, 100, 10000]

## 🛠️ Setup
Requirements
pip install torch pandas numpy matplotlib jupyter

Preprocess data (run once)
python data_processor.py # Creates data/*.pkl

## 📈 Expected Outputs

results/
├── lstm_results.csv # Baseline reproduction
├── affine_results.csv # Your improvements
├── timing_comparison.png # 73% speedup plot
└── correlation_scatter.png # r=0.71 figure


Authors: Sahil Ekhande, Yashraj Mohite
