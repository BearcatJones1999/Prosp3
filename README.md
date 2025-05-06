# 🏝️ Prosperity 3 Algorithmic Trading Competition – Strategy Archive

This repository contains my complete set of trading algorithms submitted for each round of the Prosperity 3 algorithmic trading competition. Over five rounds, I developed and refined a suite of quantitative strategies including machine learning-based classifiers, momentum filters, z-score arbitrage, and basket pricing logic.

---

## 🔁 Strategy Summary by Round

### 🥇 Round 1: Momentum and Imbalance-Based Stat Arb
- Focused on **KELP**, **RAINFOREST_RESIN**, and **SQUID_INK**
- Employed statistical arbitrage using:
  - Fair value estimation based on moving averages
  - Trade momentum signals
  - Volume imbalance and recent fills
- Dynamic skewing based on position to control exposure
- Handled each product individually with tuned parameters

### 🥈 Round 2: ML Hybrid Introduction
- Introduced a lightweight **online logistic regression classifier** for **JAMS** and **DJEMBES**
- Combined ML signals with stat-arb and z-score-based strategies:
  - ML: directional probability based on 4 features (price, spread, imbalance, momentum)
  - Z-score arbitrage for **CROISSANTS** and **SQUID_INK**
  - Synthetic pricing for **PICNIC_BASKET1/2**

### 🥉 Round 3: Multi-Strategy Scaling
- Expanded ML model to cover 9+ products (including **Vouchers** and **MACARONS**)
- Added product-specific confidence thresholds for trade signals
- Improved feature engineering and portfolio exposure control
- Introduced decay-based position scaling and z-score filters

### 🔬 Round 4: Model Tuning and Cross-Product Signal Refinement
- Further adjusted ML thresholds for better directional accuracy
- Enhanced z-score trading rules with tighter windows
- Implemented more robust basket pricing logic based on observed lagged mispricings
- Began adapting safe-limit scaling to time progression for risk management

### 🧪 Round 5: Final Integration of All Signal Types
- Unified ML, stat-arb, z-score, and basket pricing under one engine
- Tuned per-product signal confidence levels and thresholds
- Enforced dynamic position decay based on round progression
- Focused on capital allocation across ~15 products in parallel with consistent signal processing

---

## 🧠 Core Models and Techniques
- **Online Logistic Regression** (custom, per-product update)
- **Z-score Arbitrage** (short/long window MA and volatility)
- **Fair Value Estimation** with volume imbalance and momentum
- **Basket Pricing** using synthetic valuation of constituent products
- **Inventory Skew** and fill-aware position management

---

## 📁 Repository Structure
