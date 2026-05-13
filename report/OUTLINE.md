# Report Outline — Game Playing with Machine Learning
**Course**: Introduction to Artificial Intelligence  
**Group 3** | CC01 | Advisor: Dr. Trần Hồng Tài

---

## 1. Introduction
- 1.1 Game Playing in Artificial Intelligence
- 1.2 The Othello Game

## 2. Background
- 2.1 Minimax Algorithm
- 2.2 Alpha-Beta Pruning
- 2.3 Negamax Formulation
- 2.4 Hand-Crafted Heuristic Agent (HC baseline)
- 2.5 Machine Learning for Game Playing

## 3. Proposed Approach
- 3.1 Motivation — Why combine Dataset + ML?
- 3.2 Overall Pipeline Overview

## 4. Dataset Generation & Model Architecture
- 4.1 Expert Data Generation (Minimax teacher L6–L10)
  - 4.1.1 Dataset Statistics
- 4.2 4-Channel Board Encoding
- 4.3 Model Architecture (OthelloMLP)
  - 4.3.1 Backbone (6 FC layers)
  - 4.3.2 Dual Output Heads (Policy + Value)
  - 4.3.3 Parameter Count (~1.3M, ~5MB)
- 4.4 Training Pipeline
  - 4.4.1 Loss Function (policy CE + value MSE)
  - 4.4.2 Hyperparameters
- 4.5 Negamax-Augmented Inference
  - 4.5.1 Learned Leaf Evaluation
  - 4.5.2 Policy-Guided Move Ordering
  - 4.5.3 Iterative Deepening (odd depths)
  - 4.5.4 Algorithm Pseudocode

## 5. Benchmark & Result Analysis
- 5.1 Evaluation Setup
- 5.2 Results (Win/Draw/Loss vs Minimax L1–L9)
- 5.3 Discussion
  - 5.3.1 Strengths
  - 5.3.2 Limitations (MLP vs CNN, imitation ceiling)

## 6. Conclusion

## References

---

## File map

| Section | File |
|---|---|
| 1. Introduction | `report/Introduction.tex` |
| 2. Background | `report/Background.tex` |
| 3. Proposed Approach | `report/section3.tex` |
| 4. Dataset + Model | `report/ModelArchitecture.tex` |
| 5. Benchmark | `report/Result_Analysis.tex` (đầu) |
| 6. Conclusion + Refs | `report/Result_Analysis.tex` (cuối) |
