# Analyses added during the peer review

Scripts written for the revision of the paper. They do not modify the main pipeline: they import its code (`config/`, `data/`, `models/`) and read the weights of the final model. They are published as they were run on the computing cluster, so they keep its absolute paths and comments in Spanish; to run them elsewhere, change the paths at the top of each script.

| Script | Result in the paper |
| --- | --- |
| `bloque_a_baselines.py`, `bloque_a2_coatcervix_cv.py`, `bloque_a3_efficientnet_cv.py` | Table 7: each fold of the five fold cross validation (training partition only) evaluated on the independent test set |
| `bloque_b2_significancia_folds.pl` | Welch's t test and exact Mann–Whitney U test with Holm correction between CoatCervix and each baseline (Table 7 and Results). `bloque_b_significancia.pl` is an earlier version of the same test |
| `bloque_c_latencia.py` | Table 8: latency (median and interquartile range of 200 forward passes) and peak GPU memory |
| `bloque_d_preparar_hmcformer.py`, `bloque_d2_hmcformer_split_fijo.py`, `HMCFormer_train_only.py` | Our replication of HMCFormer, with cross validation restricted to the training partition (Tables 6 and 7) |
| `xai_quant.py`, `patch_wilcoxon.py`, `merge_final.py` | Table 11: faithfulness of the explanations (NAOPC, MoRF and LeRF, beam search bounds, bootstrap confidence intervals, Wilcoxon tests with Holm correction) and the model parameter randomization test. Includes SHAP (expected gradients), Grad-CAM, LayerCAM, and integrated gradients |
| `xai_lime.py`, `xai_lime_sanity.py` | LIME in Table 11 and its randomization test |
| `bloque_e_figuras_xai.py`, `bloque_e2_redibujar_fig8.py`, `bloque_e3_figuras_mapas_evaluados.py` | Figures 7 (Grad-CAM at the output of the third stage) and 8 (LIME and SHAP); the final figures are drawn by `bloque_e3_figuras_mapas_evaluados.py` from the maps evaluated in Table 11 |
| `bloque_f_figuras_eficiencia.py` | Figures 2 and 3 (operations and parameters of each model) |
| `bloque_g_figuras_roc_matriz.py` | Figures 4 and 5 (ROC curves and confusion matrix of the best run) |
| `modelo_final_seed42.py` | Retraining of the final model with seed 42 |

The scripts that draw figures from the final model first check that it reproduces the test results of the paper (accuracy 0.8133 and the confusion matrix of Figure 5) and stop otherwise.
