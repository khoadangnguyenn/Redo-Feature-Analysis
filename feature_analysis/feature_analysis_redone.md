# Redone Feature Analysis

## Setup

I reran the feature analysis on the provided `libero10_features` cache with 138,090 consecutive LIBERO-10 frame pairs. The split is video-disjoint using `train_test_split(test_size=0.2, random_state=42)`: 111,284 train pairs and 26,806 held-out pairs from 400 train videos and 100 test videos.

Features are z-scored using train statistics only. The main probe is Ridge over `alpha=1000` with the LSQR solver, evaluated by held-out R2 and Spearman rho against the xyz translation magnitude.

## Probe Results

| Feature | Dim | Alpha | Ridge R2 | Spearman rho |
|---|---:|---:|---:|---:|
| RGB | 4096 | 1e+03 | 0.4539 | 0.6826 |
| GT depth | 1024 | 1e+03 | 0.1254 | 0.3566 |
| RGB + GT depth | 5120 | 1e+03 | 0.4781 | 0.7005 |
| RGB + Model 1 | 5120 | 1e+03 | 0.4541 | 0.6835 |
| RGB + Model 2 | 5120 | 1e+03 | 0.5195 | 0.7264 |
| RGB + Model 3 | 5120 | 1e+03 | 0.4572 | 0.6844 |
| RGB + Model 4 | 5120 | 1e+03 | 0.4811 | 0.7002 |
| RGB + Model 5 | 5120 | 1e+03 | 0.4545 | 0.6832 |
| RGB + Model 6.1 | 5120 | 1e+03 | 0.5195 | 0.7249 |
| RGB + Model 7.1 | 5120 | 1e+03 | 0.4764 | 0.6962 |

## Depth-Feature Faithfulness

| Feature | MSE to GT depth feature | Mean cosine to GT |
|---|---:|---:|
| Model 1 | 0.90801 | -0.0073 |
| Model 2 | 0.89051 | 0.0228 |
| Model 3 | 1.03059 | 0.0104 |
| Model 4 | 0.14223 | 0.0066 |
| Model 5 | 0.13026 | -0.0030 |
| Model 6.1 | 0.90316 | -0.0030 |
| Model 7.1 | 0.02497 | 0.8060 |
| GT depth | 0.00000 | 1.0000 |

## Depth-Index Diagnostics

| Feature | Token accuracy | Sequence accuracy | Mean confidence |
|---|---:|---:|---:|
| Model 1 | 0.1234 | 0.0010 | 0.3872 |
| Model 2 | 0.1058 | 0.0003 | 0.4012 |
| Model 3 | 0.1295 | 0.0002 | 0.6798 |
| Model 6.1 | 0.2818 | 0.0041 | 0.3848 |

## UMAP Smoothness

| Feature | Samples | Moran's I |
|---|---:|---:|
| RGB | 3000 | 0.1906 |
| RGB + Model 1 | 3000 | 0.2154 |
| RGB + Model 2 | 3000 | 0.3435 |
| RGB + Model 3 | 3000 | 0.1793 |
| RGB + Model 4 | 3000 | 0.4716 |
| RGB + Model 5 | 3000 | 0.1815 |
| RGB + Model 6.1 | 3000 | 0.3723 |
| RGB + Model 7.1 | 3000 | 0.3497 |

## Rewritten Findings

**Finding A - RGB already carries most in-distribution geometry.** The provided RGB feature reaches R2=0.454, matching the paper's pre-VQ LAPA-LAQ reference scale rather than the finetuned-LAPA RGB reference. This cache therefore supports analysis of the frozen/pre-finetune representation, not a reproduction of the old finetuned-LAPA baseline.

**Finding B - useful depth injection is selective.** The best deployment-equivalent representation is RGB + Model 2 (R2=0.520, delta=+0.066 over RGB). The result should be interpreted as a modest decodability gain rather than a new standalone policy-success claim.

**Finding C - no-depth controls bound the parameter-only explanation.** The strongest RGB-only Stage-2.5 control is RGB + Model 3 (R2=0.457, delta=+0.003). When a depth-image variant beats this control, the gain is more plausibly tied to geometric information in the depth stream than to extra capacity alone.

**Finding D - feature-scale and target choice matter.** The continuous-distillation variants should be read together with cosine/MSE alignment to the Stage-1 depth teacher, while index-prediction variants should be read together with token and sequence accuracy. This avoids overclaiming from R2 alone.

## Suggested Replacement Text

We probe the geometric content of the deployment-equivalent representations on 138,090 LIBERO-10 frame pairs using a video-disjoint 80/20 split. Features are standardized with training statistics only and evaluated with a Ridge probe over end-effector translation magnitude. The provided RGB representation reaches R2=0.454, indicating that the cache corresponds to the pre-VQ LAPA-LAQ feature scale rather than the finetuned-LAPA reference used in the earlier appendix. Concatenating Stage-2.5 depth features gives a selective improvement: RGB + Model 2 reaches R2=0.520, a +0.066 absolute gain over RGB. RGB-only Stage-2.5 controls remain the appropriate comparison for separating depth information from added capacity. Therefore, the feature analysis supports a cautious conclusion: depth-derived representations can add geometric decodability, but the gain is incremental and should be paired with downstream policy evaluation.
