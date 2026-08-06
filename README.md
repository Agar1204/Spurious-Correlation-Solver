# Spurious-Correlation-Solver

Implements the [GEORGE](https://arxiv.org/abs/2011.12945) pipeline to fix a model that has
learned a spurious correlation instead of the actual predictive features, evaluated on
SpuCoMNIST.

## The problem

Spurious correlations occur in image classification when non-predictive features present in training data are highly correlated with certain classes, but are not predictive of those classes. If these features are easier for the model to comprehend than the actual predictive features, the model will learn to classify on these features and fail to generalize when those features aren't present in newer data. It will also improperly classify samples containing the spurious correlations in different classes of objects. 

A popular example is when classifying landbirds and waterbirds. Rather than learning about the distinct features of the animals, the model may lazily pick up on whether the background is land or water. When a waterbird is on land or a landbird is flying over water, the model will struggle immensely.


## The pipeline

GEORGE recovers the true (class, background) groups without any group labels, then
retrains with those groups balanced:

1. **ERM** — train a plain classifier. It will absorb the spurious correlation.
2. **Cluster** — run KMeans on the model's penultimate-layer features, per class, to
   separate examples by background.
3. **Group-balanced retraining** — train a fresh model with `GroupBalanceBatchERM`,
   sampling so every inferred group contributes equal mass per batch.

## A bug I faced: clustering a model that has already memorized the shortcut

The first pass through this pipeline clustered features from the **fully-trained**
10-epoch ERM model. By epoch 10, the model has fit even the ~1% minority examples, which
pulls their features inward until they sit on top of the majority examples of the same
class. There is nothing left in that feature space for KMeans to separate — the clusters
it found didn't track background at all, so group-balanced retraining barely moved worst-
group accuracy (10.2% → 28.0%).

**Fix:** cluster features from a deliberately **under-trained** snapshot (1 epoch) instead.
Early in training, the background is the dominant, easiest-to-fit signal, and the model has
not yet memorized the minority examples — so their features are still geometrically
separated from the majority blob, and KMeans can actually recover the groups.

## Result

| stage | average accuracy | worst-group accuracy |
|---|---|---|
| ERM (baseline) | 69.1% | 10.2% |
| GEORGE, clustered on overfit features | 69.3% | 28.0% |
| **GEORGE, clustered on underfit features** | **89.9%** | **66.0%** |

Clustering on underfit features closes most of the gap: worst-group accuracy goes from
being 6x worse than average to being roughly in line with it, with no loss in overall
accuracy. Group-balanced retraining was correct all along — the inferred groups were the
part that needed to actually reflect the true background, and that only happens if you
extract them before the model has had the chance to memorize past them.
