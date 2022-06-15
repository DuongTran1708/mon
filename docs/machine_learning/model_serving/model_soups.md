---
layout      : default
title       : Model Soups
parent	    : Model Serving
grand_parent: Machine Learning
has_children: false
has_toc     : false
permalink   : /machine_learning/model_serving/model_soups
---

# Model Soups: Averaging Weights of Multiple Fine-tuned Models Improves Accuracy without Increasing Inference Time

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs,
Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair
Carmon, Simon Kornblith, and Ludwig Schmidt

arXiv 2022

[Paper](data/model_soups.pdf){: .btn .fs-3 .mb-4 .mb-md-0 }
[Code](https://github.com/Burf/ModelSoups){: .btn .fs-3 .mb-4 .mb-md-0 }
[Reference](https://medium.com/@sabrinaherbst/model-soups-for-higher-performing-models-1d4818126191){: .btn .fs-3 .mb-4 .mb-md-0 }

<details open markdown="block">
  <summary>Table of contents</summary>
  {: .text-delta }
  1. TOC
  {:toc}
</details>

---

https://user-images.githubusercontent.com/6915498/172331427-f9b7a0c8-d8d3-470d-9c94-843afe2662e8.mov

## Highlight

Recently, researchers from various universities and companies published a paper
on ArXiv, introducing a concept called **Model Soups**. Model Soups are supposed
to boost the accuracy of machine learning models by averaging the weights of
models trained with various hyperparameters.

## Method

### Current Training Workflow

Usually, after having picked out a model, data scientists will start to train
the model with different hyperparameters and figure out which configurations
work best. A lot of times the best configuration is determined either by a
simple train-test split or a more rigorous Cross-Validation.

This one configuration will then be used while all other configurations (which
might only be slightly worse than the best one) are discarded.

### Model Soups

> Model Soups allow averaging the weights of the models without requiring
> additional memory or inference time. This is an advantage when compared to
> ensemble methods. Still, Model Soups increase the robustness as ensemble
> methods do.

Wortsman et al. [1] came up with three different approaches to so-called Model
Soups: **Uniform Soups**, **Greedy Soups**, **Learned Soups**.

### Uniform Soups

Uniform Soups average the weights of all trained models. In the experiments
conducted, this method mostly ended up achieving a worse performance than the
best performing model alone.

### Greedy Soups

Greedy soups rely on ordering the different models, starting with the best one.
These models will be added one by one and, if the performance increases, the
model will be added to the final model.

> This change will make sure that the resulting Model Soup will have a better
> performance than the single best one would have on its own.

### Learned Soups

The learned soup uses gradient-based minibatch optimisation. The main idea is to
remove the sequential step from Greedy Soups. The approach requires an
additional validation set and has the advantage of requiring only a single pass
through this set.

> Still, the approach requires having all models in memory at runtime, which
> results in a big limitation. Given the size of today’s most commonly used
> neural networks, it is hardly possible to compute a Learned Soup.

## Ablation Studies

## Results

## Citation