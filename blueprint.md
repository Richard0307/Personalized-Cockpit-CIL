# System Blueprint: Open-World CIL for Intelligent Cockpit

## 1. Problem Formulation

Before implementing the architecture, the data flow and boundary conditions processed by the system must be rigorously defined.

* **Setting:** The model learns on a continuous data stream $\mathcal{D} = \{\mathcal{D}_1, \mathcal{D}_2, ..., \mathcal{D}_T\}$, where $T$ is the total number of incremental task stages.
* **Input:** The training set for stage $t$ is $\mathcal{D}_t = \{(x_i^t, y_i^t)\}_{i=1}^{N_t}$, where $y_i^t \in \mathcal{C}_t$. $\mathcal{C}_t$ denotes the set of novel classes introduced in the current stage.
* **Constraint:** At stage $t$, the system is strictly prohibited from accessing the complete historical data $\mathcal{D}_{1:t-1}$. It can only access a severely memory-constrained local Replay Buffer $\mathcal{M}$.
* **Objective:** After training at stage $t$, the model must maintain high recognition performance on the union of all seen classes ($\bigcup_{i=1}^t \mathcal{C}_i$) while minimizing the catastrophic forgetting of older classes.

---

## 2. Core Modules

The system architecture must be strictly decoupled into the following four independent modules at the code level.

### Module A: Frozen Perception Backbone
* **Academic Requirement:** To meet the stringent power and thermal constraints of automotive edge chips, the backbone network must exhibit strong zero-shot generalization and remain entirely non-updatable locally.
* **Code Implementation Target:** 1. Instantiate a pre-trained Vision Transformer (e.g., `ViT-B/16` with ImageNet-21k weights).
    2. Explicitly freeze the backbone via code: iterate through model parameters and enforce `requires_grad = False`.
    3. Input/Output Constraints: For an input image $x$, output a high-dimensional feature representation $h = f(x)$ with a fixed dimensionality (e.g., $d = 768$).

### Module B: Open-World Novelty Detector
* **Academic Requirement:** The system must actively intercept unknown actions (Out-of-Distribution detection) rather than forcibly categorizing novel behaviors into known historical classes.
* **Code Implementation Target:** 1. Calculate the distance between the input feature $h$ and all known class prototypes. Implementation should prioritize the Mahalanobis Distance or an Energy-based Score.
    2. Establish a dynamic threshold $\tau$. If $Score(h) > \tau$, the input is flagged as an OOD sample (a novel driver habit), which subsequently triggers Module C.

### Module C: Parameter-Efficient CIL Updater
* **Academic Requirement:** Discard traditional full-network Fine-Tuning. Enforce Parameter-Efficient Fine-Tuning (PEFT) techniques to accommodate the VRAM limits of single-card edge devices (e.g., RTX 4090).
* **Code Implementation Target:** 1. Implement a lightweight Adapter layer (typical architecture: down-projection FC $\rightarrow$ non-linear activation $\rightarrow$ up-projection FC).
    2. Insert the Adapter into the Transformer Blocks of the ViT (either in parallel or sequentially).
    3. During the backward pass, the optimizer (e.g., `torch.optim.AdamW`) is exclusively permitted to update the parameters of the Adapters and the final Classifier Head.

### Module D: Episodic Memory & Distillation
* **Academic Requirement:** Strictly limit the memory footprint and utilize Knowledge Distillation to anchor the decision boundaries of historical features.
* **Code Implementation Target:** 1. **Herding Selection:** Implement an algorithm to retain a maximum of $K$ exemplars (e.g., $K=20$) per class that are nearest to the feature mean of their respective class.
    2. **Loss Function:** The total loss function must be formulated as a combination of two objectives:
       $$\mathcal{L}_{total} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \mathcal{L}_{KD}(p_{old}, p_{new})$$
       Where $\mathcal{L}_{CE}$ is the Cross-Entropy loss for novel classes, and $\mathcal{L}_{KD}$ is the Distillation loss to preserve the output distributions of old classes.

---

## 3. Evaluation Protocol

The repository must implement automatic logging and curve plotting for the following two standard CIL evaluation metrics.

* **Average Accuracy (AA):** Evaluates the overall recognition capability of the model across all historical tasks after learning $T$ tasks.
    $$AA = \frac{1}{T} \sum_{i=1}^T A_{T,i}$$
    *(Where $A_{T,i}$ denotes the accuracy on the test set of task $i$ after completing the training of task $T$.)*
* **Average Forgetting (AF):** The core metric for quantifying catastrophic forgetting. 
    $$AF = \frac{1}{T-1} \sum_{i=1}^{T-1} \max_{t \in \{1, ..., T-1\}} (A_{t,i} - A_{T,i})$$
    *(A lower AF value indicates stronger anti-forgetting robustness.)*

---

## 4. Experimental Setup

* **Dataset Partitioning:** Implement a `data_loader.py` script to artificially partition public datasets into incremental stages (e.g., an initial base phase of 50 classes, followed by subsequent phases adding 10 classes each).
* **Baseline Comparison:** The framework must reserve interfaces for standard control groups. It is mandatory to benchmark against naive Fine-Tuning (to establish the upper bound of forgetting) and standard iCaRL (to validate the computational advantages of the proposed lightweight architecture).