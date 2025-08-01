# The Nah Bandit

This is the codebase for our paper:  
**["The Nah Bandit: Modeling User Non-compliance in Recommendation Systems"](https://arxiv.org/abs/2408.07897)** (IEEE Transactions on Control of Network Systems).

In this paper, we address a key problem in recommendation systems: users can easily opt out of recommended options and revert to their baseline behavior. This phenomenon is common in real-world scenarios such as shopping and mobility recommendations. We name this problem the **Nah Bandit**, which lies between a typical bandit setup and supervised learning. The comparison is shown below:

|                          | User selects from **recommended** options | User selects from **all** options |
|--------------------------|------------------------------------------|----------------------------------|
| User is influenced by recommendations   | Bandit       | **Nah Bandit** (This work)     |
| User is **not** influenced by recommendations | N/A          | Supervised Learning            |

We propose a **user non-compliance model** to solve the Nah Bandit problem, which uses a linear function to parameterize the **anchoring effect** (user’s dependence on the recommendation). Based on this model, we propose the **Expert with Clustering (EWC)** algorithm to handle the Nah Bandit problem.
<figure style="text-align: center;">
<img src="readme_figures/overview_figure.png" alt="overview_figure" width="1000"/>
<figcaption>Figure 1: An overview figure of Expert with Clustering (EWC) algorithm.</figcaption>
</figure>

---

## 🔧 Setup & Data Generation

### Travel Route Recommendation

To generate synthetic travel route data:

```bash
cd travel_route_rec/data_generation
python data_gen.py --beta_scaler 0
```

- `--beta_scaler` controls the compliance level of the user population.
- Try values like `1` or `10` as used in the paper.

---

## 🚀 Running Experiments

### Travel Route Recommendation

Run the main experiment:

```bash
cd ../
python main.py --NUM_TEST 5 --beta_scaler 0
```

Plot the results:

```bash
python plot_result.py --beta_scaler 0
```

Plot the **ablation study**:

```bash
python plot_result.py --beta_scaler 0 --EXPERIMENT_NAME ablation_study
```

---

### Restaurant Recommendation

Run the experiment: 

```bash
cd restaurant_rec
python main.py --NUM_TEST 5
```

Plot the results:

```bash
python plot_result.py
```

Plot the **ablation study**:

```bash
python plot_result.py --EXPERIMENT_NAME ablation_study
```

---

## 📎 Notes

- Ensure all dependencies are installed (see `environment.yml` if available).
- For questions or issues, feel free to open an Issue or contact the authors.

---

## 📊 Experimental Results

This repository includes experiments on **travel route recommendation** and **restaurant recommendation**. Experimental results show that EWC outperforms both supervised learning and traditional contextual bandit approaches.

<table>
  <tr>
    <td align="center">
      <img src="readme_figures/beta=0_comparison.png" width="330px"><br/>
      <sub>&beta; = 0</sub>
    </td>
    <td align="center">
      <img src="readme_figures/beta=1_comparison.png" width="330px"><br/>
      <sub>&beta; = 1</sub>
    </td>
    <td align="center">
      <img src="readme_figures/beta=10_comparison.png" width="330px"><br/>
      <sub>&beta; = 10</sub>
    </td>
  </tr>
</table>

**Figure 2**: Regret of Expert with Clustering (EWC, Ours) and other baselines (DYNUCB, LinUCB, the user non-compliance model, and XGBoost) on travel route recommendation data. The x-axis denotes decision rounds; the y-axis shows regret (lower is better). EWC consistently outperforms baselines under different user compliance levels (&beta;).

<p align="center">
  <img src="readme_figures/restaurant.png" width="300px">
</p>

**Figure 3**: Regret of Expert with Clustering (EWC, Ours) and other baselines (XGBoost, LinUCB, DYNUCB, and the user non-compliance model) on restaurant recommendation data. EWC achieves lower regret than all baselines across all decision rounds.

---