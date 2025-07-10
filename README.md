# The-Nah-Bandit

This is the codebase for our paper:  
**["The Nah Bandit: Modeling User Non-compliance in Recommendation Systems"](https://arxiv.org/abs/2408.07897)**.

We propose the **Expert with Clustering (EWC)** algorithm, designed to handle user non-compliance in recommendation systems. This repository includes experiments on **travel route recommendation** and **restaurant recommendation**.

---

## 🔧 Setup & Data Generation

### Travel Route Recommendation

The travel route dataset is synthetically generated based on survey data. To generate data:

```bash
cd travel_route_rec/data_generation
python data_gen.py --beta_scaler 0
```

- `--beta_scaler` controls the compliance level of the user population.
- You can change it to other values used in the paper, such as `1` or `10`.

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

Plot the **ablation study** results:

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

Plot the **ablation study** results:

```bash
python plot_result.py --EXPERIMENT_NAME ablation_study
```

---

## 📎 Notes

- Ensure all dependencies are installed (see `environment.yml` if available).
- For questions or issues, feel free to open an Issue or contact the authors.

---
