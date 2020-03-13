<!--- README template from https://github.com/Neighborhood-Traffic-Flow/neighborhoodtrafficflow -->

# Multi-Modality Optimal Radiation Therapy (MMORT)
[![GitHub license](https://img.shields.io/github/license/MMORT/mmort)](./LICENSE)

This repository contains the code for the paper [A Proof of Principle: Multi-Modality Radiotherapy Optimization](https://arxiv.org/abs/1911.05182).
We developed non-convex bilevel optimization framework for combining different radiation modalities in optimal treatment planning for radiation therapy cancer patients.
Multi-modality radiation therapy is not used in current clinical practice and this work strives to reinforce the growing interest in this idea by developing methods 
for the multi-modality treatment planning and providing a proof of concept as well as some real patient experiments (the latter is ongoing work).

## Installation and Use

#### To clone the repository:
```
git clone https://github.com/LevinRoman/MMORT
cd MMORT
```

#### To run the experiments script (here alpha, beta, gamma, delta_mean and delta_max are radiobiological parameters described in the paper):
```
cd MMORT
python experiments.py --alpha 0.35 --beta 0.175 --gamma 0.35 --delta_mean 0.07 delta_max 0.175

```

## Project structure
* `mmort`: Main directory with the scripts
  * `optimization_tools.py`: script for fluence map u optimization
  * `experiments.py`: main script for running the multi-modality optimization

## License

This project is [MIT](./LICENSE) licenced.
