# ARMR: Adaptively Responsive Network for Medication Recommendation

Code for our IJCAI-25 paper: Adaptively Responsive Network for Medication Recommendation.

This is an implementation of our model and the baselines in the paper.

## Process Data

1. Go to [mimiciii](https://physionet.org/content/mimiciii/1.4/) and [mimiciv](https://physionet.org/content/mimiciv/3.1/) to download the raw datasets (You may need to get the certificate)

2. For a fair comparison, we use the same pre-processing scripts used in  [Carmen](https://github.com/bit1029public/Carmen), please refer to Carmen for more information. To make it more intuitive, we convert the pkl data to json.

## Train & Test Model

```bash
./run.sh
```

## Citation

If the repo and the paper are useful for you, it is appreciable to cite our paper:

```bibtex
@inproceedings{wu2025armr,
  title={ARMR: Adaptively Responsive Network for Medication Recommendation},
  author={Wu, Feiyue and Wu, Tianxing and Jing, Shenqi},
    booktitle = {Proceedings of the 34th International Joint Conference on
               Artificial Intelligence, {IJCAI} 2025},
    year = {2025}
}
```

## Thanks

The code refers to the following repos: [LEADER](https://github.com/liuqidong07/LEADER-pytorch), [COGNet](https://github.com/BarryRun/COGNet), [Carmen](https://github.com/bit1029public/Carmen), [SafeDrug](https://github.com/ycq091044/SafeDrug), [GAMENet](https://github.com/sjy1203/GAMENet), [MoleRec](https://github.com/yangnianzu0515/MoleRec). Be sure to check it out!
