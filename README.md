# ChemBias_Causal

```
# biased sampling
python sampler.py
# train baseline, previous model, proposed model
# e.g., task is predict 2th propery of QM9
CUDA_VISIBLE_DEVICES=$1 python train.py --task 'qm9_2' --trial 0
```
