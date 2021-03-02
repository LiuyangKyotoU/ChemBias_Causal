# ChemBias_Causal

```
# biased sampling
python sampler_qm9.py --trial $1
# train proposal model
CUDA_VISIBLE_DEVICES=$1 python train_qm9.py --trial $2 --bias $3 --alpha 100.0 --beta 0.1
# train baseline model
CUDA_VISIBLE_DEVICES=$1 python train_qm9.py --trial $2 --bias $3 --alpha 0.0 --beta 0.0
# train two step model
CUDA_VISIBLE_DEVICES=$1 python twostep_qm9.py --trial $2 --bias $3
```
