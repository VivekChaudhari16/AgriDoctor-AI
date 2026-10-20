# models/

Trained weights are **not stored in Git** (too large). Put them here:

| File | How to get it |
|---|---|
| `banana_resnet18_finetuned_best.pth` | Train with `notebooks/Project.ipynb` |
| `resnet18_finetuned.pth` | Train with `notebooks/Project.ipynb` (PlantVillage, 38 classes) |
| `agridoctor_field_model.pth` | `python training/train_field_model.py` (Maize, Soybean, Cotton) |

Tip: publish your weights on a GitHub Release or on Hugging Face and link them in the main README.
