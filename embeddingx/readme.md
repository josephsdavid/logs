# December 15 

**Plans for today**

- run on test set
- start retraining vision models 
- write mimic dataloader (tonight)

## Test set observations

- some images do not load properly (corrupted maybe?) (TODO for Mars )
- To avoid this, see [this git issue](https://github.com/pytorch/pytorch/issues/1137)
- Just return None if exception happens and update collate fn

```python
def __getitem__(self, idx):
    try:
        img = cv2.imread(data[idx])
    except:
        return None
    return img

# ...

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
```


## Test Set Results

cstart

Task | Loss | Auroc | Auprc 
--- | --- | --- | --- 
**Atelectasis**|l1 | .617| .752
Atelectasis|l2 | .818| .880
Atelectasis|cosine | .808| .885



Task |Loss | Auroc | Auprc 
--- | --- | --- | --- 
**Cardiomegaly**|l1 | .367| .649
Cardiomegaly|l2 | .830| .933
Cardiomegaly|cosine | .706	| .853

cmid

Task |Loss | Auroc | Auprc 
--- | --- | --- | --- 
**Consolidation**|l1 | .733| .921
Consolidation|l2 | .462| .844
Consolidation|cosine | .731| .916


Task |Loss | Auroc | Auprc 
--- | --- | --- | --- 
**Edema**|l1 | .903| .937
Edema|l2 | .903| .941
Edema|cosine | .836| .886

cend

Task |Loss | Auroc | Auprc 
--- | --- | --- | --- 
**Pleural Eff.**|l1 | .885| .875
Pleural Eff.|l2 | .886| .873
Pleural Eff.|cosine | .854| .861


## Retraining Imaging Model: Notes

- Retrain imaging model on each task, with 1, 5, 10, 25, and 100% of data and new labels
- New lightning module: **EmbeddingXImagingModel**
- Lightning cannot checkpoint on val_auprc
	- figure out a hack when it matters
- Using data augmentation, should we fine tune with augmentation?
- Using pooled output for embedding
	- do not have to pad sequences


## Retraining Imaging Model: Results

- [Weights and Biases ](https://wandb.ai/djosephs/embeddingx/sweeps/suy6ri36)

Placeholder for table/figure


# Remarks: December 15

- Why is LR finder failing?
	- assumption is small percentages give it trouble, maybe best to find and log ideal lr with full dataset and log it and never run the lr finder
- Figure out checkpointing hack
- Do not have to pad sequences on BERT
- Should we be fine tuning with augmentation now?
- Getting a mysterious error with multi gpu, so just using one for now, fix tomorrow


# December 17 

- Met with Shirley today
- New direction: focus on chexpert tasks
- Review domain adaptation papers sent to us
- If poor results on chexpert tasks, adversarial or contrastive approach 
	- Probably contrastive
- Need to think of how to make this a more technical contribution
