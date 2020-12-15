# December 15 

## Plans for today

- run on test set
- write mimic dataloader (tonight)
- start retraining vision models (day)

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


# Test Set Results

- Scroll down to see results!
- Produced by using our best val loss models on the test set
- See script test_finetune.py
- Needs to be made modular


##

 Loss | Auroc | Auprc | Accuracy | Precision | Recall
 --- | --- | --- | --- | --- | --- 
 l1 | 0| 0| 0| 0| 0
 l2 | 0| 0| 0| 0| 0
 cosine | 0| 0| 0| 0| 0


## Cardiomegaly Results

 Loss | Auroc | Auprc | Accuracy | Precision | Recall
 --- | --- | --- | --- | --- | --- 
 l1 | 0| 0| 0| 0| 0
 l2 | 0| 0| 0| 0| 0
 cosine | 0| 0| 0| 0| 0

## Consolidation Results

 Loss | Auroc | Auprc | Accuracy | Precision | Recall
 --- | --- | --- | --- | --- | --- 
 l1 | 0| 0| 0| 0| 0
 l2 | 0| 0| 0| 0| 0
 cosine | 0| 0| 0| 0| 0

## Edema Results

 Loss | Auroc | Auprc | Accuracy | Precision | Recall
 --- | --- | --- | --- | --- | --- 
 l1 | 0| 0| 0| 0| 0
 l2 | 0| 0| 0| 0| 0
 cosine | 0| 0| 0| 0| 0

## Pleural Effusion Results

 Loss | Auroc | Auprc | Accuracy | Precision | Recall
 --- | --- | --- | --- | --- | --- 
 l1 | 0| 0| 0| 0| 0
 l2 | 0| 0| 0| 0| 0
 cosine | 0| 0| 0| 0| 0

