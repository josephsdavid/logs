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
	- More domain adaptation review!
- Need to think of how to make this a more technical contribution
- Learning with noisy labels


# December 21

Plans:

- Get 1% of data full pipeline on the 5 chexpert tasks + no finding
	- Make sure to log per class auroc and auprc, dont worry about other metrics
	- Save not by loss but by auprc
- Finalize literature review on sheets
- Put any notes here
- Time allowing: Clean up discovery

Notes:

- We have interesting case of noisy labels where they are just sometimes missing


## Fixing pipeline: per class metrics

Proper per class metrics for non full task classification, and saving by auprc

```python
def evaluate_generator(metric, name):
    def result(true, probs, tasks, split):
	scores = undefined_catcher(metric, np.vstack(true), np.vstack(probs), average=None)
        tasks = [f"{split}_{task}_{name}" for task in tasks]
        scores_dict = dict(zip(tasks, list(scores)))
        scores_dict[f"{split}_{name}"] = scores.mean()
        return scores_dict
    return result

evaluate_auroc = evaluate_generator(sk_metrics.roc_auc_score, "auroc")
evaluate_auprc = evaluate_generator(sk_metrics.average_precision_score, "auprc")

# usage
auroc_dict = util.evaluate_auroc(val_true, val_probs, self.tasks, "val")
auprc_dict = util.evaluate_auprc(val_true, val_probs, self.tasks, "val")
```

## Pipeline: BERT

https://wandb.ai/djosephs/embeddingx/runs/1rd0dksu

- command: `./run.sh -l config/train_nlp.sh`
- Might consider switching to 5%


## Pipeline: Distance minimization

Placeholder

- command: `./run.sh -l config/joint_sim_frozen_bert.yaml` update with bert weights first


## Pipeline: Fine tune

data augmentation here

Placeholder

- command: `./run.sh -l config/fine_tune_base.yaml` NEEDS to be written with weights from distance!


## Pipeline: notes

Need to debug lr finder



## Lit review

- Key paper for noisy labels: https://export.arxiv.org/pdf/1809.01465
- TODO (tonight): Take offline notes for noisy labels and add to sheets
- Take Shirley DA notes + my notes and add to sheets
- Other notes: adversarial training may work for both noisy labels and domain adaptation, we might want to pursue if our current approach does not work out
