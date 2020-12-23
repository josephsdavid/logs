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


# December 21/22

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

## Pipeline: BERT 1%

https://wandb.ai/djosephs/embeddingx/runs/1rd0dksu

- command: `./run.sh -l config/train_nlp.sh`
- stdout logged to: `train_nlp.log`
- Weights saved at `/data/embeddingx/ckpts/medical_bert/iter_2/all/0/0.01/0.601`
	- TODO: update weight saving so i dont have to sort through timestamps in this script
- Might consider switching to 5%


## Pipeline: BERT 1% results

- **val auroc**: 0.707
- **val_auprc**: 0.601
- **step**: 87
- TODO: rerun on 5% after this set of runs finishes


## Pipeline: Distance minimization  1%

https://wandb.ai/djosephs/embeddingx/sweeps/deo0pf2a

- command: `./run.sh -l config/joint_sim_frozen_bert.yaml` update with bert weights first
- stdout logged to: `joint_sim_frozen_bert.log`
- Results stored in :

```python
filepath = f"{args.weights_save_path}/new_bert_embedding/{args.similarity_metric}/{args.task}/classifier_test/"
```
- TODO: update where they are stored


## Pipeline: Distance minimization results  %


Loss_fn | Val Auroc | Val Auprc | Val Loss | Train Loss
 --- | --- | --- | --- | ---
l1 | .522| .367 | .352 | .291
l2 | .582 | .4  | 13.45 | 11.55
cosine | .485 | .345  | .322 | .211


## Pipeline: Fine tune 1%

https://wandb.ai/djosephs/embeddingx/sweeps/opapysq5
try2: https://wandb.ai/djosephs/embeddingx/sweeps/3c99yw8h

- command: `./run.sh -l config/fine_tune_base.yaml`
- Data augmentation this time
- stdout logged to: `fine_tune_base.log`

## Pipeline: Finetune results 1%

NOTE: rerunning currently with different LR to verify it is this bad (it is looking better but still not awesome)


Loss_fn | Val Auroc | Val Auprc 
 --- | --- | --- 
l1 | .468| .31 
l2 | .5 | .3507  
cosine | .5902 | .438  

## Pipeline: notes

Need to debug lr finder

We can train BERT AND a fine tune model at the same time, more efficiency in iterative runs

## Pipeline: 10%

[Worse performance on 5% of data for BERT](https://wandb.ai/djosephs/embeddingx/runs/243wehbl?workspace=user-djosephs) -> [try 10%](https://wandb.ai/djosephs/embeddingx/runs/1v8wie8q?workspace=user-djosephs)

## Worries and thoughts

- Maybe we focus on framing our existing work as a clinical paper and then some other approach for a technical paper? 
- As soon as we got these physician labels things have gotten further and further from our assumptions
- Assumptions were made when we were getting 100% on the NLP labels, this is not the case
	- We have a break from deadlines, maybe instead we just try some experiments on new approaches during this relatively free time
	- We have a robust enough lit review we can try other things

## Lit review

- Loss based methods focus on softmax classification and binary classification, I am not sure how they extend to our multilabel scenario
	- I do not have the math background yet
- We have an interesting case where labels AND sometimes information is missing from the training data.
- I think we should therefore focus on how we can use our validation dataset (or some other subset of physician labels) to regularize our training.
- Two key approaches:
	- [Bilevel optimization (ECCV 2018 (30))](http://openaccess.thecvf.com/content_ECCV_2018/papers/Simon_Jenni_Deep_Bilevel_Learning_ECCV_2018_paper.pdf)
	- [SOSOLETO (ICLR 2019 best paper)(4)](https://arxiv.org/abs/1805.09622)
- I really like Shirley's citation format with the [CONFERENCE YEAR] (Citations), we should use this


