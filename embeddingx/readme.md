# December 15 

## Plans for today

- run on test set
- write mimic dataloader (tonight)
- start retraining vision models (day)

## Test set observations

- some images do not load properly (corrupted maybe?)
- To avoid this, see [this git issue](https://github.com/pytorch/pytorch/issues/1137)
- Just return None if exception happens and update collate fn

```python
collate_fn = lambda l: [x for x in l if x is not None]
```