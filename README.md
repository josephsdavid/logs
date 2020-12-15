# My Work Logs

Experiment history etc. 
Each project has its own directory with the project logs as a readme. 

# Making presentations

To make presentations with this script, please install the `nix` package manager, everything else is self contained. usage below:

```bash
./md2slides.sh PROJECT_DIR [AUTHORS] [FILE TO CONVERT]
```

authors defaults to my name, file to convert defaults to readme.md

# Current things

Images: Normal markdown image format

TODO:
- [ ] make markdown tables easy (flag + awk??)
- [ ] make two column format available and easy

Insert this, with maybe a regex identifiable flag

```css
<style>
.container{
    display: flex;
}
.col{
    flex: 1;
}
</style>

<div class="container">

<div class="col">
Column 1 Content
</div>

<div class="col">
Column 2 Content
</div>

</div>
```
