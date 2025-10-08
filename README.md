This is the PyTorch code of the <a href="https://arxiv.org/abs/2201.12086">BLIP paper</a> [[blog](https://blog.salesforceairesearch.com/blip-bootstrapping-language-image-pretraining/)]. 

## 🚀 Environment Setup

We recommend creating a new conda environment before installing the dependencies:

```bash
conda create -n d2c2 python=3.12
pip install -r requirements.txt

### Finetuned checkpoints:
BLIP retrieval w/ ViT-L CUHK-PEDES finetuned | Top-1
--- | :---:
<a href="https://drive.google.com/file/d/16nB0Kb66wEs4qb8K6Od7RXneuOuXHB2R/view?usp=drive_link">Download</a> | 78.4
 

### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{li2022blip,
      title={BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation}, 
      author={Junnan Li and Dongxu Li and Caiming Xiong and Steven Hoi},
      year={2022},
      booktitle={ICML},
}</pre>

### Acknowledgement
The implementation of D2C2 relies on resources from <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/anosorae/IRRA.git">IRRA</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a> and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
