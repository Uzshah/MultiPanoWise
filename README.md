# MultiPanoWise
This is the pytorch implementation of MultiPanoWise (MultiPanoWise: holistic deep architecture for multi-task dense prediction from a single panoramic image, CVPRW 2024)!

# Methodology

![process](./img/process_diagram.jpg)
![adjustment](./img/architecture.jpg)

# Poster

![poster](./paper/MultiPanoWise_OmniCV_Poster.pdf)
# PreTrained weights
We updated the models trained for Structured3D in this *[link (click me)](https://drive.google.com/drive/folders/1nmf_QOnCXctaXqQP-fQTAfn_49ca2LXa?usp=sharing)*, now you can download and test it! If you have downloaded it and put it in the correct folder. You can run:


Single-GPU
```bash
python main.py --batch-size 1 --num_epochs 0 --data_path path/to/dataset --load_weights_dir path/to/weights
```
Multi-GPUs
```bash
python -m torch.distributed.launch --nproc_per_node=$nGPUs --use_env main.py --batch-size 1 --num_epochs 0 --data_path path/to/dataset --load_weights_dir path/to/weights
```


# Acknowledgements
We thank the authors of the project below:

[PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer)

```
@inproceedings{shen2022panoformer,
  title={PanoFormer: Panorama Transformer for Indoor 360$$\^{}$\{$$\backslash$circ$\}$ $$ Depth Estimation},
  author={Shen, Zhijie and Lin, Chunyu and Liao, Kang and Nie, Lang and Zheng, Zishuo and Zhao, Yao},
  booktitle={European Conference on Computer Vision},
  pages={195--211},
  year={2022},
  organization={Springer}
}
```
If you like this work please cite this:
```
@inproceedings{Shah:2024:PSG,
    author = {Uzair Shah and Muhammad Tukur and Mahmood Alzubaidi and Giovanni Pintore and Enrico Gobbetti and Mowafa Househ and Jens Schneider and Marco Agus},
    title = {{MultiPanoWise}: holistic deep architecture for multi-task dense prediction from a single panoramic image},
    booktitle = {Proc. OmniCV - IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year = {2024},
    abstract = { We present a novel holistic deep-learning approach for multi-task learning from a single indoor panoramic im- age. Our framework, named MultiPanoWise, extends vi- sion transformers to jointly infer multiple pixel-wise sig- nals, such as depth, normals, and semantic segmentation, as well as signals from intrinsic decomposition, such as re- flectance and shading. Our solution leverages a specific ar- chitecture combining a transformer-based encoder-decoder with multiple heads, by introducing, in particular, a novel context adjustment approach, to enforce knowledge distil- lation between the various signals. Moreover, at train- ing time we introduce a hybrid loss scalarization method based on an augmented Chebychev/hypervolume scheme. We demonstrate the capabilities of the proposed architec- ture on public-domain synthetic and real-world datasets. We showcase performance improvements with respect to the most recent methods specifically designed for single tasks, like, for example, individual depth estimation or semantic segmentation. To the best of our knowledge, this is the first architecture able to achieve state-of-the-art performance on the joint extraction of heterogeneous signals from single in- door omnidirectional images },
    note = {To appear},
    url = {http://vic.crs4.it/vic/cgi-bin/bib-page.cgi?id='Shah:2024:PSG'},
}
```

