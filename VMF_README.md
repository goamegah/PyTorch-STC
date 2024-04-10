# Von Mises-Fisher (VMF)

The Von Mises-Fisher (VMF) algorithm is a method used for clustering directional data, often applied in fields such as computer vision, natural language processing, and bioinformatics. It is an extension of the k-means algorithm, designed specifically for data that lies on a hypersphere. **In our case we face to directional data, text or documents**.

Text documents can be represented as vectors in high-dimensional space, and when analyzing similarities between documents or words, considering the **direction (or angle)** between vectors can be more meaningful than considering their magnitudes alone.


Other fields of application:

- Computer Vision: Features extracted from images, such as gradients or orientations, can be represented directionally. Clustering such data with VMF can help in tasks like object recognition or scene understanding.

- Geographical Data: Directional data often arise in studies involving geographical information, such as wind direction or animal migration paths.

- Bioinformatics: In analyzing gene expression data or protein structures, directional patterns may emerge, making VMF clustering applicable.


## Installation

based on what is publicly available, you might need (for python users), package **spherecluster**. The package is currently being updated. Nevertheless, you can follow the instructions to see the program running.

- Set *REQUIRES_FILE* variable in **setup.py** to vmf_requiments.txt

- Create virtual environment 

We are considering **venv** but feel free to other tools available.


```
$ python -m venv torchSTC
$ source torchSTC/bin/activate
$ pip install .
```


 In such case you might want to make visualisation or use PyTorch libs like *torchinfo*, you have to lunch instead command below  

 ```
$ python -m venv torchSTC
$ source torchSTC/bin/activate
$ pip install ".[dev, vis]"
```

## Other library for VMF

Feel free to suggest other libs or efficient implementation of VMF.

## Acknowledgments
The authors would like to thank the anonymous
reviewers for their constructive feedback.
