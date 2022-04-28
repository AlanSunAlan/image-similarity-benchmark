# Benchmarking Image Similarity Metrices
## How to run the code
1. Install pre-requested packages:

```bash
    pip3 install opencv-python pillow scikit-image scipy==1.1.0 flask
```

2. Run the benckmark:
```bash
    python3 benchmark.py
```
3. The information of the benchmarking result will be printed and saved in folder ./BenchmarkResult.
## Note:
1. scipy of version higher than 1.1.0 is not compatible with lower versions and thus the code won't work with the latest versions. Please be sure to install the old version.
2. Part of the code (specifically structural_sim(), pixel_sim(), sift_sim() as well as corresponding static functions) in similarity.py is from https://github.com/jqueguiner/image-similarity.