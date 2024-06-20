

# A faster implementation version of the paper "Neural Symbolic Regression that scales"


#### Due to the following two reasons, we have re-implemented a more efficient version that can be accelerated with GPU: 
1) The official implementation of the paper did not implement inference for a batch of equations; 
2) The constant optimization after transformer-based prediction in the official implementation uses BFGS, which is usually slow and leads to insufficient utilization of the advantages of using GPUs to accelerate deep learning

#### First, please download the pretrained models from the official implementation at [10M](https://drive.google.com/file/d/1cNZq3dLnSUKEm-ujDl2mb2cCCorv7kOC/view) or [100M](https://drive.google.com/drive/folders/1LTKUX-KhoUbW-WOx-ZJ8KitxK7Nov41G), and put the downloaded pretrained model into the folder ./jupyter/weights/

#### Then, install the following dependency package (more packages can be found in the official implementation):

    pip install sympytorch

#### Now, you can run the following command to perform efficient transformer-based symbolic regression:

    python NSR_fast.py

Have fun and experience the high efficiency of neural symbol regression!!!

For more usage and settings, please refer to the official implementation of the paper at https://github.com/SymposiumOrganization/NeuralSymbolicRegressionThatScales











