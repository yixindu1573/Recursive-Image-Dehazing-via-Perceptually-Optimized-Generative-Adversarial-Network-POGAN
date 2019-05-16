# Recursive-Image-Dehazing-via-Perceptually-Optimized-Generative-Adversarial-Network-POGAN
Recursive Image Dehazing via Perceptually Optimized Generative Adversarial Network (POGAN)

Here is the code for our paper entitled "Recursive Image Dehazing via Perceptually Optimized Generative Adversarial Network (POGAN)".

The code has been tested on Tensorflow 1.4.0.

To do testing:

    python main.py --mode=e

      
To do training, put your original and hazy images in folder data/train_ori and data/train_haze respectively, then run:

    python main.py
    
If you use this code, please cite our work:

@inproceedings{du2019recursive,
  title={Recursive Image Dehazing via Perceptually Optimized Generative Adversarial Network (POGAN)},
  author={Du, Yixin and Li, Xin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  year={2019}
}

We acknowledge and thank the author of SRGAN for sharing their source code:
    https://github.com/tensorlayer/srgan
