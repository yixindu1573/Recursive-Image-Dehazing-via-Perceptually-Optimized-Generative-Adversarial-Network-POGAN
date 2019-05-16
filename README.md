# Recursive-Image-Dehazing-via-Perceptually-Optimized-Generative-Adversarial-Network-POGAN

Here is the code for our paper entitled "Recursive Image Dehazing via Perceptually Optimized Generative Adversarial Network (POGAN)".

The code has been tested on Tensorflow 1.4.0.

To do testing:

    python main.py --mode=e

      
To do training, simply put original and hazy images in folder data/train_ori and data/train_haze respectively, then run:

    python main.py
    
If find the code useful, please consider cite our work:


    @inproceedings{du2019recursive,
      title={Recursive Image Dehazing via Perceptually Optimized Generative Adversarial Network (POGAN)},
      author={Du, Yixin and Li, Xin},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
      year={2019}
    }
    
If you have any question, please contact yixindu1573@gmail.com

We acknowledge and thank the author of SRGAN for sharing their source code:
    https://github.com/tensorlayer/srgan
