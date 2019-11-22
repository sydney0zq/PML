README.txt

:Author: qiang.zhou
:Email: theodoruszq@gmail.com
:Date: 2018-10-11 21:46



Project description:

    This project dedicates to reproduce CVPR 18 paper 'Blazingly Fast Video Object 
    Segmentation with Pixel-Wise Metric Learning'. Author Chen wants to firstly
    embed frames into an embedding space, and then use metric learning to retrieve
    foreground and background pixels under the guide of first frame and annotation, 
    which is a novel way to do Video Object Segmentation task.

    This project tries to reproduce the results reported in his paper, but finally
    has a gap about 0.8~1.5. However, I think it is enough to do further research.


================================================================

Deep learning:

    1. Data preparation
        DAIVS
        + trainval
          + Annotations
          + ImageSets
          + JPEGImages
        + testdev
        + testchallenge
        You could download them from `http://davischallenge.org`.

    2. Init model preparation
        init_models/deeplabv2_voc.pth
        
        Deeplab pretrained model is borrowed from 
        `https://github.com/speedinghzl/Pytorch-Deeplab`, download it by yourself.
        Or download from: https://drive.google.com/open?id=19bHrNKQs4JzqZpoPSO5ntwMbqWQU8TIJ

    3. Start to train
        This project could train with single or multi GPU(s). You could choose one
        depending on resources you own.

        :Single GPU:
        `CUDA_VISIBLE_DEVICES=0 python3 train.py --batch_size 4 \
                                                 --num_epochs 100 \
                                                 --learning_rate 2.5e-5 \
                                                 --alpha 0.7 \
                                                 --image_size 321 321 \
                                                 --gpus 0 \
                                                 --log_file ./experiments/run.log`
    
    4. Evaluate on DAVIS 16 val dataset
        As author Chen introduces `Bilater Solver`, which is a post-process for refine
        upsampled masks, it locates in `PROJ_ROOT/net/bs.py`，and you could run test by:

        `CUDA_VISIBLE_DEVICES=0 python3 infer_bs.py`


================================================================

Coda:

    Author Chen doesnot open this project's source code, therefore I could not make sure
    my implementation absoultely right. 

    The accuracy report in paper:

    Spat.-Temp.     Online Adapt.           Mean J          Mean F      Mean J&F
                                            72.0            73.6        72.8
                        √                   73.2            75.0        74.1
        √                                   74.3            78.1        76.2
        √               √                   75.5            79.3        77.4

    ---

    My implemetation(Stable result):

    Spat.-Temp.     Online Adapt.           Mean J          Mean F      Mean J&F

        √                                   73.5










