# Project Write-Up


## Explaining Custom Layers

The process behind converting custom layers involves, in general one of these two alternatives:
- Create a CPU extension or GPU extension that allow us create de layer in the IR representation and then, the inference could be executed by the created extension. In order to do that, OpenVINO toolkit provide a tool for generate source code templates that once properly modified and compiled with the right arguments and custom functions/code it can be used for inform the model optimizer where are the custom layers in the source model, and to use cpu/gpu extensions in the inference step for execute custom layers.
- Other alternative is, in the inference step, delegate to  the original framework (Caffe, TensorFlow, etc.) the execution of customer layers. This solution has some drawbacks: The need of install the source framework, probably poor performance, and the need to modify code in source framework to acount for dynamic parameters.
- Also, in tensorflow allow a third alternative related to subsitute some no supported subgraphs by other subgraphs supported by Openvino. 

Some of the potential reasons for handling custom layers are the use of more special behaviour or configuration not found in standard IR, that allow better performance in accuracy if need, or better performance in speed in the inference phase. There are also some situations, probably in new researchs (Transformers, BERT, graph networks,), that some features or functionality are not supported by OpenVINO and is needed to use a Custom Layer.

## Comparing Model Performance

I have tested four methods from Caffe framework:
- SSD 300 with weights to FP16
- SSD 512 with weights to FP16
- MobileNet SSD with weights to FP32
- MobileNet SSD with weights to FP16

I have no tested the methods in is original framework, but these are some insights to mention:
- The small, less accuracy and best in inference time model is MobileNet SSD. Their models (.bin, file) before and after transformation are quite similar in FP32 mode but no in FP16:
  - 23147564 bytes vs 23133680 bytes. for FP32
  - 23147564 bytes vs 11566846 bytes. for FP16
- In contrast, SSD500 y SSD300 have very different sizes before and after conversion:
  - 137233124 bytes vs 68610424 bytes in SSD300
  - 144166692 bytes vs 72076404 bytes in SSD500
- The MobileNet SSD have a latency about 9 ms in FP32 and 7 ms in FP16. They are a very low latency models, it is very difficult find models some good in latency. And if we move to cloud services, in many solutions, just network delay is bigger than that.

- The MobileNet SSD are not very accuracy. I have made some code error control, but I think is enough accuracy for the counter person app. 
- SSD are accuracy models but with latencies about 400ms and 140ms. Probably two slow for video real time applications.


## Assess Model Use Cases

Some of the potential use cases of the people counter app are:

- In Retail. In Great clothing firms (Zara, Nike, etc.) for example for marketing analytics, it coul be count the ratio of (buyed items)/(people visted). In order to optimized centers with less ratio or learnt from centers with high ratio.+

- In bars and discos with limited capacity, to control exactly the numnber of people inside. (counter people come in and come out)

- In shop windows to measure the interest of people in it, by counting how many people stay more than a threshold in front of the window.


## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a
deployed edge model. The potential effects of each of these are as follows.  Deep learning solutions have very poor performance when test data distributions are lightly different to training data distributions. Are quite good models to interpolate data, but bad models to extrapolate data.  Because of this constraint, when we use the model in conditions different where has been trained, they are very sensible to errors. Obviously, these conditions include, focal distance, lighting, weather conditions, camera angles, etc. So, it must be tested in real conditions before to put in productions.  If performance is not right, probably the best solutions be re-train the model (with a supported framework) with real videos and images, and the transform to IR with OpenVINO and deploy in edge hardware.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three
different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [SSD 300]
  - [https://github.com/weiliu89/caffe/tree/ssd]
  - I converted the model to an Intermediate Representation with the following arguments...
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model VGG_coco_SSD_300x300_iter_400000.caffemodel  --input_proto deploy.prototxt --data_type FP16
  - The model was insufficient for the app because it was too slow

  
- Model 2: [SSD 512]
  - [https://github.com/weiliu89/caffe/tree/ssd]
  - I converted the model to an Intermediate Representation with the following arguments...
    python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model VGG_coco_SSD_512x512_iter_360000.caffemodel --input_proto deploy.prototxt --data_type FP16
  - The model was insufficient for the app because it was too slow

- Model 3: [MobileNet SSD FP32]
  - [https://github.com/chuanqi305/MobileNet-SSD]
  - I converted the model to an Intermediate Representation with the following arguments...
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model MobileNetSSD_deploy.caffemodel


- Model 4: [https://github.com/chuanqi305/MobileNet-SSD]
  - [Model Source]
  - I converted the model to an Intermediate Representation with the following arguments...
  python /opt/intel/openvino/deployment_tools/model_optimizer/mo.py --input_model MobileNetSSD_deploy.caffemodel --data_type FP16
