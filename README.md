# README

The code has two files:
  - main.py.  The main class that execute the code, reading a model, a file (video or image) or web cam stream, make all the inference steps and send statistics to mqts and media to ffpserver.
  - inference.py. Implement different steps of inference process.

### MPORTANT NOTE. FOR RESUMITTION

The app counter works properly with threshold of 0.1 . It could seen a little low threshold, but in videos there is some people with black clothes and it seem these resolutions with that colour generate a lot of uncertainties about if the object is a person. These effect happens in the tree models tested. 

The drawbacks of these low threshold could be the increase of false positives. For examples, in videos where appear animals, it could be counted as peope.

The code has been tested with four different models. Here is the command to execute it.

### SSD500

/Users/mmartin/anaconda3/bin/python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /Users/mmartin/workspace/udacity/AI_IoT/nd131-openvino-fundamentals-project-starter-master/models/VGGNet/coco/SSD_512x512/VGG_coco_SSD_512x512_iter_360000.xml  -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.1 |  ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

### SSD300

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /Users/mmartin/workspace/udacity/AI_IoT/nd131-openvino-fundamentals-project-starter-master/models/VGGNet/coco/SSD_300x300/VGG_coco_SSD_300x300_iter_400000.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.1 |  ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


### MobileNetSSD FP32

python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /Users/mmartin/workspace/udacity/AI_IoT/nd131-openvino-fundamentals-project-starter-master/models/VGGNet/MobileNetSSD/MobileNetSSD_deploy.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.1 |  ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


### MobileNetSSD FP16

/Users/mmartin/anaconda3/bin/python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m /Users/mmartin/workspace/udacity/AI_IoT/nd131-openvino-fundamentals-project-starter-master/models/VGGNet/MobileNetSSD_16/MobileNetSSD_deploy.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension.dylib -d CPU -pt 0.1 ù  ffmpeg/ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm


# CODE CONSIDERATIONS:

- I have added some methods to inference class, to manage preprocessing and person class id, because MobileNetSSD model has differents handling that SSD 500 and SSD 300.
- MobileNet SSD have more errors that the other two models. To manage this error, I have considered that a recogniztion is ok if it is repeated in N consecutives frames. Then buy tunning I have configured probability threshold to 0.1 and N (number of consecutive frames) to 10.
- When there are several people in the video, to simplify, I consider that first people in first people out. Obviously this behaviour has not be true. but it does not affect in average statistics considerations.



# NOTES: THINGS I CONSIDER P1 and P2 Issues in this proyect.

I mention here some issues and considerations I think, in my understanding, it has been modified to get a better solution:


- P1:
  - It is specified in some parts that to open the web application that show the video the url is http://0.0.0.0:3004. This is a bug. This is the port of the ffserver. The right url to open the web app is http://://0.0.0.0:3000. Node.js web server is listening in port 3000.
  - Workspace of the project is not loading in Safari browser. Anyway, I think is not critical because you can use chrome.
  - I think the requirement of the project about average time is frame it is at little be confusing. It talks about average, but actually you have to calculate the duration of a person in the video, because   the average is calculated in the web app. 
  - Doc about OpenCV is not so good in my opinion. You have to waist time in trying by debuggin and loggin code to know exactly how some api works.
  - In the web app, the average of the time in frame use the javascript library moment to get the time in seconds from mqtt queue. But when the duration is higher than 59 sec there is an error. In my opinion ith should be use also moment-duration library or similar.
  - When you are using the Udacity workspace to implement the project instead doing locally, it could be some problems to download models that are in google drive. I have created this command: 

    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=FILEID' -O- | awk ‘ / confirm / { split($0, a, “confirm=“), split(a[2], b, “&” ), print b[1]}’ )&id=FILEID" -O FILENAME && rm -rf /tmp/cookies.txt

    FILEID and FILENAME are the fileid and filename of google drive file. There are other version with sed instead of awk. I feel more confortable with awk.

- P2: 

  - In my opinion to make the software with better scalability in functionality, and tests different statistics, Average per frame and total count of people should be calculated in the main.py code, and put in a queue (Actually total count is doing that, but no used in web app). And web app, just show the information from the right topic in the queue.
  - Maybe is a good idea advise the students that they have to spent time researching and testing how is the format of the output of the model. Most of them are not very well documented and it is needed to make try and test. Also prepared the code for different input and output processing depending on the model (p.e. class id for detect a person, box coordinate format, frame format, etc..)
