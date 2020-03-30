"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import queue
import time
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

SECUENCIAL_SAFE_FRAMES = 10


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                        "(0.5 by default)")
    return parser


def connect_mqtt():
    # Connect to the MQTT client 
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
  
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise some counters:
    # total count -> to store total count of people
    # current people -> state of total number of people in current frames
    # last_two_counters -> to manage recognition errors. The same number of people
    # has to be recognized in three consecutive frames.
    # Frame counters of current people in order to calculate duration.

    single_image_mode = False

    total_count = 0
    current_people = 0
    last_n_counters = [0 for _ in range(SECUENCIAL_SAFE_FRAMES)]
    frame_counter = 0

    # queues used to calculate duration of each people. 
    # It is asumed that when several are people in video their
    # follow first in first out behaviour. This does not have to be
    # true, but does not affect sum of all people duration.

    init_frames = queue.Queue()
    durations = queue.Queue()
    
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Load the model through `infer_network` 
    infer_network.load_model(args.model, args.device, args.cpu_extension)

    # Handle the input stream ###
    net_input_shape = infer_network.get_input_shape()

     

    # Check if input is a web-CAM, an image, or a video.
    if args.input == 'CAM':
        cap = cv2.VideoCapture(0)
        cap.open(0)
    else:
        resources_file_name = os.path.splitext(args.input)[0]
        resources_file_ext = os.path.splitext(args.input)[1]
        
        if resources_file_ext in ['.png', '.jpg', '.jpeg']:
            # Is a image. rename it in order to read as a sequence
            single_image_mode = True
            new_name = resources_file_name + '01' + resources_file_ext
            inp = os.rename(args.input, new_name)
            cap = cv2.VideoCapture(new_name, cv2.CAP_IMAGES)
            cap.open(new_name, cv2.CAP_IMAGES)
            os.rename(new_name, args.input)
       
        else:
            cap = cv2.VideoCapture(args.input)
            cap.open(args.input)
            
                  

    # inizialize vide capture



    # Check frames per second of video or cam 
    # in order to calculate time of person in frame.
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Get width and heigh of video to calcule box positions.
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Loop until stream is over
    
    while cap.isOpened():
        

        ### Read from the video capture ###
        
        flag, frame = cap.read()
        
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### Pre-process the image as needed ###
        
        #p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        #p_frame = p_frame - 127.5
        #p_frame = p_frame * 0.007843
        #p_frame = p_frame.transpose((2,0,1))
        #p_frame = p_frame.reshape(1, *p_frame.shape)

        p_frame = infer_network.preproces_input(frame)


        # Increase frame_counter in each frame.
        frame_counter += 1

        ### Start asynchronous inference for specified request ###
        infer_network.exec_net(p_frame)

        ### Wait for the result ###
        
        if infer_network.wait() == 0:

            ### Get the results of the inference request ###
            
            result, infer_time = infer_network.get_output()
            infer_text = "inference time: " + str(round(infer_time,3)) + " ms"
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 0.4
            color = (255, 0, 0)
            org = (15, 15)

            frame = cv2.putText(frame, infer_text, org, font, fontScale, color, 1)

            current_count = 0
            safe_counter = 0

            for boxes in result[0][0]:
                if (boxes[1] == infer_network.get_person_classId() and boxes[2] > prob_threshold):
                    
                    x_1 =  int(width* boxes[3])
                    y_1 =  int(height* boxes[4])
                    x_2 =  int(width* boxes[5])
                    y_2 =  int(height* boxes[6])
                    
                    frame = cv2.rectangle(frame, (x_1, y_1), (x_2, y_2), (255,0,0), 2)
                    current_count += 1
                    
            
            # Safe control in order to minimize recoginition error.
            # A counter is considered valid when are the same in three
            # consecutives frames

            if all([l == current_count for l in last_n_counters]):
                safe_counter = current_count
            else:
                safe_counter = current_people

            for i in range(SECUENCIAL_SAFE_FRAMES - 1, 0, -1):
                last_n_counters[i] = last_n_counters[i-1]

            last_n_counters[0] = current_count


            delta_people = safe_counter - current_people
            if delta_people > 0:
                for e in range(delta_people):
                    init_frames.put(frame_counter)

                total_count += delta_people
                current_people = safe_counter

            elif delta_people < 0:
                frames_duration = frame_counter - init_frames.get()
                durations.put(frames_duration/fps)
                current_people = safe_counter
                
            
                
            

            # Extract any desired stats from the results 

            # Calculate and send relevant information on 
            # current_count, total_count and duration to the MQTT server 
            client.publish("person", json.dumps({"count": safe_counter, "total":total_count}))
            # Topic "person": keys of "count" and "total" 
            
            # Topic "person/duration": key of "duration" 
            while not durations.empty():
                client.publish("person/duration", json.dumps({"duration": durations.get()}))
    

        # Send the frame to the FFMPEG server 
        
        if not single_image_mode:
            sys.stdout.buffer.write(frame)  
            sys.stdout.flush()
            
        # Write an output image if `single_image_mode` 

        if single_image_mode:
            resources_file_name = os.path.splitext(args.input)[0]
            resources_file_ext = os.path.splitext(args.input)[1]
            cv2.imwrite(resources_file_name + "_proccesed_" + resources_file_ext, frame)



def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
