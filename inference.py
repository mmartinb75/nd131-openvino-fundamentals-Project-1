#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
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
import logging as log
import cv2
from openvino.inference_engine import IENetwork, IECore



class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """

    def __init__(self):
        # Initialize any class variables desired 
        self.plugin = None
        self.inference_engine = None
        self.net = None
        self.input_blob = None
        self.output_blob = None

        # store the model name. Class person ID and
        # preprocesing input depend on model name.
        self.model_xml = None
        

    def load_model(self, model_xml, device="CPU", cpu_extension=None):
        # Load the model 
        self.model_xml = model_xml
        self.plugin = IECore()
        model_bin = os.path.splitext(model_xml)[0] + ".bin"
        self.net = IENetwork(model=model_xml, weights=model_bin)
        
        # Add any necessary extensions
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        
        # Check for supported layers 
        supported_layers = self.plugin.query_network(network=self.net, device_name=device)
        
        unsupported_layers = [l for l in self.net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)

        # Return the loaded inference plugin 
        self.inference_engine = self.plugin.load_network(self.net, device)
        
        self.input_blob = next(iter(self.net.inputs))
        self.output_blob = next(iter(self.net.outputs))
        
        
        return

    def get_input_shape(self):
        #  Return the shape of the input layer 
        return self.net.inputs[self.input_blob].shape

    def exec_net(self, image):
        # Start an asynchronous request 
        self.inference_engine.start_async(request_id=0, 
            inputs={self.input_blob: image})
        return

    def wait(self):
        # Wait for the request to be complete. 
        status = self.inference_engine.requests[0].wait(-1)
        return status

    def get_output(self):
        # TODO: Extract and return the output results
        # Note: You may need to update the function parameters. 
        infer_time = self.inference_engine.requests[0].latency
        return (self.inference_engine.requests[0].outputs[self.output_blob], infer_time)
    
    def preproces_input(self, frame):
        # Frame preprocessing has some diference depending on model type.
        net_input_shape = self.get_input_shape()
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))

        # Here are the special handling for MobileNetSSD model
        if 'MobileNetSSD_deploy' in self.model_xml:
            p_frame = p_frame - 127.5
            p_frame = p_frame * 0.007843
        
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        return p_frame

    def get_person_classId(self):
        # Person class has distint id depending on model type

        person_id = 1
        if 'MobileNetSSD_deploy' in self.model_xml:
            person_id = 15
        
        return person_id
