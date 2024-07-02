import math
import numpy as np

def pixels_to_feet(altitude, pixel_size, original_fw):
    aspect_ratio = 1.7777777777
    fov = (altitude * aspect_ratio * 2 * math.tan(math.radians(73) / 2)) / (np.sqrt(1 + aspect_ratio ** 2))
    #TODO automate parameters here 
    size_m = pixel_size / original_fw * fov
    return size_m * 3.28084

class Track:
    def __init__(self, id, box, conf, frame, timestamp):
        self.id = id
        self.box = box
        self.top_confs = [conf]
        self.confirmed = False
        self.max_conf = conf
        self.measured_on = 'not measured'

        #call shark sizing here
        self.size = 0
        self.timestamp = timestamp

        #TODO: implement timestamp attribute 
        #TODO draw boxes on save frames, especially for low conf tracklets

        self.max_conf_frame = frame
        self.sizing_frame = frame
        self.sizing_box = box

        
    def update_track(self, box, conf, frame, altitude, high_conf_det_limit, high_conf_threshold, original_fw):
        if conf > self.max_conf:
            self.max_conf = conf
            self.max_conf_frame = frame
            self.box = box
        
        if not self.confirmed:
            self.top_confs.append(conf)
            self.top_confs.sort(reverse=True)
            self.top_confs = self.top_confs[:high_conf_det_limit]
            if all(conf >= high_conf_threshold for conf in self.top_confs) and len(self.top_confs) >= high_conf_det_limit:
                self.confirmed = True
        
        if box[2] > box[3]:
            long_side = box[2]
            short_side = box[3]
        else:
            long_side = box[3]
            short_side = box[2]
        #set aspect ratio limit for sizing here 
        if short_side / long_side >= 0.57:
            size = pixels_to_feet(altitude, math.sqrt(short_side**2 + long_side**2), original_fw)
            if  size > self.size:
                self.size = size
                self.measured_on = 'diag'
                self.sizing_frame = frame
                self.sizing_box = box
        else:
            size = pixels_to_feet(altitude, long_side, original_fw)
            if size > self.size:
                self.size = size
                self.measured_on = 'non_diag'
                self.sizing_frame = frame
                self.sizing_box = box
        # print(size)
        
        #get shark size, if shark size> than ; maybe only take length from measurements over high conf thresh

class SharkTracker:
    def __init__(self, altitude, desired_frame_rate, high_conf_threshold=.69):
        self.tracks = []
        self.altitude = altitude
        limit = round(1/2 * desired_frame_rate)
        if limit < 2:
            self.high_conf_det_limit = 2
        else:
            self.high_conf_det_limit = limit
        self.high_conf_threshold = high_conf_threshold


    def update_tracker(self, detections_list, frame, original_fw, timestamp):
        existing_track_ids = [x.id for x in self.tracks]
        for det in detections_list:
            if det[0] in existing_track_ids:
                [x for x in self.tracks if x.id == det[0]][0].update_track(det[1],
                                                                           det[2],
                                                                           frame,
                                                                           self.altitude,
                                                                           self.high_conf_det_limit,
                                                                           self.high_conf_threshold,
                                                                           original_fw)
            else:
                new_track = Track(det[0], det[1], det[2], frame, timestamp)
                self.tracks.append(new_track)
               
        return self.tracks



    
                

            






