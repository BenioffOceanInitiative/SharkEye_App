import cv2
import os
import argparse
from ultralytics import YOLO
from video_processing import ar_resize, draw_bounding_box, save_image_with_bbox
from tracker_logic import SharkTracker
from output import output

def seconds_to_minutes_and_seconds(seconds):
    minutes, seconds = divmod(seconds, 60)
    return str(minutes) + ':' + str(round(seconds)) 


def run_inference(gpu=False, imgsz=720, videos=[], video_dir='test_vids', altitude=40):
    #load the model
    model = YOLO('model_weights/exp1v8sbest.pt')

    #select optimal frame rate for device
    if gpu:
        desired_frame_rate = 8
    elif not gpu:
        desired_frame_rate = 4

    print(desired_frame_rate)
    
    # make function to concatenate videos if needed here

    # will return list of videos for inference (full file path to videos)
    if (len(videos) == 0):
        videos = [os.path.join(video_dir, file) for file in os.listdir(video_dir) if not file.startswith('.')]

    final_shark_list = []
    final_low_conf_tracks_list = []
    shark_count = 0

    for video in videos:
        cap = cv2.VideoCapture(video)
        original_frame_width = cap.get(3)

        # get h, w for model.track to resize image with
        # TODO might need to resize and reduce frame rate before inference to conserve memory
        new_imgsz = ar_resize(cap.get(3), cap.get(4), imgsz)
        
        # find the rate to sample the video to ge tthe desired frame rate
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_sample_rate = round(fps/desired_frame_rate)

        #initiate tracker
        st = SharkTracker(altitude, desired_frame_rate)
        print(st.high_conf_det_limit)

        frame_no = 0
        while cap.isOpened():
            success = cap.grab()

            #reducing video frame rate here
            if success and frame_no % frame_sample_rate == 0:
                _, frame = cap.retrieve()
                if gpu:
                    #TODO test if yolo resizing works and is efficient
                    results = model.track(frame, conf=.45, device='mps', imgsz=new_imgsz, iou=0.4, show=False, verbose=False, persist=True)
                elif not gpu:
                    results = model.track(frame, conf=.45, imgsz=new_imgsz, iou=0.4, show=False, verbose=False, persist=True)
                # Get the boxes ,classes and track IDs
                boxes = results[0].boxes.xywh.cpu().tolist()
                confidence = results[0].boxes.conf.cpu().tolist() 
                # print(boxes)
                track_ids = results[0].boxes.id
                if track_ids == None:
                    track_ids = []
                else:
                    track_ids= track_ids.cpu().tolist()
                    
                timestamp = seconds_to_minutes_and_seconds(cap.get(cv2.CAP_PROP_POS_MSEC)/fps)

                detections_list = zip(track_ids, boxes, confidence)
                all_tracks = st.update_tracker(detections_list, frame, original_frame_width, timestamp)

                # classes = results[0].boxes.cls.cpu().tolist()
                # names = results[0].names
                #print(track_ids)
                # print(classes)
                # print(names)
                #print(boxes)
                #print(track_ids)
                #print(confidence)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            elif not success:
                break

            frame_no += 1
        
        #TODO are they sorted by object id 


        for trk in all_tracks:
            if trk.confirmed:
                final_shark_list.append(trk)
                shark_count += 1
            else:
                final_low_conf_tracks_list.append(trk)

    # create and delete directories as needed

    # save ann info
    output(final_shark_list, final_low_conf_tracks_list)
    return final_shark_list
    #return final shark list

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action =argparse.BooleanOptionalAction, help='True or False this is a Macbook with an m1, m2, or m3 chip')
    parser.add_argument('--imgsz', type=int, default=720, help='image height for inference (pixels)')
    parser.add_argument('--video_dir', type=str, default='survey_video', help='folder where videos to process exist')
    parser.add_argument('--altitude', type=int, default=40, help='survey flight altitude (meters)')
    opt = parser.parse_args()
    return opt

def main(opt):
    run_inference(**vars(opt))

if __name__=='__main__':
    opt = parse_opt()
    main(opt)



