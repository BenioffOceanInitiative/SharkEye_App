import os
import pandas as pd
import cv2
from tracker_logic import Track, SharkTracker

def convert_bbox_center_to_corners(x_center, y_center, width, height):
    # Calculate half-width and half-height
    half_width = width / 2
    half_height = height / 2

    # Calculate top-left and bottom-right coordinates
    top_left_x = int(x_center - half_width)
    top_left_y = int(y_center - half_height)
    bottom_right_x = int(x_center + half_width)
    bottom_right_y = int(y_center + half_height)

    return top_left_x, top_left_y, bottom_right_x, bottom_right_y


def draw_max_conf_bounding_box(image, bbox, object_id, max_conf):
    x, y, width, height = bbox

    # Convert float coordinates to integers
    x, y, width, height = int(x), int(y), int(width), int(height)

    xtl, ytl, xbr, ybr = convert_bbox_center_to_corners(x, y, width, height)

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, thickness)

    # Display object information
    text = f"Object ID: {object_id}, Confidence: {max_conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (xbr - 30, ybr + 30)
    cv2.putText(image, text, text_position, font, font_scale, color, font_thickness)
    return image 


def draw_bounding_box(image, bbox, object_id, size, measured_on):
    x, y, width, height = bbox

    # Convert float coordinates to integers
    x, y, width, height = int(x), int(y), int(width), int(height)

    xtl, ytl, xbr, ybr = convert_bbox_center_to_corners(x, y, width, height)

    # Draw bounding box on the image
    color = (0, 255, 0)  # Green color for the bounding box
    thickness = 2
    cv2.rectangle(image, (xtl, ytl), (xbr, ybr), color, thickness)

    # Display object information
    text = f"Object ID: {object_id}, Size (ft): {size:.2f}, Measured on: {measured_on}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_position = (xbr - 30, ybr + 30)
    cv2.putText(image, text, text_position, font, font_scale, color, font_thickness)
    return image 


def save_image_with_bbox(image, filename, path):
    filepath = path + '/' + filename + '.jpeg'
    cv2.imwrite(filepath, image)




def output(final_shark_list, low_conf_objects):

    if not os.path.exists(os.path.join(os.getcwd(),'results')):
        os.makedirs('results')
    
    #TODO make this more robust

    if len([f for f in os.listdir('results') if not f.startswith('.')]) == 0:
        survey_number = 0
    else: 
        max = 0
        for fname in [f for f in os.listdir('results') if not f.startswith('.')]:
            survey_num = int(fname[6:])
            if survey_num > max:
                max = survey_num
            survey_number = max + 1
    

    survey_filename = 'survey' + str(survey_number)
    survey_path = os.path.join('results', survey_filename)
    
    high_conf_path = os.path.join(survey_path, 'high_confidence_sharks')
    high_conf_images_path = os.path.join(high_conf_path, 'frames_used_for_sizing')
    os.makedirs(high_conf_images_path)

    low_conf_path = os.path.join(survey_path, 'low_confidence_objects')
    low_conf_images_path = os.path.join(low_conf_path, 'max_confidence_frames')
    os.makedirs(low_conf_images_path)

    shark_df = pd.DataFrame(columns=['Object_ID', 'Size_(ft)', 'Measured_On', 'Timestamp', 'Max_Confidence'])

    for s in final_shark_list:
        row_df = pd.DataFrame([{'Object_ID': s.id, 'Size_(ft)': s.size, 'Measured_On': s.measured_on, 'Timestamp': s.timestamp, 'Max_Confidence': s.max_conf}])
        #TODO: fix, in fute will not be able to concatenate to empty df
        shark_df = pd.concat([shark_df, row_df], ignore_index=True)

        frame_filename = 'track_' + str(s.id)
        save_image_with_bbox(draw_bounding_box(s.sizing_frame, s.sizing_box, s.id, s.size, s.measured_on), frame_filename, high_conf_images_path)

    low_conf_obj_df = pd.DataFrame(columns=['Object_ID', 'Size_(ft)', 'Measured_On', 'Timestamp', 'Max_Confidence'])
    for o in low_conf_objects:
        row_df = pd.DataFrame([{'Object_ID': o.id, 'Size_(ft)': o.size, 'Measured_On': o.measured_on, 'Timestamp': o.timestamp, 'Max_Confidence': o.max_conf}])
        #TODO: fix, in fute will not be able to concatenate to empty df
        low_conf_obj_df = pd.concat([low_conf_obj_df, row_df], ignore_index=True)

        frame_filename = 'track_' + str(o.id)
        save_image_with_bbox(draw_max_conf_bounding_box(o.max_conf_frame, o.box, o.id, o.max_conf), frame_filename, low_conf_images_path)

    
    shark_df.to_csv(os.path.join(high_conf_path, 'high_confidence_detections.csv'))
    low_conf_obj_df.to_csv(os.path.join(low_conf_path, 'low_confidence_detections.csv'))


