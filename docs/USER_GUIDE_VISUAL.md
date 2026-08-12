Testing Post from Github actions
# Running Analysis![][image1]

## How to Process Footage

1. Press **Select Video(s)** to import videos for analysis. The name of each selected video will be displayed in the table in the center of the page. You can use the **Remove All Videos** button, or press the X next to each row to remove videos from the queue. The model currently works best by processing 1-2 videos at a time  
2. Choose the drone model used to capture the selected videos using the **Select Drone Model** dropdown menu. ***The drone model must be the same for all videos entered in order for the length predictions to be accurate\!***    
3. Enter the **Drone Altitude** in meters and **Flight Location**  
4. Press **Process Videos** to begin the AI detection.

![][image2]

## Processing Preview

Once video processing begins, a preview window will appear with information about the process. The main window begins displaying frame-by-frame object detection for each video, displaying the classification of the object (e.g. “shark”), the confidence of the detection, and a bounding box around the detected object. Beneath is a progress bar, text display describing the processing status, and current time elapsed. You can also press the **Cancel Processing** button to halt all processing and return to the home screen.

![][image3]
Once the processing is complete an additional pop-up will appear displaying the total detections and time taken to process. Press **OK** to move on to the review screen.

![][image4]  
## Where Results Are Saved

After processing, the resulting files are stored locally in a *results* folder

- ***If you installed the app from SharkEye website***, the folder is located in the same location as SharkEye executable.  
- ***If you cloned the SharkEye repository using Git,*** the folder is located at “src/results”

Each run gets a timestamped folder (e.g. MMDDYYYY\_HHMMSS) containing:

* **detection\_results** — CSV files (one per video) containing the following data from the model:  
  * **video\_name** — The local path of the video file where the detection occurred.  
  * **Flight Location** — The physical location, site, or area associated with the flight or video capture.  
  * **Track Id** — A unique identifier assigned to a tracked object across frames in the video.  
  * **Highest Conf Timestamp** — The timestamp at which the object was detected with its highest confidence score.  
  * **Highest Confidence** — The maximum confidence score recorded for this tracked object across all detections.  
  * **Average Confidence** — The mean confidence score calculated across all detections for this tracked object.  
  * **Lowest Confidence** — The minimum confidence score recorded for this tracked object across all detections.  
  * **Longest Length** — The longest recorded length of the detected object.  
  * **Highest Confidence Length** — The length associated with the detection that had the highest confidence.  
  * **Number of Detections** — The total count of individual detections associated with this tracked object.  
  * **Meets Thresholds** — An indicator showing whether the track satisfies the confidence threshold and minimum frames (True/False)  
  * **Confidence of Longest Length** — The confidence score associated with the longest detection segment.  
  * **Label** — The classification or category assigned to the detected object.  
  * **manual\_length\_px** — Optional manually measured length in pixels from the Frame Editor. Empty until you confirm a drawn line.  
  * **manual\_length\_ft** — Optional manually measured length in feet from the Frame Editor. Empty until you confirm a drawn line. When set, this value is shown in the review table’s **Length** column instead of the automatic estimate.  
* **bounding\_boxes** — Image of detection with bounding boxes, track ID, confidence, and length in feet.  
* **frames** — Image of the detection  
* **masks** — Image of the detection overlaid with a mask.  
* **tracking\_gifs** — GIFs of the detection tracks with bounding boxes, same as the clips used during review stage.  
* **experiment\_note.txt** — Optional text note for the experiment (see **Experiment Notes** below).

***Note:** The images saved in **bounding\_boxes, frames, and masks** are the frame for which the object has the longest length*  
When you Save Changes on the Review screen, the app updates the CSV (and optionally uploads to the cloud). When you Export Selected Results using the Settings menu, you choose where to save the combined CSV on your computer.

## Reviewing Detections

The review screen lets you view all detected results from the processed videos. You can press the icon in the top left to **Go to Home** and return to the main screen. If you have unsaved label changes, you’ll be asked whether to save, discard, or cancel.
![][image6]
### Video player

The center of the window displays the currently selected detection. It will display either a looping clip of the entire video with a bounding box similar to the preview screen, or a static image with a mask overlay. You can toggle between the two modes by pressing the switch in the bottom right of the frame, which is placed between two shark fin icons.
![][image7a]
If the selected detection has confidence below 0.65, a warning will appear like “Low confidence in this detection. Please review before saving\!” appears. 

### Detection Table

The table lists each detection (or “track”) for the current or selected experiment. Clicking a row in the table updates the player to show that detection’s clip.The columns include:

* **Video** — Source video file name.  
* **Timestamp** — Time in the video (e.g. MM:SS).  
* **Confidence** — Detection confidence (0–1). Low values may be shown in red.  
* **Length** — Estimated length in feet. If you have confirmed a manual measurement with the Frame Editor, this column shows that manual length instead of the automatic estimate.  
* **Label** — Current label (e.g. Shark, Kelp, Dolphin, Surfer, Boat, Bird, Duplicate, None, Other). Use this dropdown to correct incorrectly labeled entries.  
* **Delete Button** — Button to remove delete an entry.

To enable making changes to a track or its label, use the buttons:

* **Edit Tracks** — Enables editing: you can change the Label dropdown for each row and use the delete button (when viewing a past experiment).   
* **Save Changes** — Saves any label changes (and deletions) to the experiment’s CSV and related data.

If Cloud upload is enabled in Settings, the app may then share the updated experiment data with the development team. A confirmation window will appear when the upload is complete. See the ***Cloud Features*** section for additional details. 

### Manual Length Correction (Frame Editor)

When the mask view is active, you can open the **Frame Editor** to draw your own measurement line over the shark and save a corrected length.

![][image16]

1. Select a detection in the table.  
2. Toggle the player to **mask mode** using the switch in the bottom right of the frame (between the two shark fin icons).  
3. Press the **Edit Frame** button (draw-line icon) in the bottom left of the frame. The player is replaced by the in-place Frame Editor.  
4. At the top of the editor, confirm the **Drone** model and **Altitude (m)**. These must match the flight so the length in feet is accurate.  
5. Press **Draw Line (R)** (or press **R**) to enter drawing mode.  
6. Click and drag on the image to draw a straight segment along the shark. A live length readout appears near the cursor while you draw.
![][image17]
7. To add another segment to the same path, hold **Shift** and click-drag from the end of the existing line.  
8. Use the mouse wheel to **zoom**, and (when not in drawing mode) click-drag to **pan**. Press **R** again or click **Move Frame (R)** to leave drawing mode so you can pan freely.  
9. Press **Clear Line** if you need to erase the line and start over.  
10. Press **Confirm Changes** to save the measured length. A popup confirms that a new length was saved. Press **Cancel Changes** to leave without saving.
![][image18]
**What gets saved**

* The measured length is written to that track’s row in the experiment CSV as **manual\_length\_px** and **manual\_length\_ft**.  
* The automatic length fields are left unchanged.  
* The **Length** column in the review table updates immediately to show the manual length in feet.  
* Re-opening the experiment later will continue to show the manual length whenever those CSV fields are filled in.

### Reviewing Previous Experiments

![][image5]By pressing the clock icon in the top left of the home menu, you can view the results of previously processed videos. The interface is the same as the review page that appears after processing a batch of videos. 

When reviewing experiments this way, a dropdown menu will appear above the picture frame that will display the currently selected experiment, where the date and time are when the batch of videos was originally processed, along with a count of videos and detections (e.g. 2026/2/1 10:00:00 AM (3 videos, 3 detections)).

The experiment dropdown also appears after you finish the initial **Confirm Detections** step for a newly processed batch.

### Experiment Notes

When the experiment dropdown is visible, a **pencil icon** appears to the right of it. Use this to add or edit a short note for the selected experiment.

1. Click the **pencil icon** (tooltip: *Add or edit experiment note*).  
2. In the **Add Experiment Note** dialog, enter your note (e.g. beach name, weather, transect ID, or anything that helps you identify the run later).  
3. Press **Save** to keep the note, **Clear** to remove it, or **Cancel** to close without saving changes.

**How notes are stored and displayed**

* Each note is saved in that experiment’s results folder as **experiment\_note.txt**.  
* When you start processing, the app creates a default note using the video file names (comma-separated). You can replace this with your own text at any time.  
* If a note exists, it is appended to the experiment name in the dropdown, after the date and video/detection counts. For example:  
  `2026/2/1 10:00:00 AM (3 videos, 3 detections) — Santa Monica morning transect`

Notes are useful when browsing **Previous Experiments** or switching between runs in the review screen, so you can quickly tell which batch of footage you are looking at without opening the folder.


# Settings

## Drone Settings 
In order for the model to make accurate sizing estimates, it must know the camera’s field of view and resolution. This page lists those specifications for each drone model compatible with the app. By default, the app includes the required specifications for the DJI Mavic and Air 2S models, but you can also add in additional drones if necessary. Each drone can also support settings for multiple resolutions. 
![][image8] 

### Adding in a New Drone 
Pressing the **Add New Drone** will create a dialogue box to enter the drone’s name, camera resolution, and field of view in radians. If a drone already exists with the entered name, entering a unique width and height will create an additional settings option for that drone. When selecting a drone on the home screen, the model will automatically use the correct resolution and FOV based on the resolutions of the videos provided.
![][image9] 

### Edit Existing Drone
To edit an existing drone, select the resolution or FOV under a drone name and press the **Edit** button. You can both edit the resolution and FOV, and delete the settings for a given resolution in the resulting pop-up menu. Resolution and FOV for particular drone models can be found online, though manual calibrations with known-size objects are the most reliable and accurate way to measure your specific drone. 
![][image10]  

## Past Experiments
This table lists out a summary of all previously run experiments. Each row includes the date that the videos were originally processed, the number of videos processed, and the total number of detections found. 

* **Delete Selected Results** — Permanently deletes the **selected** experiment folders, deleting all their respective CSVs, images, masks, etc.   
* **Export Selected Results** — Exports the **selected** experiments’ detection data into one combined CSV at a specified location. You can also check the **Export only sharks to CSV** box to only include detections labeled as a ‘Shark’.

![][image11]

## Confidence Threshold![][image12]

### Quick Overview

For a quick explanation, the higher the **confidence threshold,** the fewer objects will appear on the results page, but the accuracy of those resulting objects will be more reliable. Lowering the **minimum frames** will increase the number of objects appearing on the results page, but can lead to an increase in poor quality results only a few frames long. 

### Confidence Threshold and Minimum Frames

For the purposes of this brief explanation**,** a **detection** is the result of the model finding an object on a singular frame, or image, of a given video. A **Track** is the collection of detections representing a single shark or object. For each frame, the model provides the level of confidence that the object captured in the bounding box is accurately labeled.

The machine learning model uses two values in determining what is considered a valid track:

* The **Confidence Threshold** is a value between 0 and 1 (e.g. 0.40) that serves as a cutoff for determining valid tracks. Each detection is given a confidence score, representing how certain the model is that the detection is correct. If the average confidence for all detections within a track are below this level, the track is not kept for the review stage. A higher threshold is stricter, meaning the model picks up fewer detections, which are usually more reliable. The default is 0.40.  
* **Minimum Frames** is the minimum number of frames an object must appear in for the track is to be kept for review. This ensures that short flickers are discarded. The default is 5 frames.

## Cloud Features

Users can share their generated data with the SharkEye team at Benioff Ocean Science Laboratory directly via cloud upload. The previously run experiments are displayed in a table similar to the **Past Experiments** page. Select any number of experiments and press the **Upload Selected Results to Cloud** button to share the CSVs, images, and masks of each experiment. By checking **Enable automatic Cloud upload when saving** , the app uploads the updated experiment to the configured cloud storage when you click **Save Changes** on the Review screen. *We advise only using this feature if you intend to share data for additional training with the developers. Automatic Cloud upload may lead to duplicate or low-quality data being shared with the team.*   
![][image13]

## Accessibility

You can change the following attributes to improve the visibility of bounding boxes generated after processing videos.

* **Annotation Color (RGB)** — Color used for bounding boxes and text on the video. Click the button to open a color picker.  
* **Box Thickness** — Line thickness for bounding boxes (e.g. 1–20).  
* **Text Thickness** — Line thickness for text labels.  
* **Text Scale** — Size of text (e.g. 0.1–10.0).  
* **Reset to Default** — Restores default color and sizes.  
* **Save** — Saves these values for future processing and review.![][image14]

# Model Details

## Classification

The model performs inference on a single frame and saves the bounding boxes and confidences of possible detection if those detections have a confidence above the set detection\_threshold. Instead of performing inference on every frame, the model runs prediction on every 10 frames by default, increasing the number of frames skipped if not detections are found.

## Tracking

After processing the frames from each of the selected videos, the app constructs *tracks*, or series of detections that each represent a single object across all frames in time. If a video had two sharks present for example, there would be two tracks for that video, one per shark. In order to figure out which detections belong to which object, the app follows the following algorithm:

1. For the first frame, create a track for each detection.  
2. For the next frame, predict the next position by adding the position of the track’s last detection and its current velocity(calculated by taking the change in position between detections over the number of frames).   
3. Create a cost matrix where rows represent existing tracks, and columns represent current detections. Each cell is assigned a cost, equal to the Euclidean distance between the predicted position of the track and the current position of the detection, plus 10 times the number of frames since the track’s last detection.  
4. Find a one-to-one assignment of tracks and detections that minimizes total cost.  
5. If the cost is below a distance threshold of 250, update that track to include the detection. If the cost exceeds the threshold, treat it as no match for that pair and create a new track for that detection.  
6. If a detection was not assigned to any track, create a new track.  
7. Repeat steps 2-6 for each remaining frame

Once every detection is assigned its own track, the app will only perform length estimation on *significant tracks*. Tracks are considered significant if they both have above a minimum number of frames(5 by default), and the mean confidence of all detections in the track is above a threshold(.8 by default)

## Segmentation

After organizing detections into tracks, the app selects one frame from each track to be passed into a segmentation model to draw an outline of the shark within the given bounding box.

Each bounding frame gets generated a preliminary length estimate based on the height of the bounding box. Of all frames with a confidence above .8, the frame with the longest of these lengths is selected as the longest track. We choose this frame assuming that the frame with the longest length contains the most stretched out version of the shark, which would provide the most accurate length.

The image and bounding box coordinates are then passed into Meta’s [Segment Anything Model(SAM)](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file), which generates a mask of the object within the box. The longest line segment within the outline of the generated mask is chosen as the final length in pixels, and converted into an estimate in feet.