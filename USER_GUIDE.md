# SharkEye App — User Guide

This guide explains how to use SharkEye to process drone videos and review detections. You don’t need any machine learning experience: the app runs the analysis for you. You only need to add videos, choose a few options, run processing, and then review or correct the results.

---

## Table of Contents

1. [Overview](#overview)
2. [Main Screen (Home)](#main-screen-home)
3. [Processing Your Videos](#processing-your-videos)
4. [Review Screen](#review-screen)
5. [Previous Experiments (History)](#previous-experiments-history)
6. [Settings](#settings)
7. [Where Results Are Saved](#where-results-are-saved)

---

## Overview

SharkEye analyzes drone footage to:

- **Detect** objects that might be sharks (or other things like kelp, boats, etc.).
- **Track** each object across multiple video frames so you get one “detection” per animal or object.
- **Estimate length** of sharks (in feet) using your drone’s altitude and camera settings.
- **Segment** the shark in the best frame (an outline/mask) for clearer visualization.

You provide:

- One or more video files.
- The drone model and altitude.
- A flight location name.

The app then runs the analysis and opens a **Review** screen where you can watch short clips of each detection, change labels (e.g., Shark vs. Kelp), and save or export the data.

---

## Main Screen (Home)

When you open SharkEye, the first screen is the **Home** page.

### Top banner

- **Left (clock icon)** — **Previous Experiments**. Opens the list of past processing runs so you can review or edit them.
- **Center** — SharkEye logo.
- **Right (gear icon)** — **Settings**. Opens the settings window (drones, confidence, cloud, etc.).

### Video list area

- **Select Video(s)** — Opens a file picker so you can choose one or more video files. Supported formats depend on your system (e.g. MP4, MOV). After selection, file names appear in the table below.
- **Remove All Videos** — Clears the entire video list. Enabled only when there is at least one video in the list.
- **Video list (table)** — Shows the names of the videos you added. Order matches the order in which they will be processed. You can reorder by dragging rows if the app supports it.

### Drone and flight options

- **Select Drone Model:** — Dropdown of drone models you have configured in Settings (e.g. Mavic 2 Pro, Air 2S). The app uses this to match your video resolution and camera field of view for length and tracking.
- **Enter Drone Altitude:** — Altitude in meters (e.g. `40`). Used to convert pixel size to real-world length (feet). Default is 40.
- **Enter Flight Location:** — A name for this flight (e.g. beach name, date). Stored with the results and used when saving. Your last entry is remembered for the next run.

### Process Videos

- **Process Videos** — Starts analysis of all videos in the list. It is only enabled when at least one video is in the list.  
  Before processing, the app checks that you have entered **Flight Location** and **Drone Altitude**; if either is missing, a warning appears and processing does not start.

---

## Processing Your Videos

When you click **Process Videos**:

1. **Processing Preview** window opens:
   - **Live frame** — Shows the current video frame being analyzed.
   - **Progress bar** — Overall progress (e.g. video 1 of 3, then frame progress within each video).
   - **Status text** — Short message such as “Running Inference” or “Uploading Frames” so you know which step is running.
   - **Timer** — Elapsed time (HH:MM:SS).
   - **Cancel Processing** — Stops the run. Already-processed videos keep their results; the current video may be incomplete.

2. The app runs, for each video:
   - **Inference** — A built-in model scans each frame for objects (sharks and similar).
   - **Tracking** — Detections are linked across frames into “tracks” (one per object).
   - **Filtering** — Only tracks that meet the **confidence** and **minimum frames** set in Settings are kept.
   - **Length & segmentation** — For each track, length (feet) is estimated and a mask/outline is generated where supported.

3. When all videos are done:
   - A **Processing Complete** popup shows total number of detections and total time.
   - The window closes and the app switches to the **Review** screen with the experiment you just ran (or the latest one) so you can review and edit.

**Tips:**

- Use videos that match a resolution configured for your selected drone (e.g. 2688×1512). If the resolution doesn’t match, you’ll get a warning before processing.
- Processing time depends on video length, resolution, and your computer. The preview and progress bar help you monitor it.

---

## Review Screen

The Review screen appears after processing (or when you open a past experiment). It is used to:

- Watch a short clip (GIF/MP4) of each detection.
- See detection details (video, time, confidence, length, label).
- Change the **label** (e.g. Shark, Kelp, Boat) and **Save** to update the stored results.
- Optionally switch between “bounding box” view and “mask” (segmentation) view.

### Top banner (on Review)

- **Left (house icon)** — **Go to Home**. Returns to the main screen. If you have unsaved label changes, you’ll be asked whether to save, discard, or cancel.
- **Center** — Logo.  
  (Settings is not available on this view.)

### Experiment selector (dropdown)

- When viewing **Previous Experiments**, a **dropdown** at the top lists past runs (newest first). Each entry shows a readable date and a summary like “(1 video, 3 sharks)”. Choose an experiment to load its detections into the table below.

### Video / clip player

- Large area that shows either:
  - A **looping clip** (GIF or MP4) of the selected detection, or  
  - A **static image** (e.g. mask overlay) when you use the display toggle.
- **Display toggle** (near the bottom of the player):
  - One side (e.g. box icon) — Bounding box / normal clip view.
  - Other side (e.g. shark fin icon) — Mask/segmentation overlay for that detection.  
  Toggle between them to compare how the shark was outlined.

### Low-confidence warning

- If the selected detection has **confidence below the app’s low-confidence threshold** (e.g. 0.65), a message like “Low confidence in this detection. Please review before saving!” appears. Use the **Label** dropdown to correct mislabels (e.g. change “Shark” to “Kelp”) and then **Save Changes**.

### Detection / history table

- A table lists each detection (each “track”) for the current or selected experiment. Columns typically include:
  - **Experiment** — Experiment date/name (may be hidden in some views).
  - **Video** — Source video file name.
  - **ID** — Track ID (may be hidden in some views).
  - **Timestamp** — Time in the video (e.g. MM:SS).
  - **Confidence** — Detection confidence (0–1). Low values may be shown in red.
  - **Length** — Estimated length in feet.
  - **Label** — Current label (e.g. Shark, Kelp, Dolphin, Surfer, Boat, Bird, Duplicate, None, Other). You can change this in **Edit** mode.
  - **Actions** — e.g. a delete (trash) button when editing past experiments.

- **Clicking a row** updates the player to show that detection’s clip (or mask) and updates the low-confidence warning if applicable.

### Buttons

- **Edit Tracks** — Enables editing: you can change the **Label** dropdown for each row and use the delete button (when viewing a past experiment). Click again to disable editing.
- **Save Changes** — Saves any label changes (and deletions) to the experiment’s CSV and related data. If **Cloud upload** is enabled in Settings, the app may then upload the updated experiment. You’ll see a confirmation when saving is done.

---

## Previous Experiments (History)

You can work with past runs in two ways:

1. **From the banner** — On the Home screen, click the **clock (history)** icon. The app switches to the Review screen and shows the **Previous Experiments** dropdown and the **historical** detection table (past runs only).
2. **From the dropdown** — On the Review screen, use the experiment dropdown to pick a date/run. The table below shows all detections for that run.

In both cases:

- Select an experiment from the dropdown to load its detections.
- Click **Edit Tracks**, then change **Label** or delete a row (trash icon) as needed.
- Click **Save Changes** to write label changes (and deletions) back to the CSV and, if enabled, trigger cloud upload.

So “Previous Experiments” is the same Review screen, but focused on older runs instead of the one you just processed.

---

## Settings

Click the **gear** icon on the Home screen to open **Settings**. The left side has a list of categories; the right side shows the selected page.

### Drone Settings

- **Tree list** — Shows each drone (e.g. Mavic 2 Pro, Air 2S) and under each, resolutions (e.g. 2688×1512) and their FOV (field of view) in radians. The app uses this to validate video resolution and compute length.
- **Edit** — Select a resolution or FOV row, then click Edit to change width, height, or FOV (radians). Only enabled when a resolution or FOV row is selected.
- **Add New Drone** — Add a new drone name and one resolution (width, height, FOV in radians).
- **Delete Drone** — Remove the selected drone from the list. Only enabled when a top-level drone name is selected.

These settings define which resolutions are valid for **Select Drone Model** on the Home screen.

### Past Experiments

- **Past Experiments table** — Lists all saved experiment runs (newest first). Each row has a checkbox and a short description (e.g. date and “(1 video, 3 sharks)”).
- **Export only sharks to CSV** — If checked, exports include only rows labeled “Shark”; otherwise all labels are included.
- **Select All** — Toggles all checkboxes (select all or clear all).
- **Export Selected Results** — Exports the **selected** experiments’ detection data into one combined CSV. You choose the save location. If “Export only sharks to CSV” is checked, only Shark rows are included.
- **Delete Selected Results** — Permanently deletes the **selected** experiment folders (all their CSVs, images, masks, etc.). A confirmation dialog appears; this cannot be undone.

Use this page to export data for analysis elsewhere or to remove old experiments you no longer need.

### Confidence Threshold

- **Enter Confidence Threshold** — A number between 0 and 1 (e.g. 0.40). Detections below this are not kept as tracks. Higher = stricter (fewer detections, usually more reliable). Default is 0.40.
- **Minimum Frames** — Minimum number of frames a track must appear in to be kept (e.g. 5). Short flickers are discarded.
- **Reset to Default** — Sets confidence to 0.40 and minimum frames to 5.
- **Save** — Saves the current values. They are used the next time you run **Process Videos**.

No machine learning knowledge needed: think of “confidence” as “how sure the model is.” You can try 0.3–0.5 and use **Minimum Frames** to avoid noise.

### Cloud Features

- **Past Experiments table** — Same list of experiments as in **Past Experiments** (with checkboxes).
- **Enable automatic Cloud upload when saving** — If checked, when you click **Save Changes** on the Review screen, the app uploads the updated experiment to the configured cloud storage.
- **Select All** — Select or clear all experiments in the table.
- **Upload Selected Results to Cloud** — Manually upload the **selected** experiment(s) to the cloud. Use this for one-off uploads without enabling auto-upload.

Use this to back up or share results with a team.

### Accessibility

- **Annotation Color (RGB)** — Color used for bounding boxes and text on the video. Click the button to open a color picker.
- **Box Thickness** — Line thickness for bounding boxes (e.g. 1–20).
- **Text Thickness** — Line thickness for text labels.
- **Text Scale** — Size of text (e.g. 0.1–10.0).
- **Reset to Default** — Restores default color and sizes.
- **Save** — Saves these values for future processing and review.

Use this to make boxes and text easier to see (e.g. high contrast, thicker lines).

---

## Where Results Are Saved

- Results are stored in a **results** folder:
  - **Installed app:** next to the SharkEye executable.
  - **Development:** typically in the project root, in a `results` folder.
- Each run gets a **timestamped folder** (e.g. `MMDDYYYY_HHMMSS`) containing:
  - **detection_results** — CSV files (one per video) with track ID, label, confidence, length, timestamps, etc.
  - **bounding_boxes**, **frames**, **masks**, **tracking_gifs** — Images and clips used on the Review screen and for export/upload.

When you **Save Changes** on the Review screen, the app updates the CSV (and optionally uploads to the cloud). When you **Export Selected Results** or **Export only sharks to CSV** in Settings, you choose where to save the combined CSV on your computer.

---

## Quick Start Summary

1. **Home** — Click **Select Video(s)** and add one or more videos. Choose **Drone Model**, enter **Drone Altitude** and **Flight Location**. Click **Process Videos**.
2. **Processing** — Watch the preview and progress; use **Cancel Processing** if you need to stop.
3. **Review** — After processing, review each detection in the table and player. Use **Edit Tracks** to change **Label** (e.g. Shark/Kelp). Click **Save Changes** when done.
4. **Previous runs** — Use the **clock** icon and the experiment dropdown to open past runs, edit labels, and save again.
5. **Export / backup** — In **Settings → Past Experiments**, select runs and use **Export Selected Results** (and optionally **Export only sharks to CSV**). In **Cloud Features**, enable auto-upload or use **Upload Selected Results to Cloud** to back up to the cloud.

If anything is unclear, use the in-app labels and tooltips (e.g. “Previous Experiments”, “Settings”) as a reminder of each button’s role.
