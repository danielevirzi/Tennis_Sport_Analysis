import torch
import cv2
from tqdm import tqdm
import numpy as np
import argparse
from itertools import groupby
from scipy.spatial import distance
import librosa 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, sosfilt
from collections import defaultdict
from utils import euclidean_distance
from copy import deepcopy
import pandas as pd
import pickle 
from utils import get_center_of_bbox



device = "cuda" if torch.cuda.is_available() else "cpu"


def postprocess(feature_map, scale=2):
    # Scaling factor is dependent on original video width & height
    scaling_x, scaling_y = scale 
    feature_map *= 255
    feature_map = feature_map.reshape((360, 640))
    feature_map = feature_map.astype(np.uint8)
    ret, heatmap = cv2.threshold(feature_map, 127, 255, cv2.THRESH_BINARY)
    circles = cv2.HoughCircles(heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, param1=50, param2=2, minRadius=2,
                               maxRadius=7)
    x,y = None, None
    if circles is not None:
        if len(circles) == 1:
            x = circles[0][0][0]*scaling_x
            y = circles[0][0][1]*scaling_y
    return x, y


def infer_model(frames, model, scale):
    """ Run pretrained model on a consecutive list of frames    
    :params
        frames: list of consecutive video frames
        model: pretrained model
    :return    
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
    """

    # Image size input for TrackNet is (360,640)
    height = 360
    width = 640
    dists = [-1]*2
    ball_track = [(None,None)]*2
    for num in tqdm(range(2, len(frames))):
        # Take 3 consecutive frames as input
        img = cv2.resize(frames[num], (width, height))
        img_prev = cv2.resize(frames[num-1], (width, height))
        img_preprev = cv2.resize(frames[num-2], (width, height))
        imgs = np.concatenate((img, img_prev, img_preprev), axis=2)
        imgs = imgs.astype(np.float32)/255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)

        out = model(torch.from_numpy(inp).float().to(device))
        output = out.argmax(dim=1).detach().cpu().numpy()
        x_pred, y_pred = postprocess(output, scale = scale)
        ball_track.append((x_pred, y_pred))

        # for not None values
        if ball_track[-1][0] and ball_track[-2][0]:
            dist = distance.euclidean(ball_track[-1], ball_track[-2])
        else:
            dist = -1
        dists.append(dist)  
    return ball_track, dists 


def remove_outliers(ball_track, dists, max_dist = 50):
    """ Remove outliers from model prediction    
    :params
        ball_track: list of detected ball points
        dists: list of euclidean distances between two neighbouring ball points
        max_dist: maximum distance between two neighbouring ball points
    :return
        ball_track: list of ball points
    """
    outliers = list(np.where(np.array(dists) > max_dist)[0])
    for i in outliers:
        if (dists[i+1] > max_dist) | (dists[i+1] == -1):    
            ball_track[i] = (None, None)
            outliers.remove(i)
        elif dists[i-1] == -1:    
            ball_track[i-1] = (None, None)
    return ball_track  

def split_track(ball_track, max_gap=4, max_dist_gap=80, min_track=5):
    """ Split ball track into several subtracks in each of which we will perform
    ball interpolation.    
    :params
        ball_track: list of detected ball points
        max_gap: maximun number of coherent None values for interpolation  
        max_dist_gap: maximum distance at which neighboring points remain in one subtrack
        min_track: minimum number of frames in each subtrack    
    :return
        result: list of subtrack indexes    
    """
    list_det = [0 if x[0] else 1 for x in ball_track]
    groups = [(k, sum(1 for _ in g)) for k, g in groupby(list_det)]

    cursor = 0
    min_value = 0
    result = []
    for i, (k, l) in enumerate(groups):
        if (k == 1) & (i > 0) & (i < len(groups) - 1):
            dist = distance.euclidean(ball_track[cursor-1], ball_track[cursor+l])
            if (l >=max_gap) | (dist/l > max_dist_gap):
                if cursor - min_value > min_track:
                    result.append([min_value, cursor])
                    min_value = cursor + l - 1        
        cursor += l
    if len(list_det) - min_value > min_track: 
        result.append([min_value, len(list_det)]) 
    return result    

def interpolation(coords):
    """ Run ball interpolation in one subtrack    
    :params
        coords: list of ball coordinates of one subtrack    
    :return
        track: list of interpolated ball coordinates of one subtrack
    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    x = np.array([x[0] if x[0] is not None else np.nan for x in coords])
    y = np.array([x[1] if x[1] is not None else np.nan for x in coords])

    nons, yy = nan_helper(x)
    x[nons]= np.interp(yy(nons), yy(~nons), x[~nons])
    nans, xx = nan_helper(y)
    y[nans]= np.interp(xx(nans), xx(~nans), y[~nans])

    track = [*zip(x,y)]
    return track


def remove_outliers_final(ball_track, thresh = 150, consecutive_frames = 3):
    """
    
    After interpolation, still some outliers : we now
    check for consecutive frames where the ball
    is tracked closely to each other and then use
    this as the reference point for replacing
    outliers.
    
    """

    dists = []
    # Recalculate distances of each point pair
    for i in range(0, len(ball_track) - 1):
        # Check that it is not None
        if ball_track[i][0] and ball_track[i+1][0]:
            dist = euclidean_distance(ball_track[i], ball_track[i+1])
            dists.append(dist)
        else:
            dist = None 
            dists.append(None)

        if dist is not None and dist > thresh and i >= consecutive_frames:
            check_distances = [dists[x] for x in range(i - consecutive_frames, i)]
            # If there was any untracked distance (i.e. we lost tracking), we skip the 
            # iteration (because we could now have a large distance that is correct)
            if None in check_distances:
                continue
            check_distances = list(filter(lambda a : a is not None, check_distances))
            if ball_track[i] is not None and np.average(check_distances) <= thresh:
                ball_track[i + 1] = ball_track[i]
                # Reset the last calculated distance
                dists[-1] = 0




    return ball_track


def convert_yolo_to_tracknet_format(ball_detections_yolo):
    """
    Convert YOLO ball detections format to TrackNet format for use with write_track.
    
    Args:
        ball_detections_yolo: List of dictionaries where each dict has keys as track IDs 
                            and values as bounding boxes [x1, y1, x2, y2]
    
    Returns:
        ball_track: List of tuples (x, y) representing ball center coordinates,
                   or (None, None) if no ball detected in that frame
    """
    ball_track = []
    
    for frame_dict in ball_detections_yolo:
        if frame_dict and 1 in frame_dict:
            # Get bounding box coordinates
            x1, y1, x2, y2 = frame_dict[1]
            # Calculate center point
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ball_track.append((center_x, center_y))
        else:
            # No ball detected in this frame
            ball_track.append((None, None))
    
    return ball_track


def write_track(frames, ball_track, ball_shots_frames=[], trace = 7, draw_mode = 'circle'):

    # Use a set (because it has lookup in O(1))
    ball_shots_frames = set(ball_shots_frames)
    
    output_video_frames = []
    curr_rack_hit = -1000
    for num in range(len(frames)):
        frame = frames[num].copy()

                    


        # Draw trace of the ball
  
        # Store valid points we find
        valid_points = []
        # Save the closest current racket hit
        if num in ball_shots_frames:
            curr_rack_hit = num 
        # Collect valid points first
        for i in range(trace):
            if (num-i > 0) and ball_track[num-i][0]:
                # Check if there was a racket hit ; we want to reset the trace for each racket hit (such that is doesn't track the "weird"
                # patterns pre&post hit)
                # Only track starting every time from the racket hit 
                if (num-i) < curr_rack_hit:
                    pass 
                else:
                    x = int(ball_track[num-i][0])
                    y = int(ball_track[num-i][1])
                    valid_points.append((x, y))
            else:
                break
        
        # Draw circles for all valid points
        if draw_mode == 'circle':
            for point in valid_points:
                frame = cv2.circle(frame, point, radius=2, color=(0, 255, 255), thickness=10)
        
        elif draw_mode == 'line':
            # Draw lines between consecutive points
            for i in range(1, len(valid_points)):
                frame = cv2.line(frame, valid_points[i], valid_points[i-1], color=(0, 0, 255), thickness=3)

        output_video_frames.append(frame)
    return output_video_frames


def detect_frames_TRACKNET(video_frames, video_number, tracker, video_width, video_height, read_from_stub, stub_path):


    if stub_path is not None and read_from_stub == True:
        with open(stub_path, 'rb') as f:
            ball_detections = pickle.load(f)
        
    if stub_path is not None and read_from_stub == False: 
        # Calculate the correct scale factor for scaling back 
        # with TrackNet, we scaled to 640 width, 360 height
        scaling_x = video_width/640
        scaling_y = video_height/360
        ball_detections, dists = infer_model(video_frames, tracker, scale = (scaling_x, scaling_y))
        ball_detections = remove_outliers(ball_detections, dists)
        subtracks = split_track(ball_detections)
        for r in subtracks:
            ball_subtrack = ball_detections[r[0]:r[1]]
            ball_subtrack = interpolation(ball_subtrack)
            ball_detections[r[0]:r[1]] = ball_subtrack
    
        with open(stub_path, 'wb') as f:
            pickle.dump(ball_detections, f)


    # Final removal of outliers based on distances after initial interpolation method
  #  ball_detections = remove_outliers_final(ball_detections, thresh= 300)
    

    # Copy TrackNet ball_detections
    ball_detections_tracknet = ball_detections.copy()
    ball_detections = convert_ball_detection_to_bbox(ball_detections)

    return ball_detections, ball_detections_tracknet

    

def get_ball_shot_frames_visual(ball_positions,fps, mode = 'tracknet'):#, area):
    """Based on change of direction in the mini court coordinates"""

    if mode == 'yolo':
        for dict_item in ball_positions:
            for key, bbox in dict_item.items():
                dict_item[key] = get_center_of_bbox(bbox)
        ball_positions = [x.get(1,[]) for x in ball_positions]
    df_ball_positions = pd.DataFrame(ball_positions,columns=['x','y'])
   # df_ball_positions = df_ball_positions.iloc[area[0]:area[1]]
    # Create a rolling window for the y positions
    window_size = int(fps * 0.4) # 0.4 second windows  # TODO : old one was this #max(10, fps // 5) 
    df_ball_positions['y_rolling_mean'] = df_ball_positions['y'].rolling(window=window_size, min_periods=1, center=True).mean() # TODO : before : center False, but True makes more sense
  #  df_ball_positions['delta_y'] = df_ball_positions['y_rolling_mean'].diff()
    df_ball_positions['ball_hit'] = 0

    plt.plot(df_ball_positions['y_rolling_mean'])
    plt.xlabel('Frame Number')
    plt.ylabel('Ball y coordinate (rolling mean)')
    plt.title('Ball y coordinate (rolling mean) over time')
    plt.savefig("VISUAL.png")



    # Make this small to catch as much as possible, but large enough such that it is not too much influenced by false tracking (i.e if one/two frames see something else)
    minimum_change_frames_for_hit = int(fps * 0.2) #max(5, fps//20) 


    for i in range(len(df_ball_positions)- int(minimum_change_frames_for_hit)):
        # Look for max & mins (change of direction indicators, given our ball tracking is consistent)

        maximum = (df_ball_positions['y_rolling_mean'].iloc[i-1] < df_ball_positions['y_rolling_mean'].iloc[i]) and \
            (df_ball_positions['y_rolling_mean'].iloc[i+1] < df_ball_positions['y_rolling_mean'].iloc[i])
        
        minimum = (df_ball_positions['y_rolling_mean'].iloc[i-1] > df_ball_positions['y_rolling_mean'].iloc[i]) and \
            (df_ball_positions['y_rolling_mean'].iloc[i+1] > df_ball_positions['y_rolling_mean'].iloc[i])
        
        # Check if it really is a directional change
        if minimum or maximum:
            array_check = []

            # Check the area around the current position we are looking at
            for change_frame in range(i- int(minimum_change_frames_for_hit), i+ int(minimum_change_frames_for_hit)):
                # Exclude the min/max to be checked
                if change_frame == i:
                    continue 
                # Check if in the range, our hypothesized minimum is really a minimum
                if minimum:
                    array_check.append(df_ball_positions['y_rolling_mean'].iloc[change_frame] > df_ball_positions['y_rolling_mean'].iloc[i])
                elif maximum:
                    array_check.append(df_ball_positions['y_rolling_mean'].iloc[change_frame] < df_ball_positions['y_rolling_mean'].iloc[i])
            
            if all(array_check) and len(array_check) > 0:
                df_ball_positions['ball_hit'].iloc[i] = 1
            '''
            else:
                # In this case, we smooth the current area even more to make another check (its possible that we have some fluctuation around the min/max due to
                # the ball tracker not being perfectly consistent)
                for window_size_curr in [int(fps * 0.6), int(fps * 0.8), int(fps * 0.9)]:  # TODO : old one was this #[15,20,25,30]:
                    array_check = []
                    df_ball_y_positions_of_interest = df_ball_positions['y'].rolling(window=window_size_curr, min_periods=1, center=True).mean()

                    # Check the area around the current position we are looking at
                    for change_frame in range(i- int(minimum_change_frames_for_hit), i+ int(minimum_change_frames_for_hit)):
                        # Exclude the min/max to be checked
                        if change_frame == i:
                            continue 
                        # Check if in the range, our hypothesized minimum is really a minimum
                        if minimum:
                            array_check.append(df_ball_positions['y_rolling_mean'].iloc[change_frame] > df_ball_positions['y_rolling_mean'].iloc[i])
                        elif maximum:
                            array_check.append(df_ball_positions['y_rolling_mean'].iloc[change_frame] < df_ball_positions['y_rolling_mean'].iloc[i])
                    
                    # If with more smoothing we found something, break out of the loop
                    if all(array_check) and len(array_check) > 0:
                        df_ball_positions['ball_hit'].iloc[i] = 1
                        break
            '''


    hit_frames = df_ball_positions[df_ball_positions['ball_hit']==1].index.tolist()

    return hit_frames



def get_ball_shot_frames_audio(audio_file, fps, height=0.01, distance_c = 0.25, prominence=0.006, plot=False):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Apply bandpass filter (150Hz-1800Hz)
    nyquist = 0.5 * sr
    low = 150 / nyquist
    high = 1800 / nyquist
    sos = butter(N=6, Wn=[low, high], btype='band', output='sos')
    y_filtered = sosfilt(sos, y)
    
    # Compute the envelope of the filtered signal
    y_abs = np.abs(y_filtered)
    
    # Apply smoothing to the envelope 
    window_size = int(0.01 * sr)  # 10ms window
    y_envelope = np.convolve(y_abs, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks in the envelope
    # Lower height threshold to catch more peaks
    peaks, _ = find_peaks(y_envelope, 
                        height=height,  # Lower threshold to catch more peaks
                        distance=int(distance_c * sr),  # Minimum distance between peaks 
                        prominence=prominence)  # Find all distinct peaks 
    
    # Convert peak positions to time (seconds)
    hit_times = peaks / sr
    
    # Convert times to frame numbers
    hit_frames = [int(round(time * fps)) for time in hit_times]
    
    if plot:
        plt.figure(figsize=(12, 10))
        
        # Convert audio samples to frame numbers for plotting
        frame_numbers = np.linspace(0, len(y_filtered) * fps / sr, len(y_filtered))
        
        # Plot filtered waveform with detected hits
        plt.subplot(3, 1, 1)
        plt.plot(frame_numbers, y_filtered)
        plt.vlines(hit_frames, -0.2, 0.2, color='r', linewidth=1)
        plt.title('Filtered Audio Waveform (150Hz-1800Hz) with Detected Hits')
        plt.xlabel('Frame Number')
        
        # Plot the envelope with detected peaks
        plt.subplot(3, 1, 2)
        plt.plot(frame_numbers, y_envelope)
        plt.vlines(hit_frames, 0, np.max(y_envelope), color='r', linewidth=1, label='Detected Hits')
        plt.title('Signal Envelope with Detected Peaks')
        plt.xlabel('Frame Number')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("AUDIO.png")
        # plt.show()
    
    return hit_frames




def get_ball_shot_frames_audio_refinement(audio_file, fps, frames_start=None, frames_end=None, 
                              peak_height=0.02, peak_prominence=0.01, peak_distance=0.5,
                              plot=False):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    
    # Convert frame range to time range if specified
    time_start = frames_start / fps if frames_start is not None else 0
    time_end = frames_end / fps if frames_end is not None else len(y) / sr
    
    # Convert time range to sample indices
    sample_start = int(time_start * sr)
    sample_end = min(int(time_end * sr), len(y))
    
    # Extract the relevant section of audio
    y_section = y[sample_start:sample_end]
    
    # Apply bandpass filter (150Hz-1800Hz)
    nyquist = 0.5 * sr
    low = 150 / nyquist
    high = 1800 / nyquist
    sos = butter(N=6, Wn=[low, high], btype='band', output='sos')
    y_filtered = sosfilt(sos, y_section)
    
    # Compute the envelope of the filtered signal
    y_abs = np.abs(y_filtered)
    
    # Apply smoothing to the envelope
    window_size = int(0.01 * sr)  # 10ms window
    y_envelope = np.convolve(y_abs, np.ones(window_size)/window_size, mode='same')
    
    # Find peaks in the envelope with customizable parameters
    peaks, _ = find_peaks(y_envelope, 
                        height=peak_height, 
                        distance=int(peak_distance * sr),  
                        prominence=peak_prominence) 
    
    # Convert peak positions to time (seconds), adding the start time offset
    hit_times = (peaks / sr) + time_start
    
    # Convert times to frame numbers
    hit_frames = [int(round(time * fps)) for time in hit_times]

    return hit_frames


import bisect

def find_closest_matches(array_a, array_b, threshold):
    """
    Find all elements in array_b that are within a given threshold distance
    of each element in array_a. Returns in the same format as the original function.
    """
    # Sort array_b first
    sorted_b = sorted(array_b)
    matches = []
    
    for a in array_a:
        # Find insertion point
        pos = bisect.bisect_left(sorted_b, a)
        
        # Check elements to the left (smaller values)
        left_idx = pos - 1
        while left_idx >= 0 and a - sorted_b[left_idx] <= threshold:
            matches.append((a, sorted_b[left_idx], abs(a - sorted_b[left_idx])))
            left_idx -= 1
        
        # Check elements to the right (larger values)
        right_idx = pos
        while right_idx < len(sorted_b) and sorted_b[right_idx] - a <= threshold:
            matches.append((a, sorted_b[right_idx], abs(a - sorted_b[right_idx])))
            right_idx += 1
        
        # If no matches were found within threshold, add the closest one
        if left_idx + 1 == pos and right_idx == pos:
            if pos == 0:
                closest_b = sorted_b[0]
            elif pos == len(sorted_b):
                closest_b = sorted_b[-1]
            else:
                # Compare the two neighboring elements
                if abs(a - sorted_b[pos-1]) <= abs(a - sorted_b[pos]):
                    closest_b = sorted_b[pos-1]
                else:
                    closest_b = sorted_b[pos]
            matches.append((a, closest_b, abs(a - closest_b)))
    
    return matches


def cluster_court_positions(hits):
    """
    Cluster consecutive court positions from a list of (hit_frame, court_position) tuples.
    
    Args:
        hits: List of (hit_frame, court_position) tuples, ordered by hit_frame ascending
        
    Returns:
        List of clusters, where each cluster is a list of (hit_frame, court_position) tuples
        with the same court position
    """
    if not hits:
        return []
    
    clusters = []
    current_cluster = [hits[0]]
    current_position = hits[0][1]
    
    for hit in hits[1:]:
        if hit[1] == current_position:
            # Same court position, add to current cluster
            current_cluster.append(hit)
        else:
            # Different court position, start a new cluster
            clusters.append(current_cluster)
            current_cluster = [hit]
            current_position = hit[1]
    
    # Don't forget to add the last cluster
    if current_cluster:
        clusters.append(current_cluster)
    
    return clusters


def combine_audio_visual(ball_shots_frames_visual, ball_shots_frames_audio, fps, player_boxes, ball_detections, keypoints, max_distance_param = 10,MINI_COURT = False, net_y = 450, CLUSTERING = False):
    """
    Idea is that audio will detect every little peak in audio and therefore also a lot of wrong
    information (shoe sounds, players moaning, crowd screaming etc.). We try to refine that by
    finding the closest match to every found directional change in the visual approach ;
    if there is no close match to a found directional change in the visual approach, we deem it
    as a faulty detected racket hit.

    Additionally, also check the position of the ball to the player(s) and see if the ball
    is close to one player (indicating that a racket hit is happening). This is in order
    to make the racket hit detection even more robust.


    max_distance_param would probably need video-to-video tuning.
    """
    max_distance_param = 0.5 * fps 
    if MINI_COURT == True:
        final_racket_hits = []

        # Get the player positions over all frames
        player_positions_centered = defaultdict(list)
        # Process each frame
        for frame_num, player_bbox in enumerate(player_boxes):
            # Process players
            output_player_bboxes_dict = {}
            
            for player_id, bbox in player_bbox.items():
                # Get center positions of players
                center_pos = bbox

                player_positions_centered[player_id].append(center_pos)

        # Get player ids
        player_ids = list(player_positions_centered.keys())

        player_0_pos = player_positions_centered[player_ids[0]]
        player_1_pos = player_positions_centered[player_ids[1]]



        # Calculate the net's y-coordinate based on the keypoint found at the net
        # since we use the mini court coordinates, one single keypoint is enough for us
        net_y = (keypoints[1] + keypoints[5]) // 2
        matches = find_closest_matches(ball_shots_frames_visual, ball_shots_frames_audio, threshold= max_distance_param)

        for elem in matches:
            # Only add the ones that are actually close to one another
            if elem[2] <= max_distance_param:
                
                y_coord_hit_frame = ball_detections[elem[1]][1][1]
                change = False
                # if we have no track for the mapping to the audio based shot, use the visual one
                if y_coord_hit_frame == None:
                    change = True
                    y_coord_hit_frame = ball_detections[elem[0]][1][1]
                if y_coord_hit_frame <= net_y:
                    # Upper court : 1
                    if change == False:
                        final_racket_hits.append((elem[1], 1))
                    else:
                        final_racket_hits.append((elem[0], 1))
                else:
                    # Lower court : 0
                    if change == False:
                        final_racket_hits.append((elem[1],0))
                    else:
                        final_racket_hits.append((elem[0],0))





        # Cluster the found positions of each court side (i.e such that we can then determine
        # which is the correct racket hit, because the player only hits the ball once)

        clustered_final_racket_hits = cluster_court_positions(final_racket_hits)


        complete_final_racket_hits = []
        # Now search the closest the ball was to the player on the respective court side
        for clustered_hit_frame in clustered_final_racket_hits:
            curr_min = np.inf
            correct_hit_frame = 0
            print(clustered_hit_frame)


            for hit_frame in clustered_hit_frame:
                distances = []
                
                #find all ball detections that are behind he court line and take the one that has
                #  the highest distance to the respective keypoint on the current side in terms of being completely out of the field
     
            #    if ball_detections[hit_frame[0]][1][1] < keypoints[1]:
                    # Measure y-distance 
            #        dist_ball_court = keypoints[1] - ball_detections[hit_frame[0]][1][1]
            #        distances.append((hit_frame[0], dist_ball_court))
                


             #   elif ball_detections[hit_frame[0]][1][1] > keypoints[5]:
             #       dist_ball_court = ball_detections[hit_frame[0]][1][1] - keypoints[5]
             #       distances.append((hit_frame[0], dist_ball_court))



                # Calculate the euclidean distances of players to the possible hits and find the minimal one
                dist_0 = int(euclidean_distance(tuple(map(int, ball_detections[hit_frame[0]][1])), player_0_pos[hit_frame[0]]))
                dist_1 = int(euclidean_distance(tuple(map(int, ball_detections[hit_frame[0]][1])), player_1_pos[hit_frame[0]]))
                found_min = min(dist_0,dist_1)
                if found_min < curr_min:
                    correct_hit_frame = hit_frame[0]
                    curr_min = found_min 
            
            # If there was a out-of-bounds ball detected, find the one with max distance
          #  if distances:
          #      correct_hit_frame = max(distances, key = lambda x:x[1])[0]
            complete_final_racket_hits.append(correct_hit_frame)

        return sorted(list(set(complete_final_racket_hits)))
            




    else:

        final_racket_hits = []

        # Get the player positions over all frames
        player_positions_centered = defaultdict(list)
        # Process each frame
        for frame_num, player_bbox in enumerate(player_boxes):
            # Process players
            output_player_bboxes_dict = {}
            
            for player_id, bbox in player_bbox.items():
                # Get center positions of players
                center_pos = get_center_of_bbox(bbox)

                player_positions_centered[player_id].append(center_pos)

        # Get player ids
        player_ids = list(player_positions_centered.keys())

        player_0_pos = player_positions_centered[player_ids[0]]
        player_1_pos = player_positions_centered[player_ids[1]]



        # Calculate the net's y-coordinate based on the keypoints found at the net
       # net_y = ((keypoints[10][1] + keypoints[11][1]) / 2) - adjustment

        matches = find_closest_matches(ball_shots_frames_visual, ball_shots_frames_audio)

        for elem in matches:
            # Only add the ones that are actually close to one another
            if elem[2] <= max_distance_param:
                
                # And also check in which court side we are
                y_coord_hit_frame = ball_detections[elem[1]][1]
                change = False
                # if we have no track for the mapping to the audio based shot, use the visual one
                if y_coord_hit_frame == None:
                    change = True
                    y_coord_hit_frame = ball_detections[elem[0]][1]
                if y_coord_hit_frame <= net_y:
                    # Upper court : 1
                    if change == False:
                        final_racket_hits.append((elem[1], 1))
                    else:
                        final_racket_hits.append((elem[0], 1))
                else:
                    # Lower court : 0
                    if change == False:
                        final_racket_hits.append((elem[1],0))
                    else:
                        final_racket_hits.append((elem[0],0))

        # Check if first from audio is in : should be the initial racket hit (serve) , that the visual model
        # might have more problems with (due to ball tracking of first hit)

        if final_racket_hits[0] != ball_shots_frames_audio[0]:
            if ball_detections[ball_shots_frames_audio[0]][1] <= net_y:
                final_racket_hits.insert(0,(ball_shots_frames_audio[0], 1))
            else:
                final_racket_hits.insert(0,(ball_shots_frames_audio[0], 0))


        # Cluster the found positions of each court side (i.e such that we can then determine
        # which is the correct racket hit, because the player only hits the ball once)


        if CLUSTERING == True:
            clustered_final_racket_hits = cluster_court_positions(final_racket_hits)


            complete_final_racket_hits = []
            # Now search the closest the ball was to the player on the respective court side
            for clustered_hit_frame in clustered_final_racket_hits:
                curr_min = np.inf
                correct_hit_frame = 0
                for hit_frame in clustered_hit_frame:
                    # Calculate the euclidean distances of players to the possible hits and find the minimal one
                    dist_0 = int(euclidean_distance(tuple(map(int, ball_detections[hit_frame[0]])), player_0_pos[hit_frame[0]]))
                    dist_1 = int(euclidean_distance(tuple(map(int, ball_detections[hit_frame[0]])), player_1_pos[hit_frame[0]]))
                    found_min = min(dist_0,dist_1)
                    if found_min < curr_min:
                        correct_hit_frame = hit_frame[0]
                        curr_min = found_min 
                
                complete_final_racket_hits.append(correct_hit_frame)
        else:
            complete_final_racket_hits = set()
            for i in final_racket_hits:
                complete_final_racket_hits.add(i[0])

            complete_final_racket_hits = sorted(list(complete_final_racket_hits))



    return sorted(list(set(complete_final_racket_hits)))
    
        


            






def refine_audio(ball_shots_frames_audio, fps, audio_file):
    """ 
    Idea is :
        The first hit in each game is very loud and not registered by the "change of direction"
        logic --> therefore we detect it with audio. Even if the game starts in the middle,
        since we work on a set, it should still be okay, but with this we catch the possible
        first hit.
        Audio signals give us a more clear point of where the ball was really hit : Therefore
        we want to use these frames as our reference points
    """
    
 
    # use a set to not have any duplicates
    ball_shots_frames_final = set()

    # Add all audio hits (since these are very consistent)
    for i in ball_shots_frames_audio:
        ball_shots_frames_final.add(i)


    ## GAP CHECKING 

    # Next, we need to take into account that there will be more silent hits that our audio model might not recognize, 
    # Therefore, we check if we have large gaps in the audio results and then refine the audio detection by detecting
    # lower peaks in the audio signal


    thresh = fps * 1.5 # 1.5 seconds

    for idx in range(0, len(ball_shots_frames_audio) - 1, 1):
        if ball_shots_frames_audio[idx + 1] - ball_shots_frames_audio[idx] > thresh:
            
            peak_heights = [0.19,0.18,0.17,0.16,0.15,0.14,0.13,0.12,0.11,0.10,0.009,0.008]

            for peak_height in peak_heights:
                refined_hits = get_ball_shot_frames_audio_refinement(audio_file, fps, frames_start= ball_shots_frames_audio[idx], 
                                                                      frames_end = ball_shots_frames_audio[idx + 1], peak_height = peak_height)
            
                if refined_hits:
                    for i in refined_hits:
                        # Too close (i.e. still from old racket hit signal)
                        if (ball_shots_frames_audio[idx]  <= i <= ball_shots_frames_audio[idx] + fps//10) or (ball_shots_frames_audio[idx + 1] - fps//10  <= i <= ball_shots_frames_audio[idx + 1]):
                            pass
                        else:
                            ball_shots_frames_final.add(i)
                    break 
            

                   

    ball_shots_frames_final = sorted(list(ball_shots_frames_final))
    return ball_shots_frames_final





def draw_racket_hits(video_frames, hit_frames):
    """ Draw the ball hits on the video frames """
    output_video_frames = []
    counter = 0
    for i,frame in enumerate(video_frames):
        cv2.putText(frame, f"Racket Hits: {counter}", (20, 100), cv2.FONT_HERSHEY_DUPLEX , 1, (255,144,30), 2)
        if i in hit_frames:
            counter += 1

    
        output_video_frames.append(frame)
    
    return output_video_frames 

def convert_ball_detection_to_bbox(ball_track, padding=5):
    """ Convert ball detection to bounding box format, similar as in YOLO implementation.
        Therefore, we want to return a list of dictionaries (one for each frame), with
        format 1 : [x_min, y_min, x_max, y_max]"""
    
    lst_of_bboxes = []
    
    # Iterate over all TrackNet (x,y) coordinates
    # len(ball_track) --> number of frames

    for i in range(len(ball_track)):
        bboxes = {}
        if ball_track[i][0]:
            x = ball_track[i][0]
            y = ball_track[i][1]
            # key 1 and values x_min, y_min, x_max, y_max
            bboxes[1] = [x-padding, y-padding, x+padding, y+padding]
        else: # take last consistent track
            bboxes[1] = lst_of_bboxes[-1][1]
        
        lst_of_bboxes.append(bboxes)

    return lst_of_bboxes

def filter_ball_shots_by_player(ball_shots_frames, player_detections, ball_detections):
    """
    Filters ball shot frames into two lists - one for upper player and one for lower player.
    The first shot is assigned to the player whose upper bbox y-coordinate is closest to the ball, then alternates.
    
    Args:
        ball_shots_frames (list): Sorted list of frame numbers where ball shots were detected
        player_detections (list): List of dictionaries containing player bounding boxes for each frame
        ball_detections (list): List of dictionaries containing ball positions for each frame
        
    Returns:
        tuple: (upper_player_shots, lower_player_shots) - Lists of frame numbers for each player's shots
    """
    if not ball_shots_frames or len(player_detections) == 0 or len(ball_detections) == 0:
        return [], []
    
    # Get the first ball shot frame
    first_shot_frame = ball_shots_frames[0]
    
    # Check if the first shot frame is within range
    if (first_shot_frame >= len(player_detections) or 
        first_shot_frame >= len(ball_detections) or
        1 not in ball_detections[first_shot_frame]):
        print("Warning: Cannot determine first player. Using default assignment.")
        # Default assignment if we can't determine
        upper_player_shots = ball_shots_frames[::2]  # Even indices: 0, 2, 4, ...
        lower_player_shots = ball_shots_frames[1::2]  # Odd indices: 1, 3, 5, ...
        return upper_player_shots, lower_player_shots
    
    # Get ball position at first shot
    ball_pos = ball_detections[first_shot_frame][1]
    
    # Get player positions at first shot frame
    player_positions = player_detections[first_shot_frame]
    
    # Get all player IDs
    all_player_ids = list(player_positions.keys())
    if len(all_player_ids) < 2:
        print("Warning: Less than 2 players detected. Cannot assign shots properly.")
        return [], []
    
    # Determine which player is upper and which is lower based on y-coordinate
    player_y_positions = {}
    for player_id, player_bbox in player_positions.items():
        # Use center y-coordinate of bbox to determine upper vs lower player
        y_center = (player_bbox[1] + player_bbox[3]) / 2
        player_y_positions[player_id] = y_center
    
    # Sort players by y-coordinate (smaller y = upper, larger y = lower)
    sorted_players = sorted(player_y_positions.items(), key=lambda x: x[1])
    upper_player_id = sorted_players[0][0]  # Player with smaller y-coordinate
    lower_player_id = sorted_players[1][0]  # Player with larger y-coordinate
    
    # Find the player whose upper bbox y-coordinate is closest to the ball's y-coordinate for the first shot
    min_y_distance = float('inf')
    serving_player_id = None
    
    for player_id, player_bbox in player_positions.items():
        # Get the upper y-coordinate of the player's bounding box
        # player_bbox format: [x_min, y_min, x_max, y_max]
        y_upper = player_bbox[1]  # Top of the bounding box
        
        # Calculate the absolute difference in y-coordinates between ball and player's upper bbox
        y_distance = abs(ball_pos[1] - y_upper)
        
        if y_distance < min_y_distance:
            min_y_distance = y_distance
            serving_player_id = player_id
    
    if serving_player_id is None:
        print("Warning: Could not find closest player. Using default assignment.")
        # Default assignment
        upper_player_shots = ball_shots_frames[::2]
        lower_player_shots = ball_shots_frames[1::2]
        return upper_player_shots, lower_player_shots
    
    # Assign shots alternately, starting with the serving player
    serving_player_shots = []
    receiving_player_shots = []
    
    for i, shot_frame in enumerate(ball_shots_frames):
        if i % 2 == 0:  # Even indices: 0, 2, 4, ... (serving player)
            serving_player_shots.append(shot_frame)
        else:  # Odd indices: 1, 3, 5, ... (receiving player)
            receiving_player_shots.append(shot_frame)
    
    # Return shots in the order (upper_player_shots, lower_player_shots)
    if serving_player_id == upper_player_id:
        # Serving player is the upper player
        return serving_player_shots, receiving_player_shots
    else:
        # Serving player is the lower player
        return receiving_player_shots, serving_player_shots

def filter_ball_detections_by_player(frames, ball_mini_court_detections, minicourt):
    """
    Filters ball detection frames into two lists - one for upper player and one for lower player.
    
    Args:
        frames (list): Sorted list of frame numbers where racket hits were detected ore where a ball landed
        ball_mini_court_detections (list): List of dictionaries containing ball positions in mini court
        minicourt (class): MiniCourt object 
    Returns:
        tuple: (upper_player_hits, lower_player_hits) - Lists of frame numbers for each player's hits
    """
    upper_player_area_ball_detections = []
    lower_player_area_ball_detections = []
    
    if not frames or len(ball_mini_court_detections) == 0:
        return upper_player_area_ball_detections, lower_player_area_ball_detections
    
    # Get the ball position at the first ball detection frame
    first_ball_detection = frames[0]
    
    # Calculate the net's y-coordinate (middle of court)
    net_y = minicourt.net_y
    
    # Default to upper player first (will be used if we can't determine)
    is_upper_first = True
    
    # Check if the first ball detection frame is within range of available frames
    if first_ball_detection < len(ball_mini_court_detections):
        # Try to get ball position from the dictionary
        if 1 in ball_mini_court_detections[first_ball_detection]:
            ball_position = ball_mini_court_detections[first_ball_detection][1]
            
            # Determine first player based on ball position
            if ball_position[1] <= net_y:
                # Ball is in upper part of court (hit/bounce in upper player area)
                is_upper_first = True
            else:
                # Ball is in lower part of court (hit/bounce in lower player area)
                is_upper_first = False
        else:
            print("Warning: No ball detected at first hit frame. Defaulting to upper player first.")
    else:
        print("Warning: First hit frame is out of range. Defaulting to upper player first.")
    
    # Distribute frames to each player based on alternating pattern
    for i, frame in enumerate(frames):
        if is_upper_first:
            if i % 2 == 0:  # 0, 2, 4, ...
                upper_player_area_ball_detections.append(frame)
            else:  # 1, 3, 5, ...
                lower_player_area_ball_detections.append(frame)
        else:
            if i % 2 == 0:  # 0, 2, 4, ...
                lower_player_area_ball_detections.append(frame)
            else:  # 1, 3, 5, ...
                upper_player_area_ball_detections.append(frame)
    
    return upper_player_area_ball_detections, lower_player_area_ball_detections