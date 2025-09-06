from utils import (read_video, 
                   save_video, 
                   write_track, 
                   get_ball_shot_frames_audio, 
                   draw_racket_hits, 
                   euclidean_distance,
                   convert_pixel_distance_to_meters,
                   draw_player_stats,
                   create_black_video,
                   scraping_data_for_inference,
                   detect_frames_TRACKNET,
                   cluster_series,
                   filter_ball_detections_by_player,
                   filter_ball_shots_by_player,
                   draw_debug_window,
                   draw_frames_number,
                   draw_ball_landings,
                   map_court_position_to_player_id,
                   get_ball_shot_frames_visual, 
                   combine_audio_visual,
                   convert_mp4_to_mp3,
                   convert_yolo_to_tracknet_format
                   )

from trackers import (PlayerTracker, BallTrackerNetTRACE, BallTracker)
from mini_court import MiniCourt
from ball_landing import (BounceCNN, make_prediction, evaluation_transform)
from court_line_detector import CourtLineDetector

import torch 
import pandas as pd
import info
import os 
import pickle
import numpy as np
from copy import deepcopy



def main():


    ######## CONFIG ########
    TRACE = 6 # Set trace to same amount that BounceCNN was trained on (currently : 6)
    
    # Select the player
    SELECTED_PLAYER = 'Upper' # 'Upper' or 'Lower'
    
    # Draw Options
    DRAW_MINI_COURT = True
    DRAW_STATS_BOX = False
    COMPUTE_PROBABILITIES = True  # Whether to compute and draw probabilities on the mini court (can be slow)

    # Debugging Mode to use ground truth values for racket hits and ball landings (if available)
    DEBUG = True
    DRAW_DEBUG_WINDOW = False  # Whether to draw the debug window with all info (can be cluttered)

    # Video Number to run inference on
    VIDEO_NUMBER = 1008
    print(f"Running inference on video {VIDEO_NUMBER}")
    
    # Insert ground truth values for the racket hits and ball landings for best accuracy
    GT_BOUNCES_FRAMES = []
    GT_RACKET_HITS_FRAMES = []
    GT_SERVE_PLAYER = None  # 'Upper' or 'Lower' if known, else None

    # Video Paths (data/new_input_videos/)
    INPUT_VIDEO_PATH = f'data/new_input_videos/input_video_{VIDEO_NUMBER}.mp4'  #
    INPUT_VIDEO_PATH_AUDIO = f'data/new_input_videos/input_video_{VIDEO_NUMBER}_audio.mp3'
    OUTPUT_VIDEO_PATH = f'output/final/output_video{VIDEO_NUMBER}.mp4'

    # Check if we already processed that video by looking if output with video number reference exists (for faster testing)
    if os.path.exists(OUTPUT_VIDEO_PATH):
        READ_STUBS = True
    else:
        READ_STUBS = False
    
    if not os.path.exists(INPUT_VIDEO_PATH_AUDIO):
        print(f"Converting video {VIDEO_NUMBER} to audio")
        convert_mp4_to_mp3(INPUT_VIDEO_PATH, INPUT_VIDEO_PATH_AUDIO)
        
    # Check if GPU is available
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using DEVICE: {DEVICE}")



    ######## DETECTIONS ########

    # Initialize Ball Tracker
    
    # YOLO
    ball_tracker_yolo = BallTracker(model_path = 'models/yolo11best.pt')
    
    # TrackNet
    ball_tracker_TRACKNET = BallTrackerNetTRACE(out_channels= 2)
    saved_state_dict = torch.load('models/tracknet_TRACE.pth', map_location=DEVICE, weights_only=False)
    ball_tracker_TRACKNET.load_state_dict(saved_state_dict['model_state'])
    ball_tracker_TRACKNET.to(DEVICE)
    ball_tracker_TRACKNET.eval() 

    # Initialize Player Tracker
    player_tracker = PlayerTracker(model_path = 'models/yolov8x.pt')

    # Initialize Courline Detector
    courtline_detector = CourtLineDetector(model_path = 'models/keypoints_model.pth', machine = DEVICE) 

    # Read in the video
    video_frames, fps, video_width, video_height = read_video(INPUT_VIDEO_PATH)

    # Detect & Track Players
    player_detections = player_tracker.detect_frames(video_frames,
                                                    read_from_stub = READ_STUBS,
                                                    stub_path = f'tracker_stubs/player_detections_{VIDEO_NUMBER}.pkl')

    # Detect & Track Ball (TrackNet)
    ball_detections, ball_detections_tracknet = detect_frames_TRACKNET(video_frames, video_number=VIDEO_NUMBER, tracker=ball_tracker_TRACKNET,
                        video_width=video_width, video_height= video_height, read_from_stub = READ_STUBS, 
                        stub_path=  f'tracker_stubs/tracknet_ball_detections_{VIDEO_NUMBER}.pkl')
    
    # Detect & Track Ball (YOLO)
    ball_detections_YOLO = ball_tracker_yolo.detect_frames(video_frames, 
                                                           read_from_stub = READ_STUBS, 
                                                            stub_path = f'tracker_stubs/ball_detections_YOLO_{VIDEO_NUMBER}.pkl')
            
    ball_detections_YOLO = ball_tracker_yolo.interpolate_ball_positions(ball_detections_YOLO)

    # Detect court lines (on just the first frame, then they are fixed) 
    refined_keypoints = courtline_detector.predict(video_frames[0])

    # Filter players (such that only the two actual players are tracked)
    player_detections, chosen_players_ids = player_tracker.choose_and_filter_players(refined_keypoints, player_detections)

    # Map player positions to player IDs
    player_position_to_id = map_court_position_to_player_id(refined_keypoints, player_detections)
    print(f"Player mapping: Upper is player_{player_position_to_id.get('Upper')}, Lower is player_{player_position_to_id.get('Lower')}")
    print(f"Selected Player : {SELECTED_PLAYER}")
    print(f"Selected Player ID : {player_position_to_id.get(SELECTED_PLAYER)}")
 
    # Initialize Mini Court
    mini_court = MiniCourt(video_frames[0])

    # Convert player positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                            ball_detections,
                                                                                                                            refined_keypoints,
                                                                                                                            chosen_players_ids)
    
    mini_court_keypoints = mini_court.drawing_key_points

    # Get Racket Hits based on audio & visual information

    # Get first hit with less sensitive audio peak detection
    first_hit = (get_ball_shot_frames_audio(INPUT_VIDEO_PATH_AUDIO, fps, height = 0.01, prominence=0.01))[0]
    ball_shots_frames_visual = get_ball_shot_frames_visual(ball_detections_YOLO, fps, mode = 'yolo')
    ball_shots_frames_audio = get_ball_shot_frames_audio(INPUT_VIDEO_PATH_AUDIO, fps, plot = True)

    ball_shots_frames = combine_audio_visual(ball_shots_frames_visual= ball_shots_frames_visual,
                                                  ball_shots_frames_audio= ball_shots_frames_audio, 
                                                  fps = fps,
                                                  player_boxes = player_mini_court_detections, 
                                                  keypoints = mini_court_keypoints,
                                                  ball_detections = ball_mini_court_detections,
                                                  max_distance_param = 7,
                                                  MINI_COURT= True,
                                                  CLUSTERING= False)


    if ball_shots_frames[0] != first_hit:
        ball_shots_frames.insert(0, first_hit)
        


    print("Ball Shots from Visual : ", ball_shots_frames_visual)
    print("Ball Shots from Audio : ", ball_shots_frames_audio)
    print("Combined :", ball_shots_frames)


    ball_shots_frames_stats = ball_shots_frames.copy()

    # First, create a completely black video with same dimensions & fps of actual video 
    frame_count = len(video_frames)
    input_video_black_path = f"data/trajectory_model_videos/output_video{VIDEO_NUMBER}.mp4"
    create_black_video(input_video_black_path, video_width, video_height, fps, frame_count)

    # Read in this video
    video_frames_black, fps, video_width, video_height = read_video(input_video_black_path)

    # Draw Ball Detection into black video
    output_frames_black = write_track(video_frames_black, ball_detections_tracknet, ball_shots_frames, trace = 10, draw_mode= 'circle')

    # Draw Keypoints (and lines) of the court into black video 
   # output_frames_black = courtline_detector.draw_keypoints_on_video(output_frames_black, refined_keypoints)

    scraping_data_for_inference(video_n= VIDEO_NUMBER, output_path = 'data_inference', input_frames = output_frames_black,
                                 ball_shots_frames = ball_shots_frames , trace = TRACE, ball_detections = ball_detections_tracknet)

    # Instantiate Bounce Model
    model_bounce = BounceCNN()
    with open('data_bounce_stubs/data_mean_std.pkl', 'rb') as f:
        data_mean_and_std = pickle.load(f)
    
    mean = data_mean_and_std[0]
    std = data_mean_and_std[1]

    # Make Predictions
    predictions, confidences, img_idxs = make_prediction(model = model_bounce, best_model_path = 'models/best_bounce_model.pth',
                                         input_frames_directory = 'data_inference/images_inference', transform = evaluation_transform(mean, std), device = DEVICE)
    
    # Get Bounce Frames
    mask = np.array(predictions) == 1
    img_idxs_bounce = np.array(img_idxs)[mask].tolist()
    ball_landing_frames = cluster_series(img_idxs_bounce)
    print(f"Predicted V Shaped Frames : {img_idxs_bounce}")
    print(f"Predicted Bounce Frames : {ball_landing_frames}")
    
    # Filter Ball Detections by Player
    upper_court_balls_frames, lower_court_balls_frames = filter_ball_detections_by_player(ball_landing_frames, ball_mini_court_detections, mini_court)
    print(f"Upper Court Bounce Frames : {upper_court_balls_frames}")
    print(f"Lower Court Bounce Frames : {lower_court_balls_frames}")
    if SELECTED_PLAYER == 'Upper':
        player_balls_frames = lower_court_balls_frames
    else:
        player_balls_frames = upper_court_balls_frames
    print(f"Selected player is {SELECTED_PLAYER} so the bounce used in ball heatmap are in the opposite court and are: {player_balls_frames}")
    
    


    ######## GROUND TRUTH ########
    
    # Set the ground truth values for the racket hits and ball landings
    if DEBUG:
        
        # Example ground truth values for videos
        match VIDEO_NUMBER:
            case 101:
                GT_RACKET_HITS_FRAMES = [12, 28, 58, 89, 118, 153, 179, 211, 243, 283]
                GT_BOUNCES_FRAMES = [20,50,77,106,138,168,197,230,270,301]
                GT_SERVE_PLAYER = 'Upper'
                
            case 103:
                GT_RACKET_HITS_FRAMES = [14, 32, 83]
                GT_BOUNCES_FRAMES = [23, 66, 97]
                GT_SERVE_PLAYER = 'Lower'

                
            case 105:
                GT_RACKET_HITS_FRAMES = [34, 54, 91, 126, 162, 187, 222, 257]
                GT_BOUNCES_FRAMES = [48, 88, 115, 146, 186, 208, 256, 269] 
            
            case 107:
                GT_RACKET_HITS_FRAMES = [58, 84, 119, 144, 182, 219, 256]
                GT_BOUNCES_FRAMES = [69, 108, 133, 175, 201, 255, 278]
                GT_SERVE_PLAYER = 'Lower'  
            
            case 109:
                GT_RACKET_HITS_FRAMES = [23, 50, 90, 130, 170, 210, 250]
                GT_BOUNCES_FRAMES = [34, 77, 115, 155, 195, 235, 275]
                GT_SERVE_PLAYER = 'Lower'
                
            case 116:
                GT_RACKET_HITS_FRAMES = [1, 22, 62]
                GT_BOUNCES_FRAMES = [13, 52, 82]
                GT_SERVE_PLAYER = 'Upper'
                
            case 1008:
                GT_RACKET_HITS_FRAMES = [7, 25, 61, 89, 118, 146, 179]
                GT_BOUNCES_FRAMES = [17, 48, 79, 109, 138, 172, 201]
                GT_SERVE_PLAYER = 'Upper'

            
            case 1009:
                GT_RACKET_HITS_FRAMES = [1, 11, 47]
                GT_BOUNCES_FRAMES = [3, 37, 68]
                GT_SERVE_PLAYER = 'Upper'

                
            case 1010:
                GT_RACKET_HITS_FRAMES = [10, 32, 65, 98, 135, 166, 192, 222, 257, 286, 316, 348, 393, 423, 456, 482, 514, 543, 569, 601, 647, 676, 717, 750, 784, 816]
                GT_BOUNCES_FRAMES = [23, 54, 84, 125, 153, 183, 214, 242, 275, 304, 336, 383, 416, 444, 477, 503, 531, 565, 591, 636, 664, 711, 739, 769, 805, 836]
                GT_SERVE_PLAYER = 'Upper'

            case 1012:
                GT_RACKET_HITS_FRAMES = [23, 42, 74]
                GT_BOUNCES_FRAMES = [34, 65, 90]  
                GT_SERVE_PLAYER = 'Upper'

                
            case 1026:
                GT_RACKET_HITS_FRAMES = [8, 28, 59, 103]
                GT_BOUNCES_FRAMES = [19, 51, 95, 128]  
                GT_SERVE_PLAYER = 'Upper'

                
        if GT_SERVE_PLAYER == 'Upper':
            upper_court_balls_frames = GT_BOUNCES_FRAMES[1::2]    
            lower_court_balls_frames = GT_BOUNCES_FRAMES[::2]
        else:
            upper_court_balls_frames = GT_BOUNCES_FRAMES[::2]    
            lower_court_balls_frames = GT_BOUNCES_FRAMES[1::2]
        print(f"Upper Court Bounce Frames Corrected : {upper_court_balls_frames}")
        print(f"Lower Court Bounce Frames Corrected : {lower_court_balls_frames}")    
        
        if SELECTED_PLAYER == 'Upper':
            player_balls_frames = lower_court_balls_frames
        else:
            player_balls_frames = upper_court_balls_frames
        

            
        
        # If ground truth is available for this video, use it; otherwise use detected values
        if GT_RACKET_HITS_FRAMES and GT_BOUNCES_FRAMES:
            ball_landing_frames_stats = GT_BOUNCES_FRAMES.copy()
            ball_shots_frames_stats = GT_RACKET_HITS_FRAMES.copy()
        else:
            # Use detected values when ground truth is not available
            ball_landing_frames_stats = ball_landing_frames.copy()
            ball_shots_frames_stats = ball_shots_frames.copy()
    else:
        # When not in DEBUG mode, always use detected values
        ball_landing_frames_stats = ball_landing_frames.copy()
        ball_shots_frames_stats = ball_shots_frames.copy()

    print(f"Selected player is {SELECTED_PLAYER} so the bounce used in ball heatmap are in the opposite court and are: {player_balls_frames}")

    # Get the racket hit frames for the player (only needed for probabilities or debugging)
    if COMPUTE_PROBABILITIES or DEBUG:
        ball_shots_frames_upper, ball_shots_frames_lower = filter_ball_shots_by_player(ball_shots_frames_stats, player_detections, ball_detections)
        print(f"Ball Shots Upper Player: {ball_shots_frames_upper}")
        print(f"Ball Shots Lower Player: {ball_shots_frames_lower}")
        
        # Check if both lists have elements before accessing them
        if len(ball_shots_frames_upper) > 0 and len(ball_shots_frames_lower) > 0:
            if ball_shots_frames_upper[0] < ball_shots_frames_lower[0]:
                serve_player = 'Upper'
            else:
                serve_player = 'Lower'
            print(f"The Player who serves is: {serve_player}")
        else:
            print(f"Warning: Cannot determine serving player - Upper shots: {len(ball_shots_frames_upper)}, Lower shots: {len(ball_shots_frames_lower)}")
            serve_player = 'Unknown'
        
        if GT_SERVE_PLAYER == 'Upper':
            ball_shots_frames_upper = GT_RACKET_HITS_FRAMES[::2]
            ball_shots_frames_lower = GT_RACKET_HITS_FRAMES[1::2]
        elif GT_SERVE_PLAYER == 'Lower':
            ball_shots_frames_upper = GT_RACKET_HITS_FRAMES[1::2]
            ball_shots_frames_lower = GT_RACKET_HITS_FRAMES[::2]
            
        print(f"Corrected Ball Shots Upper Player: {ball_shots_frames_upper}")
        print(f"Corrected Ball Shots Lower Player: {ball_shots_frames_lower}")
        
        # Check if both lists have elements before accessing them
        if len(ball_shots_frames_upper) > 0 and len(ball_shots_frames_lower) > 0:
            if ball_shots_frames_upper[0] < ball_shots_frames_lower[0]:
                serve_player = 'Upper'
            else:
                serve_player = 'Lower'
            print(f"The Corrected Player who serves is: {serve_player}")
        else:
            print(f"Warning: Cannot determine serving player - Upper shots: {len(ball_shots_frames_upper)}, Lower shots: {len(ball_shots_frames_lower)}")
            serve_player = 'Unknown'

        # Print the ground truth values for the racket hits and ball landings for debugging purposes
        if DEBUG and len(ball_shots_frames_stats) > 0 and len(GT_BOUNCES_FRAMES) > 0:
            print(f"Ground Truth Racket Hit frames : {ball_shots_frames_stats}")
            print(f"Ground Truth Bounce Frames : {GT_BOUNCES_FRAMES}")

    ######## MATCH STATS ########

    player_stats_data = [{
        'frame_num': 0,

        # player 1 stats
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_min_shot_speed': float('inf'),
        'player_1_max_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,
        'player_1_hits_counter': 0,
        'player_1_distance_covered': 0,

        #player 2 stats
        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_min_shot_speed': float('inf'),
        'player_2_max_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
        'player_2_hits_counter': 0,
        'player_2_distance_covered': 0,
    } ]

    # Track player positions for calculating total distance
    player_positions_history = {1: [], 2: []}
    
    # Ensure both lists have the same length to avoid index errors
    min_length = min(len(ball_shots_frames_stats), len(ball_landing_frames_stats))
    if len(ball_shots_frames_stats) != len(ball_landing_frames_stats):
        print(f"Warning: Mismatch in lengths - ball_shots_frames_stats: {len(ball_shots_frames_stats)}, ball_landing_frames_stats: {len(ball_landing_frames_stats)}")
        print(f"Using minimum length: {min_length}")
    
    # Loop over all ball shots except the last one since it doesn't have an answer shot.
    for ball_shot_ind in range(min_length):
        start_frame = ball_shots_frames_stats[ball_shot_ind]               # Starting frame of the ball shot
        end_frame = ball_landing_frames_stats[ball_shot_ind]               # Ending frame of the ball shot
        ball_shot_time_in_seconds = (end_frame - start_frame) / fps        # Time taken by the ball to travel from the player to the opponent

        # Get distance covered by the ball
        distance_covered_by_ball_pixels = euclidean_distance(ball_mini_court_detections[start_frame][1],
                                                             ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels, 
                                                                           info.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court()) # Distance covered by the ball in meters

        # Speed of the ball shot in km/h
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6  # 3.6 to convert m/s to km/h

        # Player who shot the ball
        player_position = player_mini_court_detections[start_frame] # Player positions at the start of the ball shot
        player_shot_ball = min(player_position.keys(), key = lambda player_id: euclidean_distance(player_position[player_id], 
                                                                                                    ball_mini_court_detections[start_frame][1]))
        
        # Opponent player's speed
        opponent_player_id = 1 if player_shot_ball == 2 else 2  # Opponent player's ID

        # Track player distances between frames
        for player_id in [1, 2]:
            # Calculate total distance covered by this player between shots
            player_distance = 0
            for frame in range(start_frame, end_frame):
                if (player_id in player_mini_court_detections[frame] and 
                    player_id in player_mini_court_detections[frame+1]):
                    
                    dist_pixels = euclidean_distance(
                        player_mini_court_detections[frame][player_id],
                        player_mini_court_detections[frame+1][player_id]
                    )
                    dist_meters = convert_pixel_distance_to_meters(
                        dist_pixels,
                        info.DOUBLE_LINE_WIDTH,
                        mini_court.get_width_of_mini_court()
                    )
                    player_distance += dist_meters

            player_positions_history[player_id].append(player_distance)

        # Check if opponent player was detected in both frames
        if opponent_player_id in player_mini_court_detections[start_frame] and opponent_player_id in player_mini_court_detections[end_frame]:
            distance_covered_by_opponent_pixels = euclidean_distance(player_mini_court_detections[start_frame][opponent_player_id],
                                                                    player_mini_court_detections[end_frame][opponent_player_id]) 
            distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels, 
                                                                                info.DOUBLE_LINE_WIDTH,
                                                                                mini_court.get_width_of_mini_court())
            opponent_speed = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6  # 3.6 to convert m/s to km/h
        else:
            # Handle missing player detection
            distance_covered_by_opponent_meters = 0
            opponent_speed = 0



        # Update player stats
        current_player_stats = deepcopy(player_stats_data[-1]) # Copy of previous stats
        current_player_stats['frame_num'] = start_frame
        
        # Player who shot the ball stats
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        
        # Only update hits counter if we have the upper/lower frame data
        if COMPUTE_PROBABILITIES or DEBUG:
            if start_frame in ball_shots_frames_upper:
                # This shot was made by the upper player
                upper_player_id = player_position_to_id.get('Upper')
                current_player_stats[f'player_{upper_player_id}_hits_counter'] += 1
            elif start_frame in ball_shots_frames_lower:
                # This shot was made by the lower player
                lower_player_id = player_position_to_id.get('Lower')
                current_player_stats[f'player_{lower_player_id}_hits_counter'] += 1
            
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot
        
        # Update min and max shot speeds
        # Min
        current_player_stats[f'player_{player_shot_ball}_min_shot_speed'] = min(
            current_player_stats[f'player_{player_shot_ball}_min_shot_speed'], 
            speed_of_ball_shot
        )
        # Max
        current_player_stats[f'player_{player_shot_ball}_max_shot_speed'] = max(
            current_player_stats[f'player_{player_shot_ball}_max_shot_speed'], 
            speed_of_ball_shot
        )
        
        # Update distance covered
        if player_positions_history[player_shot_ball]:
            player_distance = player_positions_history[player_shot_ball][-1]
            current_player_stats[f'player_{player_shot_ball}_distance_covered'] += player_distance

        # Opponent player stats
        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += opponent_speed
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = opponent_speed
        
        # Update opponent distance covered
        if player_positions_history[opponent_player_id]:
            opponent_distance = player_positions_history[opponent_player_id][-1]
            current_player_stats[f'player_{opponent_player_id}_distance_covered'] += opponent_distance

        player_stats_data.append(current_player_stats)
    
    # Convert player stats data to a DataFrame
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frames_df = pd.DataFrame({'frame_num':list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frames_df, player_stats_data_df, on='frame_num', how='left') # Merge the player stats data with the frames data
    player_stats_data_df = player_stats_data_df.ffill() # Fill the missing values by replacing them with the previous value

    # Calculate average shot speed for each player
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / player_stats_data_df['player_2_number_of_shots']

    # Calculate average player speed for each player
    # We calculate the average player speed by dividing the total player speed by the number of shots of the opponent player
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / player_stats_data_df['player_1_number_of_shots']


    ######## DRAW OUTPUT ########
    
    # Draw Player Detection
    output_frames = player_tracker.draw_ellipse_bboxes(video_frames, player_detections, SELECTED_PLAYER, player_position_to_id)

    # Draw Ball Detection
    # Convert YOLO format to TrackNet format for write_track function
    #ball_track_converted = convert_yolo_to_tracknet_format(ball_detections_YOLO)
    output_frames = write_track(video_frames, ball_detections_tracknet)

    # Draw keypoints, according to the first frame
    all_keypoints = courtline_detector.detect_keypoints_for_all_frames(video_frames)
    output_frames = courtline_detector.draw_keypoints_on_video_dinamically(output_frames, all_keypoints)

    # Draw Mini Court
    if DRAW_MINI_COURT:
        # Draw the mini court on the video frames
        output_frames = mini_court.draw_mini_court(output_frames)

        if COMPUTE_PROBABILITIES:
            # Draw the heatmaps on the mini court
            _, ball_landing_heatmaps = mini_court.draw_ball_landing_heatmap(output_frames, player_balls_frames, ball_mini_court_detections, ball_shots_frames_upper, ball_shots_frames_lower, selected_player=SELECTED_PLAYER)
            _, player_distance_heatmaps = mini_court.draw_player_distance_heatmap(output_frames, player_mini_court_detections, selected_player=SELECTED_PLAYER)
            
            # Compute score prediction and probability for each ball shot
            output_frames = mini_court.draw_score_heatmap(output_frames, ball_landing_heatmaps, player_distance_heatmaps)
            
            # Draw the trajectories of the ball on the mini court
            output_frames = mini_court.draw_shot_trajectories(output_frames, player_mini_court_detections, ball_mini_court_detections, ball_landing_frames_stats, ball_shots_frames_stats, serve_player=serve_player)     
        else:
            # Draw the ball positions on the mini court
            output_frames = mini_court.draw_points_on_mini_court(output_frames, ball_mini_court_detections, object_type='ball')

        # Draw the player positions on the mini court
        output_frames = mini_court.draw_points_on_mini_court(output_frames, player_mini_court_detections, object_type='player')

    # Draw player stats box
    if DRAW_STATS_BOX:
        output_frames = draw_player_stats(output_frames, player_stats_data_df, 
                                            selected_player=SELECTED_PLAYER,
                                            player_mapping=player_position_to_id)

    # Draw Frames Number, Racket Hits and Ball Landings for debugging purposes
    
    if DEBUG and DRAW_DEBUG_WINDOW:
        output_frames = draw_debug_window(output_frames)
        output_frames = draw_frames_number(output_frames)
        output_frames = draw_racket_hits(output_frames, ball_shots_frames_stats)
        output_frames = draw_ball_landings(output_frames, ball_landing_frames, GT_BOUNCES_FRAMES, ball_detections_tracknet)
    
    # Save video
    save_video(output_frames, OUTPUT_VIDEO_PATH, fps)
    print(f"Output video saved to: {OUTPUT_VIDEO_PATH}")


if __name__ == '__main__':
    main()

