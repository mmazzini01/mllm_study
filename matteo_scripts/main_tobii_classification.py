import sys
import os
import argparse
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from tobiipy.tobii_classifier import Tobii_Classifier

# Add the parent directory to sys.path to allow importing tobiipy


def main():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Tobii eye tracking data classification')
    
    # Add arguments
    parser.add_argument('input_file', type=str, help='Path to input CSV file containing raw eye tracking data')
    
    # Blink detection parameters
    parser.add_argument('--blink_detection', type=str, default='true', choices=['true', 'false'],
                      help='Enable or disable blink detection (default: true)')
    parser.add_argument('--fs', type=int, default=60,
                      help='Sampling rate of the eye tracker in Hz (default: 60)')
    parser.add_argument('--gap_dur', type=int, default=40,
                      help='Max time gap between missing data to be considered a single blink in ms (default: 40)')
    parser.add_argument('--min_amplitude', type=float, default=0.1,
                      help='Minimum blink amplitude as % of full openness (default: 0.1)')
    parser.add_argument('--min_separation', type=int, default=100,
                      help='Minimum duration between two blinks in ms (default: 100)')
    parser.add_argument('--debug', type=str, default='false', choices=['true', 'false'],
                      help='Print debug info during blink detection (default: false)')
    parser.add_argument('--filter_length', type=int, default=25,
                      help='Filter length used for blink smoothing in ms (default: 25)')
    parser.add_argument('--width_of_blink', type=int, default=15,
                      help='Width of the blink detection window in ms (default: 15)')
    parser.add_argument('--min_blink_dur', type=int, default=30,
                      help='Minimum duration to classify an event as a blink in ms (default: 30)')
    
    # Fixation detection parameters
    parser.add_argument('--noise_reduction', type=str, default='median', choices=['median', 'average', 'false'],
                      help='Type of smoothing applied to gaze data (default: median)')
    parser.add_argument('--noise_reduction_window', type=int, default=3,
                      help='Window size for gaze smoothing in samples (default: 3)')
    parser.add_argument('--velocity_threshold', type=int, default=30,
                      help='Velocity threshold to detect fixations in degrees/second (default: 30)')
    parser.add_argument('--merge_fixations', type=str, default='true', choices=['true', 'false'],
                      help='Whether to merge consecutive fixations (default: true)')
    parser.add_argument('--merge_max_time', type=int, default=75,
                      help='Maximum time allowed between fixations to merge them in ms (default: 75)')
    parser.add_argument('--discard_short', type=str, default='true', choices=['true', 'false'],
                      help='Whether to discard fixations below the duration threshold (default: true)')
    parser.add_argument('--min_fixation_duration', type=int, default=60,
                      help='Minimum allowed fixation duration in ms (default: 60)')

    # Parse arguments
    args = parser.parse_args()
    
    # Convert string boolean arguments to actual booleans
    blink_detection = args.blink_detection.lower() == 'true'
    debug = args.debug.lower() == 'true'
    discard_short = args.discard_short.lower() == 'true'
    merge_fixations = args.merge_fixations.lower() == 'true'
    noise_reduction = args.noise_reduction if args.noise_reduction != 'false' else False

    # Run classification
    Tobii_Classifier(
        file_path=args.input_file,
        blink_detection=blink_detection,
        Fs=args.fs,
        gap_dur=args.gap_dur,
        min_amplitude=args.min_amplitude,
        min_separation=args.min_separation,
        debug=debug,
        filter_length=args.filter_length,
        width_of_blink=args.width_of_blink,
        min_blink_dur=args.min_blink_dur,
        noise_reduction=noise_reduction,
        noise_reduction_window=args.noise_reduction_window,
        velocity_threshold=args.velocity_threshold,
        merge_fixations=merge_fixations,
        merge_max_time=args.merge_max_time,
        discard_short_fixation=discard_short,
        min_fixation_duration=args.min_fixation_duration
    )

if __name__ == '__main__':
    main()