import os
import sys
import pathlib
import pandas as pd

sys.path.append("../..")
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
cwd = os.getcwd()
import numpy as np

sys.path.append(cwd)

from eyetrackpy.data_generator.fixations_predictor.models.generate_fixations_predictor import (
    CreateFixationsPredictorModel,
)
from eyetrackpy.data_generator.models.fixations_aligner import FixationsAligner

from eyetrackpy.data_processor.models.eye_tracking_data_image import (
    EyeTrackingDataImage,
)
from eyetrackpy.data_processor.models.eye_tracking_data_simple import (
    EyeTrackingDataUserSet,
) 


def save_word_fixations_syn_trial(folder_name, trial, words_fix_trial):
    """
    Save coordinates of words of trial in csv file"""
    # create_csv with words_fix_trial
    words_fix_trial.to_csv(
        folder_name + "/" + "/word_gaze_synthetic_" + str(trial) + ".csv",
        sep=";",
    )
    return True


class SyntheticGazeGenerator:
    """
    A class to generate synthetic gaze data for eye tracking experiments.
    """
    
    def __init__(self, version=2, remap=False):
        """
        Initialize the SyntheticGazeGenerator.
        
        Args:
            version (int): Version of the fixations predictor model
            remap (bool): Whether to remap the model
        """
        self.create_fixations = CreateFixationsPredictorModel(version=version, remap=remap)
        self.datauserset = EyeTrackingDataUserSet()
        self.fixations_aligner = FixationsAligner()
        
    def generate_synthetic_gaze_for_user(self, user_id, input_path, output_path):
        """
        Generate synthetic gaze data for a specific user.
        
        Args:
            user_id (str): The user ID
            input_path (str): Path to the input data folder
            output_path (str): Path to save the synthetic data
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            folder = input_path + "participant_" + str(user_id) + '_' + str(user_id) + '/session_1/vertices/'
            folder_syn = output_path + "participant_" + str(user_id) + '_' + str(user_id) + '/session_1/vertices/'
            
            files = self.datauserset.search_word_coor_fixations_files(folder)
            
            for trial, file in files.items():
                coordinates_data = self.datauserset._read_coor_trial(file)
                list_words_first = list(coordinates_data.text)
                list_words_first = [str(x) for x in list_words_first]
                text = " ".join(list_words_first)

                (
                    fixations,
                    fixations_attention_mask,
                    _,
                    _,
                    text_tokenized_fix,
                    sentences,
                ) = self.create_fixations.FP_model._compute_mapped_fixations(sentences=text)
                
                # align to the words of the original text
                features = np.transpose(fixations.detach().cpu().numpy()).squeeze()
                list_words_first = [x.strip().lower() for x in list_words_first]
                features_mapped = self.fixations_aligner.map_features_from_words_to_words(
                    list_words_first=list_words_first,
                    text=text,
                    text_tokenized=text_tokenized_fix,
                    features=features,
                    mode="max",
                )
                
                name_features = [
                    "fix_number",
                    "first_fix_duration",
                    "GPT",
                    "fix_duration",
                    "fixProp",
                ]
                features_mapped = dict(zip(name_features, features_mapped))
                features_mapped["text"] = list_words_first
                features_mapped = pd.DataFrame(features_mapped)
                
                # check if folder_syn exists
                if not os.path.exists(folder_syn):
                    os.makedirs(folder_syn)
                    
                save_word_fixations_syn_trial(folder_syn, trial, features_mapped)
                
            return True
            
        except Exception as e:
            print(f"Error generating synthetic gaze for user {user_id}: {str(e)}")
            return False
    
    def generate_synthetic_gaze_for_all_users(self, user_ids, input_path, output_path):
        """
        Generate synthetic gaze data for multiple users.
        
        Args:
            user_ids (list): List of user IDs
            input_path (str): Path to the input data folder
            output_path (str): Path to save the synthetic data
            
        Returns:
            dict: Dictionary with user_id as key and success status as value
        """
        results = {}
        
        for user_id in user_ids:
            print(f"Processing user {user_id}...")
            success = self.generate_synthetic_gaze_for_user(user_id, input_path, output_path)
            results[user_id] = success
            
        return results


if __name__ == "__main__":
    # Initialize the synthetic gaze generator
    gaze_generator = SyntheticGazeGenerator(version=2, remap=False)
    
    # Define paths and user list
    input_path = "users/"
    output_path = "users_synthetic/"
    users = ['1','2','3','4','5','6','7','8','9','10','11','13','14','15','16']
    
    # Generate synthetic gaze data for all users
    print("Starting synthetic gaze generation...")
    results = gaze_generator.generate_synthetic_gaze_for_all_users(users, input_path, output_path)
    
    # Print results summary
    successful_users = [user for user, success in results.items() if success]
    failed_users = [user for user, success in results.items() if not success]
    
    print(f"\nGeneration completed!")
    print(f"Successfully processed {len(successful_users)} users: {successful_users}")
    if failed_users:
        print(f"Failed to process {len(failed_users)} users: {failed_users}")