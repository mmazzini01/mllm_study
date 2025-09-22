import pathlib
import sys
import copy

sys.path.append("../..")
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(
    pathlib.Path(__file__)
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
    .parent.resolve()
)
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve().parent.resolve())
sys.path.append(path)
path = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(path)
from eye_tracking_data import EyeTrackingData
import os

import pytesseract
from pytesseract import Output
import cv2
import numpy as np
import pandas as pd
import re
import traceback


class EyeTrackingDataImage(EyeTrackingData):
    def __init__(
        self,
        user,
        session,
        user_set=0,
        trial=None,
        show=False,
        x_screen=1280,
        y_screen=1024,
        path=None,
    ):
        super().__init__(
            user=user,
            session=session,
            user_set=user_set,
            trial=trial,
            show=show,
            path=path,
        )
        self.x_screen = x_screen
        self.y_screen = y_screen
        # dataframe with general data of all trials
        self.data = self._read_general_file()
        # dataframe with row fixations of all trial
        self.fixations_raw = self._read_fixations_file()
        self.trials = [
            float(x)
            for x in self.fixations_raw["USER"].unique()
            if len(str(x).split(".")) > 1
        ]
        self.coordinates = {}
        #prompt_trial = list({int(t) for t in self.trials})
        #self.trials.extend(prompt_trial)

        # self.words_fix : dic with a dataframe for each trial of fixations assigned to words
        # self.words_fix_trial: dataframe with words and fixations assigned for one trial
        # self.coordinates: dic with a dataframe for each trial with coordinates of words
        # self.coordinates_trial: dataframe with coordinates of words of one trial
        # self.fixations_trial: dataframe with fixations of one trial
        if  self.fixations_raw is None: #or self.data is None 
            raise ValueError("No data found for this user and session")

    def _read_image_trial(self, trial):
        """
        Read image trial and return dataframe with coordinates of words
        """
        coordinates_trial = {}
        #for file in os.listdir(self.folder_name + "/" + "vertices"):
        for file in os.listdir(self.path+ "/" + "vertices"):
            if "_" + str(trial) in file and ".png" in file and "fixations" not in file:
                #img = cv2.imread(self.folder_name + "/" + "vertices" + "/" + file)
                img = cv2.imread(self.path + "/" + "vertices" + "/" + file)
                img = cv2.resize(img, (self.x_screen, self.y_screen))
                d = pytesseract.image_to_data(
                    img, output_type=Output.DICT, config="--psm 6"
                )
                n_boxes = len(d["level"])
                counter = 0
                for i in range(n_boxes):
                    (x, y, w, h) = (
                        d["left"][i],
                        d["top"][i],
                        d["width"][i],
                        d["height"][i],
                    )
                    if d["text"][i] != "":
                        # first version with this join with a ',' of the tokenize box
                        # I removed it becouse a lot of times I end with things like "outside,?" instead of  original "outside?"
                        # text = ','.join([str(x) for x in self.tokenize_sentence(d['text'][i])])
                        text = d["text"][i]
                        # print(text)
                        # try:
                        coordinates_trial[counter] = [counter, text, x, y, x + w, y + h]
                        # except Exception as e:
                        #     print(e)

                        counter += 1
                # try:
                coordinates_trial = pd.DataFrame.from_dict(
                    coordinates_trial,
                    orient="index",
                    columns=["number", "text", "x1", "y1", "x2", "y2"],
                )
                # except Exception as e:
                #     print(e)
        return coordinates_trial
    
    def _read_image_trial_prompt(self, trial, text_ratio=0.25):
        coordinates_trial = {}
        #for file in os.listdir(self.folder_name + "/" + "vertices"):
        for file in os.listdir(self.path + "/" + "vertices"):
            if re.fullmatch(f".*_prompt_{trial}.png", file) and "fixations" not in file:

                #img = cv2.imread(self.folder_name + "/" + "vertices" + "/" + file)
                img = cv2.imread(self.path + "/" + "vertices" + "/" + file)
                img = cv2.resize(img, (self.x_screen, self.y_screen))
                h, w, _ = img.shape
                y_start = int(h *(1-text_ratio)) # 20% from the bottom
                y_end = h
                x_start = 0
                x_end = w
                cropped = img[y_start:y_end, x_start:x_end]
                d = pytesseract.image_to_data(
                    cropped, output_type=Output.DICT, config="--psm 6"
                )
                n_boxes = len(d["level"])
                counter = 0
                for i in range(n_boxes):
                    (x, y, w, h) = (
                        d["left"][i],
                        d["top"][i] + y_start,
                        d["width"][i],
                        d["height"][i],
                    )
                    if d["text"][i] != "":
                        # first version with this join with a ',' of the tokenize box
                        # I removed it becouse a lot of times I end with things like "outside,?" instead of  original "outside?"
                        # text = ','.join([str(x) for x in self.tokenize_sentence(d['text'][i])])
                        text = d["text"][i]
                        # print(text)
                        # try:
                        coordinates_trial[counter] = [counter, text, x, y, x + w, y + h]
                        # except Exception as e:
                        #     print(e)

                        counter += 1
                # try:
                coordinates_trial = pd.DataFrame.from_dict(
                    coordinates_trial,
                    orient="index",
                    columns=["number", "text", "x1", "y1", "x2", "y2"],
                )
                # except Exception as e:
                #     print(e)
        return coordinates_trial


    def _preprocess_fixations_trial(self, fixations_trial):
        if "FPOGX" in fixations_trial.columns:
            # better to use: BPOGX, BPOGY
            fixations_trial = fixations_trial.rename(
                columns={"FPOGX": "x", "FPOGY": "y", "FPOGD": "duration", "FPOGID": "ID"}
            )
            fixations_trial["pupil_r"] = round(
                fixations_trial["RPD"] * fixations_trial["RPS"], 4
            )
            fixations_trial["pupil_l"] = round(
                fixations_trial["LPD"] * fixations_trial["LPS"], 4
            )
            fixations_trial = fixations_trial[["ID", "x", "y", "duration", "pupil_r", "pupil_l"]]
        else:
            fixations_trial = fixations_trial[["ID", "x", "y", "duration", "pupil_r", "pupil_l"]]
        return fixations_trial
    
    def _scale_fixations_trial(self, fixations_trial):
        """
        Scale trial fixations to screen size
        """
        
        fixations_trial["x"] = fixations_trial["x"] * self.x_screen
        fixations_trial["y"] = fixations_trial["y"] * self.y_screen
        return fixations_trial


    def _get_fixations_trial(self, trial):
        """
        Get fixations of trial and scale to screen size
        """
        fixations_trial = self.fixations_raw[self.fixations_raw["USER"] == str(trial)]
        fixations_trial = self._preprocess_fixations_trial(fixations_trial)
        fixations_trial = self._scale_fixations_trial(fixations_trial)
        return fixations_trial
    
    def _get_fixations_trial_prompt(self, trial):
        """
        Get fixations of trial and scale to screen size
        """
        fixations_trial = self.fixations_raw[self.fixations_raw["USER"] == str(trial)]
        fixations_trial = self._preprocess_fixations_trial(fixations_trial)
        fixations_trial = self._scale_fixations_trial(fixations_trial)
        fixations_trial = fixations_trial[fixations_trial["y"] >= self.y_screen * 0.75]
        return fixations_trial

    def _compute_calibration_coordinates_trial(self, trial):
        """
        Compute calibration coordinates of trial with"""
        # obtain index position of first row os trial
        trial = self.fixations_raw[self.fixations_raw["USER"] == str(trial)]
        trial = trial.index[0]
        if "FPOGX" in self.fixations_raw.columns:
        # get FPOGX and FPOGY the previous row of trial in data which user in 'FIX_CROSS'
            calibrate_x = self.fixations_raw.loc[trial - 1]["FPOGX"] - 0.5
            calibrate_y = self.fixations_raw.loc[trial - 1]["FPOGY"] - 0.5
        else:
            calibrate_x = self.fixations_raw.loc[trial - 1]["x"] - 0.5
            calibrate_y = self.fixations_raw.loc[trial - 1]["y"] - 0.5
        # scale to screen size
        calibrate_x *= self.x_screen        
        calibrate_y *= self.y_screen
        return calibrate_x, calibrate_y


    
    def _compute_calibration_coordinates_trial_prompt(self, trial):
        """
        Compute calibration coordinates of trial using the most recent previous 'FIX_CROSS' row.
        """
        # Trova indice della prima riga del trial corrente
        trial_index = self.fixations_raw[self.fixations_raw["USER"] == str(trial)].index[0]

        # Cerca la prima riga con USER == 'FIX_CROSS' prima del trial_index
        fix_cross_rows = self.fixations_raw.loc[:trial_index - 1]
        fix_cross_index = fix_cross_rows[fix_cross_rows["USER"] == "FIX_CROS"].last_valid_index()

        if fix_cross_index is None:
            return None, None  

        # Calcola le coordinate di calibrazione
        if "FPOGX" in self.fixations_raw.columns:
            calibrate_x = self.fixations_raw.loc[fix_cross_index]["FPOGX"] - 0.5
            calibrate_y = self.fixations_raw.loc[fix_cross_index]["FPOGY"] - 0.5
        else:
            calibrate_x = self.fixations_raw.loc[fix_cross_index]["x"] - 0.5
            calibrate_y = self.fixations_raw.loc[fix_cross_index]["y"] - 0.5            
        calibrate_x *= self.x_screen
        calibrate_y *= self.y_screen

        return calibrate_x, calibrate_y


    def read_image_trials(self, trials):
        """
        Read image trials and return dictionary with coordinates of words of different trials
        """
        # check if trials is a list:
        if not isinstance(trials, list):
            trials = [trials]
        self.coordinates = {}
        for trial in trials:
            print(trial)
            self.coordinates[trial] = self._read_image_trial(trial)
        return self.coordinates

    def save_coordinates(self, trials, coordinates):
        """
        Save coordinates of words of different trials in csv files"""
        if not isinstance(trials, list):
            trials = [trials]
        for trial in trials:
            self.save_coordinates_trial(trial, coordinates[trial])

    def save_fixations(self, words_fix: dict, fixations_all: dict = None, info: list = None):
        """
        Save coordinates of words of different trials in csv files"""
        if info is not None:
            info = pd.DataFrame(info)
            info.to_csv(
                self.folder_name +  "/info" + ".csv",
                sep=";",
            )
        trials = list(words_fix.keys())
        for trial in trials:
            self.save_word_fixations_trial(trial, words_fix[trial])
            if fixations_all is not None:
                self.save_fixations_trial(trial, fixations_all[trial])

    def plot_image_trial(
        self, trial, fixations=False, coordinates=False, calibrate=False
    ):
        """
        Plot image trial with fixations and coordinates of words"""
        for file in os.listdir(self.folder_name + "/" + "vertices"):
            if "_" + str(trial) in file and ".png" in file:
                img = cv2.imread(self.folder_name + "/" + "vertices" + "/" + file)
                img = cv2.resize(img, (self.x_screen, self.y_screen))
                if coordinates:
                    self._plot_image_coordinates(img)
                if fixations:
                    self._plot_image_fixations(trial, img, calibrate)
                cv2.imshow("img", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        return True

    def _plot_image_coordinates(self, img, color=(0, 255, 0)):
        """
        coordinates of words on an image"""
        d = pytesseract.image_to_data(img, output_type=Output.DICT, config="--psm 6")
        n_boxes = len(d["level"])
        for i in range(n_boxes):
            (x, y, w, h) = (d["left"][i], d["top"][i], d["width"][i], d["height"][i])
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        return True

    def _plot_total_distance(self, img, fixations_trial, fixations_trial_cal=None, ids_removed=[], ids_removed_cal=[]):
        if isinstance(fixations_trial, pd.DataFrame):
            total_distance = round(fixations_trial["distance"].sum(), 2)
        else:
            total_distance = 0
        if isinstance(fixations_trial_cal, pd.DataFrame):
            total_distance_calibrate = round(fixations_trial_cal["distance"].sum(), 2)
        else:
            total_distance_calibrate = 0

        if len(ids_removed)>0:
            fixations_trial_notremoved = fixations_trial[~fixations_trial['ID'].isin(ids_removed)]
            total_distance_notremoved = round(fixations_trial_notremoved["distance"].sum(), 2)
        else:
            total_distance_notremoved = 0

        if len(ids_removed_cal)>0:
            fixations_trial_cal_notremoved = fixations_trial_cal[~fixations_trial_cal['ID'].isin(ids_removed_cal)]
            total_distance_calibrate_notremoved = round(fixations_trial_cal_notremoved["distance"].sum(), 2)
        else:
            total_distance_calibrate_notremoved = 0

        if total_distance > 0:
            text = f"Total dis.: {total_distance}"
            if total_distance_notremoved > 0:
                text += f" wihtout removed ({total_distance_notremoved})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            thickness = 1
            cv2.putText(
                img,
                text,
                (10, 30),
                font,
                fontScale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
        if total_distance_calibrate > 0:
            text = f"Total dis. cal.: {total_distance_calibrate}"
            if total_distance_calibrate_notremoved > 0:
                text += f" wihtout removed ({total_distance_calibrate_notremoved})"

            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            thickness = 1
            cv2.putText(
                img,
                text,
                (10, 50),
                font,
                fontScale,
                (255, 165, 0),
                thickness,
                cv2.LINE_AA,
            )

    def _plot_mean_distance(self, img, fixations_trial, fixations_trial_cal=None, ids_removed=[], ids_removed_cal=[]):
        if isinstance(fixations_trial, pd.DataFrame):
            mean_distance = round(fixations_trial["distance"].mean(), 2)
        else:
            mean_distance = 0
        if isinstance(fixations_trial_cal, pd.DataFrame):
            mean_distance_calibrate = round(fixations_trial_cal["distance"].mean(), 2)
        else:
            mean_distance_calibrate = 0

        if len(ids_removed)>0:
            fixations_trial_notremoved = fixations_trial[~fixations_trial['ID'].isin(ids_removed)]
            mean_distance_notremoved = round(fixations_trial_notremoved["distance"].mean(), 2)
        else:
            mean_distance_notremoved = 0

        if len(ids_removed_cal)>0:
            fixations_trial_cal_notremoved = fixations_trial_cal[~fixations_trial_cal['ID'].isin(ids_removed_cal)]
            mean_distance_calibrate_notremoved = round(fixations_trial_cal_notremoved["distance"].mean(), 2)
        else:
            mean_distance_calibrate_notremoved = 0


        if mean_distance > 0:
            text = f"mean dis.: {mean_distance}"
            if mean_distance_notremoved > 0:
                text += f" wihtout removed ({mean_distance_notremoved})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            thickness = 1
            cv2.putText(
                img,
                text,
                (10, 70),
                font,
                fontScale,
                (0, 255, 0),
                thickness,
                cv2.LINE_AA,
            )
        if mean_distance_calibrate > 0:
            text = f"mean dis. cal.: {mean_distance_calibrate}"
            if mean_distance_calibrate_notremoved > 0:
                text += f" wihtout removed ({mean_distance_calibrate_notremoved})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.7
            thickness = 1
            cv2.putText(
                img,
                text,
                (10, 90),
                font,
                fontScale,
                (255, 165, 0),
                thickness,
                cv2.LINE_AA,
            )
    def plot_image_trial_colors(
        self,
        trial,
        fixations_trial="",
        fixations_trial_cal="",
        words_fix_trial="",
        fix_diff="",
        words_diff="",
        fixations=False,
        coordinates=False,
        calibrate=False,
        save=False,
        ids_removed=[],
        ids_removed_cal=[],
        print_fix_distance=False,
    ):
        """
        Plot image trial with fixations and coordinates of words"""

        #for file in os.listdir(self.folder_name + "/" + "vertices"):
        for file in os.listdir(self.path + "/" + "vertices"):
            if "_" + str(trial) in file and ".png" in file and "fixations" not in file:
                #img = cv2.imread(self.folder_name + "/" + "vertices" + "/" + file)
                img = cv2.imread(self.path + "/" + "vertices" + "/" + file)
                img = cv2.resize(img, (self.x_screen, self.y_screen))
                # # Convert the image to RGBA (if not already)
                # if img.shape[2] == 3:  # If it's a 3-channel image
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                # # Create an overlay with a transparent background
                # overlay = img.copy()
                # alpha_value = 0.8  # Adjust this value for transparency (0: fully transparent, 1: no transparency)

                # # Add transparency to the background (non-text regions)
                # overlay[:, :, 3] = int(255 * alpha_value)  # Set alpha value

                # # Blend the image and the overlay
                # img = cv2.addWeighted(overlay, alpha_value, img, 1 - alpha_value, 0)

                if fixations:
                    self._plot_image_fixations_colors(
                        img,
                        trial,
                        fixations_trial,
                        ids_diff=fix_diff,
                        ids_removed = ids_removed,
                        calibrate=False,
                        color=(114,128,250),
                        print_fix_distance=print_fix_distance,
                    )

                    #if calibrate we move the coordinates or we print the fixations of the trial already calibrated if available
                    if calibrate is True:
                        if isinstance(fixations_trial_cal, pd.DataFrame):
                            self._plot_image_fixations_colors(
                                img,
                                trial,
                                fixations_trial_cal,
                                ids_diff = fix_diff,
                                ids_removed = ids_removed_cal,
                                calibrate=False,
                                color=(208,224,64),
                                color2=(0, 128, 255),
                                print_fix_distance=print_fix_distance,
                            )
                        else:
                            
                            self._plot_image_fixations_colors(
                                img,
                                trial,
                                fixations_trial,
                                ids_diff = fix_diff,
                                ids_removed = ids_removed,
                                calibrate=True,
                                color= (255, 165, 0),
                                print_fix_distance=False,
                            )

                #print each word box with coordenates
                if coordinates:
                    self._plot_image_coordinates_colors(
                        img, words_fix_trial, words_diff, color=(109,206,0)
                    )
                #print total distance
                self._plot_total_distance(img, fixations_trial, fixations_trial_cal, ids_removed, ids_removed_cal)
                #print mean distance
                self._plot_mean_distance(img, fixations_trial, fixations_trial_cal, ids_removed, ids_removed_cal)
                #save image
                if save:
                    cv2.imwrite(
                        self.folder_name
                        + "/"
                        + "vertices"
                        + "/"
                        + ".".join(file.split(".")[:-1])
                        + "_fixations_pruebaaa."
                        + file.split(".")[-1],
                        img,
                    )
                else:
                    cv2.imshow("img", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        return True
    
    def plot_image_trial_prompt_colors(
        self,
        trial,
        fixations_trial="",
        fixations_trial_cal="",
        words_fix_trial="",
        fix_diff="",
        words_diff="",
        fixations=False,
        coordinates=False,
        calibrate=False,
        save=False,
        ids_removed=[],
        ids_removed_cal=[],
        print_fix_distance=False,
    ):
        """
        Plot image trial with fixations and coordinates of words"""

        #for file in os.listdir(self.folder_name + "/" + "vertices"):
        for file in os.listdir(self.path + "/" + "vertices"):
            if re.fullmatch(f".*_prompt_{trial}.png", file) and "fixations" not in file:
                #img = cv2.imread(self.folder_name + "/" + "vertices" + "/" + file)
                img = cv2.imread(self.path + "/" + "vertices" + "/" + file)
                img = cv2.resize(img, (self.x_screen, self.y_screen))
                # # Convert the image to RGBA (if not already)
                # if img.shape[2] == 3:  # If it's a 3-channel image
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

                # # Create an overlay with a transparent background
                # overlay = img.copy()
                # alpha_value = 0.8  # Adjust this value for transparency (0: fully transparent, 1: no transparency)

                # # Add transparency to the background (non-text regions)
                # overlay[:, :, 3] = int(255 * alpha_value)  # Set alpha value

                # # Blend the image and the overlay
                # img = cv2.addWeighted(overlay, alpha_value, img, 1 - alpha_value, 0)

                if fixations:
                    self._plot_image_fixations_colors(
                        img,
                        trial,
                        fixations_trial,
                        ids_diff=fix_diff,
                        ids_removed = ids_removed,
                        calibrate=False,
                        color=(114,128,250),
                        print_fix_distance=print_fix_distance,
                    )

                    #if calibrate we move the coordinates or we print the fixations of the trial already calibrated if available
                    if calibrate is True:
                        if isinstance(fixations_trial_cal, pd.DataFrame):
                            self._plot_image_fixations_colors(
                                img,
                                trial,
                                fixations_trial_cal,
                                ids_diff = fix_diff,
                                ids_removed = ids_removed_cal,
                                calibrate=False,
                                color=(208,224,64),
                                color2=(0, 128, 255),
                                print_fix_distance=print_fix_distance,
                            )
                        else:
                            
                            self._plot_image_fixations_colors(
                                img,
                                trial,
                                fixations_trial,
                                ids_diff = fix_diff,
                                ids_removed = ids_removed,
                                calibrate=True,
                                color= (255, 165, 0),
                                print_fix_distance=False,
                            )

                #print each word box with coordenates
                if coordinates:
                    self._plot_image_coordinates_colors(
                        img, words_fix_trial, words_diff, color=(109,206,0)
                    )
                #print total distance
                self._plot_total_distance(img, fixations_trial, fixations_trial_cal, ids_removed, ids_removed_cal)
                #print mean distance
                self._plot_mean_distance(img, fixations_trial, fixations_trial_cal, ids_removed, ids_removed_cal)
                #save image
                if save:
                    cv2.imwrite(
                        self.folder_name
                        + "/"
                        + "vertices"
                        + "/"
                        + ".".join(file.split(".")[:-1])
                        + "_fixations_pruebaaa."
                        + file.split(".")[-1],
                        img,
                    )
                else:
                    cv2.imshow("img", img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
        return True

    def _plot_image_coordinates_colors(
        self, img, words_fix_trial, ids_diff=[], color=(192, 150, 128), color2=(0, 0, 255)
    ):
        # check if trials is a list:
        if not isinstance(ids_diff, list):
            ids_diff = [ids_diff]
        for idx, row in words_fix_trial.iterrows():
            if idx in ids_diff:
                cv2.rectangle(
                    img, (row["x1"], row["y1"]), (row["x2"], row["y2"]), color2, 2
                )
            else:
                cv2.rectangle(
                    img, (row["x1"], row["y1"]), (row["x2"], row["y2"]), color, 2
                )

        return True

    def _plot_image_fixations_colors(
        self,
        img,
        trial,
        fixations_trial,
        ids_diff=[],
        ids_removed=[],
        calibrate=False,
        color=(192, 228, 255),
        color1=(0, 0, 255),
        color2=(255, 128, 255),
        print_fix_distance=False,
    ):
        """
        Plot image trial with fixations of trial"""
        # check if trials is a list:
        font = cv2.FONT_HERSHEY_SIMPLEX
        if print_fix_distance:
            fontScale = 0.6
            thickness = 1
        else:
            fontScale = 1
            thickness = 1
        fixations_trial = copy.deepcopy(fixations_trial)
        if not isinstance(ids_diff, list):
            ids_diff = [ids_diff]
        if not isinstance(ids_removed, list):
            ids_removed = [ids_removed]
        if calibrate:
            calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial(
                trial
            )
            fixations_trial["x"] = fixations_trial["x"] - calibrate_x
            fixations_trial["y"] = fixations_trial["y"] - calibrate_y
        counter = 1
        for _, row in fixations_trial.iterrows():
            if pd.isna(row["x"]) or pd.isna(row["y"]):
                continue  # Salta la fissazione con coordinate non valide
            x = int(round(row["x"], 0))
            y = int(round(row["y"], 0))
            if print_fix_distance:
                value = str(round(row["distance"], 1))
            else:
                value = str(counter)
            if row["ID"] in ids_removed:
                cv2.putText(
                    img,
                    value,
                    (x, y),
                    font,
                    fontScale,
                    color2,
                    thickness,
                    cv2.LINE_AA,
                )
            elif row["ID"] in ids_diff:
                # cv2.circle(img, (x, y), 5, color2, -1)
                cv2.putText(
                    img,
                    value,
                    (x, y),
                    font,
                    fontScale,
                    color1,
                    thickness,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    img,
                    value,
                    (x, y),
                    font,
                    fontScale,
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
                # cv2.circle(img, (x, y), 5, color, -1)
            counter += 1
        return True

    def _plot_image_fixations(self, trial, img, calibrate=False, color=(0, 0, 255)):
        """
        Plot image trial with fixations of trial"""
        fixations_trial = self._get_fixations_trial(trial)
        if calibrate:
            calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial(
                trial
            )
            fixations_trial["x"] = fixations_trial["x"] - calibrate_x
            fixations_trial["y"] = fixations_trial["y"] - calibrate_y
        for _, row in fixations_trial.iterrows():
            x = int(round(row["x"], 0))
            y = int(round(row["y"], 0))
            cv2.circle(img, (x, y), 5, color, -1)

        return True

    # def plot_image_fixations(self, trial, calibrate=False, color = (0, 0, 255)):
    #     for file in os.listdir(self.folder_name + '/' + 'vertices'):
    #         if "_" + str(trial) in file and '.png' in file:
    #             img = cv2.imread(self.folder_name + '/' + 'vertices' + '/' + file)
    #             img = cv2.resize(img, (self.x_screen, self.y_screen))
    #             fixations_trial = self._get_fixations_trial(trial)
    #             if calibrate:
    #                 calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial(trial)
    #                 fixations_trial['x'] = fixations_trial['x'] - calibrate_x
    #                 fixations_trial['y'] = fixations_trial['y'] - calibrate_y
    #             for _, row in fixations_trial.iterrows():
    #                 x = int(round(row['x'], 0))
    #                 y = int(round(row['y'], 0))
    #                 cv2.circle(img, (x, y), 5, color, -1)
    #             cv2.imshow('img', img)
    #             cv2.waitKey(0)
    #             cv2.destroyAllWindows()
    #     return True

    def save_coordinates_trial(self, trial, coordinates_trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with data_trial
        coordinates_trial.to_csv(
            self.folder_name
            + "/"
            + "vertices"
            + "/word_cor_image"
            + str(trial)
            + ".csv",
            sep=";",
        )
        return True

    def save_word_fixations_trial(self, trial, words_fix_trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        words_fix_trial.to_csv(
            self.folder_name
            + "/"
            + "vertices"
            + "/word_cor_image_fixations_"
            + str(trial)
            + ".csv",
            sep=";",
        )
        return True

    def load_words_fixations_trial(self, trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        try:
            words_fix_trial = pd.read_csv(
                self.folder_name
                + "/"
                + "vertices"
                + "/word_cor_image_fixations_"
                + str(trial)
                + ".csv",
                sep=";",
            )
            return words_fix_trial
        except:
            return None

    def load_fixations_trial(self, trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        try:
            words_fix_trial = pd.read_csv(
                self.folder_name
                + "/"
                + "vertices"
                + "/fixations_assigned_"
                + str(trial)
                + ".csv",
                sep=";",
            )
            return words_fix_trial
        except:
            return None

    def compute_entropy_trial(
        self, trial, columns=["fix_duration", "first_fix_duration", "fix_number"]
    ):
        """
        general_features_plus_entrophy : dict with this information:
            - number_words_mean: Number of words in the text for this trial
            - first_fix_duration_mean: Mean duration of the first fixation in each word in this text
            - fix_duration_mean: Mean duration of all fixations in each word in this text
            - fix_number_mean: Mean number of fixations in each word in this text
            - pupil_mean: Mean pupil size in each word (mean of the two eyes) in this text
            - fixations_regressions_mean: Percentages of fixations that are regressions in this text (A regression is when fixations + 1 is in a word before the word of a fixations)
            - trial: trail tag
            - fix_duration_increase: Percentage of words with more fixation duration in comparisom with previous world
            - fix_duration_decrease: Percentage of words with less fixation duration in comparisom with previous world
            - first_fix_duration_increase: Percentage of words with more first fixation duration in comparisom with previous world
            - first_fix_duration_decrease: Percentage of words with less first fixation duration in comparisom with previous world
            - fix_number_increase: Percentage of words with more fixations in comparisom with previous world
            - fix_number_decrease: Percentage of words with less fixations in comparisom with previous world
            - user: user tag
            - session: session tag
        """
        words_fix_trial = self.load_words_fixations_trial(trial)
        fixations_trial = self.load_fixations_trial(trial)
        '''
        if words_fix_trial is None or fixations_trial is None:
            fixations_trial, words_fix_trial, total_distance,*_ = (
                self.asign_fixations_words_trial(trial)
            )
        '''
        if words_fix_trial is None or fixations_trial is None:
            print(f"[SKIP ENTROPY] Trial {trial} non ha CSV (scartato in fase di assegnazione).")
            return {
                "trial": trial,
                "user": self.user,
                "session": self.session,
                "discarted": True
            }

        # 3. Verifica integrità minima dei dati
        if words_fix_trial.empty or fixations_trial.empty or "word_number" not in fixations_trial.columns:
            print(f"[SKIP ENTROPY] Trial {trial} ha dati incompleti o corrotti.")
            return {
                "trial": trial,
                "user": self.user,
                "session": self.session,
                "discarted": True
            }
        general_features_plus_entrophy = self.compute_general_features_trial(
            fixations_trial, words_fix_trial
        )

        general_features_plus_entrophy["trial"] = trial
        # Calculate the difference and the increase in the fixations and the number
        df_diff = words_fix_trial[columns].diff()
        for col in columns:
            general_features_plus_entrophy[f"{col}_increase"] = round(
                (df_diff[col] > 0).astype(int).sum() / len(df_diff), 4
            )
            general_features_plus_entrophy[f"{col}_decrease"] = round(
                (df_diff[col] <= 0).astype(int).sum() / len(df_diff), 4
            )
        return general_features_plus_entrophy

        # compute diff

    def compute_entropy_all(self, trials=None):
        if trials is None:
            trials = [
                float(x)
                for x in self.fixations_raw["USER"].unique()
                if len(str(x).split(".")) > 1
            ]
        fixations_entropy = []
        int_trials = []
        for trial in trials:
            if int(trial) not in int_trials:
                int_trials.append(int(trial))
            fixations_diff = self.compute_entropy_trial(trial)
            fixations_diff["user"] = self.user
            fixations_diff["session"] = self.session
            fixations_entropy.append(fixations_diff)
        for int_trial in int_trials:
            fixations_diff = self.compute_entropy_trial(int_trial)
            fixations_diff["user"] = self.user
            fixations_diff["session"] = self.session
            fixations_entropy.append(fixations_diff)
        
        return fixations_entropy

    def save_fixations_trial(self, trial, fixations_trial):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        fixations_trial.to_csv(
            self.folder_name
            + "/"
            + "vertices"
            + "/fixations_assigned_"
            + str(trial)
            + ".csv",
            sep=";",
        )
        return True

    def save_features(self, features):
        """
        Save coordinates of words of trial in csv file"""
        # create_csv with words_fix_trial
        features.to_csv(self.folder_name + "/general_features" + ".csv", sep=";")
        return True

    def euclidean_distance(self, row, x=0, y=0):
        """
        Calculate euclidean distance a word (with 4 coordinates) and 2 coordinates (x,y)
        """
        if x < max(row["x1"], row["x2"]) and x > min(row["x1"], row["x2"]):
            # word is in the midle of both
            x_word = x
        elif abs(row["x1"] - x) < abs(row["x2"] - x):
            x_word = row["x1"]
        else:
            x_word = row["x2"]

        if y < max(row["y1"], row["y2"]) and y > min(row["y1"], row["y2"]):
            # word is in the midle of both
            y_word = y
        elif abs(row["y1"] - y) < abs(row["y2"] - y):
            y_word = row["y1"]
        else:
            y_word = row["y2"]

        return np.sqrt((x - x_word) ** 2 + (y - y_word) ** 2)

    def _read_csv_trial(self, trial):
        data_trial = None
        for file in os.listdir(self.folder_name + "/" + "vertices"):
            if re.fullmatch(f"word_cor_image_fixations_{trial}.csv", file):
                data_trial = pd.read_csv(
                    self.folder_name + "/" + "vertices" + "/" + file,
                    sep=";",
                    index_col=0,
                )
                data_trial = data_trial.loc[
                    :, ~data_trial.columns.str.contains("^Unnamed")
                ]
        return data_trial
    
    def _read_csv_trial_prompt(self, trial):
        data_trial = None
        for file in os.listdir(self.folder_name + "/" + "vertices"):
            if re.fullmatch(f"word_cor_image_fixations_{trial}.csv", file):
                data_trial = pd.read_csv(
                    self.folder_name + "/" + "vertices" + "/" + file,
                    sep=";",
                    index_col=0,
                )
                data_trial = data_trial[data_trial["y1"] >= self.y_screen * 0.75]
                data_trial = data_trial.loc[
                    :, ~data_trial.columns.str.contains("^Unnamed")
                ]
        return data_trial

    def _search_lines_trial(self, data_trial):
        lines = {}
        try:
        # create column named y as the media of y1 and y2
            data_trial["y"] = data_trial[["y1", "y2"]].mean(axis=1)
        except Exception as e:
            print(e)
            print(data_trial)
        # create new column as the difference between y and y of the previous row
        data_trial["diff_y"] = data_trial["y"].diff().abs()
        # compute mean of diff_y
        mean_diff_y = data_trial["diff_y"].mean()
        # create a new column with True if diff_y is greater than mean_diff_y*5
        data_trial["line"] = data_trial["diff_y"] > mean_diff_y * 5
        lines = []
        counter = 1
        for _, row in data_trial.iterrows():
            if row["line"]:
                counter += 1
            lines.append(counter)
        data_trial["line"] = lines
        return data_trial

    def asign_fixations_words_all(self, check_calibration=False):
        """
        Asign fixations to words of all trials. change fixations_trial and data_trial dataframes to add this information
        """
        trials = [
            float(x)
            for x in self.fixations_raw["USER"].unique()
            if len(str(x).split(".")) > 1
        ]
        # check if words_fix is already computed
        if not hasattr(self, "words_fix"):
            words_fix = {}
        if not hasattr(self, "fixations"):
            fixations = {}
        for trial in trials:
            print(trial)

            fixations_trial, words_fix_trial, total_distance = (
                self.asign_fixations_words_trial(trial, check_calibration=True)
            )

            words_fix[trial] = words_fix_trial
            fixations[trial] = fixations_trial
        return fixations, words_fix
    

    def asign_fixations_process_words_all(self):
        """
        Asigna fixations alle parole per tutti i trial float + gestisce i prompt interi (1, 2, 3...) separatamente.
        """
        trials = [
            float(x)
            for x in self.fixations_raw["USER"].unique()
            if len(str(x).split(".")) > 1
        ]

        words_fix, fixations, info = {}, {}, []
        int_trials = []

        for trial in trials:
            if int(trial) not in int_trials:
                int_trials.append(int(trial))
            print(f"[INFO] Processing trial {trial}")
            try:
                fixations_trial, words_fix_trial, total_distance, info_trial = (
                    self.asign_fixations_process_words_trial(trial, show=True)
                )

                if not info_trial["discarted"]:
                    words_fix[trial] = words_fix_trial
                    fixations[trial] = fixations_trial
                else:
                    print(f"[SKIP] Trial {trial} discarded for distance or insuffient data.")
                info.append(info_trial)
            except Exception as e:
                print(f"[ERRORE] Trial {trial} fallito: {str(e)}")
                traceback.print_exc()  # <<< Aggiungi questa riga
                info.append({
                    "trial": trial,
                    "discarted": True,
                    "calibrate": False,
                    "mean_distance": None
                })
       

        # === PROMPT INT(TRIAL) ===
        for int_trial in int_trials:
            print(f"[PROMPT] Processing trial {int_trial}")
            try:
                fix_prompt, words_prompt, total_prompt, info_prompt = (
                    self.asign_fixations_process_words_trial_prompt(int_trial, show=False)
                )
            except Exception as e:
                print(f"[internal ERROR] in prompt {int_trial}: {type(e)}, {e}")
                raise e  # per vedere il traceback completo

            if not info_prompt["discarted"]:
                words_fix[int_trial] = words_prompt
                fixations[int_trial] = fix_prompt
            else:
                print(f"[SKIP] Prompt {int_trial} discarded for distance or insuffient data.")
            info.append(info_prompt)
        

        return fixations, words_fix, info

    def compute_fixations_regressions_trial(self, word_fix_trial):
        """
        Compute fixations regressions of trial
        A regression is when fixations + 1 is in a word before the word of a fixations
        """
        fixations_regressions = 0
        for i in range(len(word_fix_trial) - 1):
            if word_fix_trial["word_number"][i] < word_fix_trial["word_number"][i + 1]:
                fixations_regressions += 1
        return fixations_regressions

    def compute_general_features_trial(self, fixations_trial, words_fix_trial):
        avg_word_length = self.average_word_length(words_fix_trial)
        number_words = len(words_fix_trial)
        number_fixations = len(fixations_trial)
        fixations_regressions = round(
            self.compute_fixations_regressions_trial(fixations_trial)
            / number_fixations,
            4,
        )
        if "first_fix_duration" in words_fix_trial.columns:
            first_fix_duration = round(
                words_fix_trial["first_fix_duration"].sum() / number_words, 4
            )
        else:
            first_fix_duration = 0
        fix_duration = round(words_fix_trial["fix_duration"].sum() / number_words, 4)
        fix_number = round(words_fix_trial["fix_number"].sum() / number_words, 4)
        pupil = round(words_fix_trial["pupil"].mean(), 4)
        if "go_past_time" in words_fix_trial.columns:
            go_past_time = round(words_fix_trial["go_past_time"].mean(), 4)
        else:
             go_past_time = 0
        total_reading_time = round(fixations_trial.iloc[1:]["duration"].sum(), 4)# first fixation never assigned to any word

        return {
            "number_words_mean": number_words,
            "avg_word_length_mean": avg_word_length,
            "first_fix_duration_mean": first_fix_duration,
            "fix_duration_mean": fix_duration,
            "fix_number_mean": fix_number,
            "pupil_mean": pupil,
            "fixations_regressions_mean": fixations_regressions,
            "go_past_time_mean": go_past_time,
            "trt_trial": total_reading_time,
        }

    def _asign_fixations_words_trial(self, words_fix_trial, fixations_trial):
        words_fix_trial = copy.deepcopy(words_fix_trial)
        fixations_trial = copy.deepcopy(fixations_trial)
        words_fix_trial["fixations"] = [list() for _ in range(len(words_fix_trial))]
        words_fix_trial["fix_distance"] = [list() for _ in range(len(words_fix_trial))]
        words_fix_trial["fix_duration"] = 0.0
        words_fix_trial["first_fix_duration"] = 0.0
        words_fix_trial["fix_number"] = 0 
        words_fix_trial["pupil_l"] = 0.0
        words_fix_trial["pupil_r"] = 0.0

        word_line_prev, fixation_x_prev = 1, 0
        total_distance = 0
        for fixation_idx, fixation_row in fixations_trial.iterrows():
            # first fixation is not assigned to any word becouse usually is in the cross
            if fixation_idx == fixations_trial.index.min():
                continue
            # main algorithm to asign fixations to words
            words_fix_trial["distance"] = words_fix_trial.apply(
                self.euclidean_distance,
                args=(fixation_row["x"], fixation_row["y"]),
                axis=1,
            )
            closest_word_row, distance = self.compute_closets_word_row(
                words_fix_trial, fixation_row, word_line_prev, fixation_x_prev
            )
            '''
            if closest_word_row is None and distance is None:
                # if closest_word_row is None and distance is None we have to skip this fixation
                continue
            '''

            total_distance += distance
            words_fix_trial.loc[closest_word_row["number"], "fixations"].append(
                fixation_row["ID"]
            )
            words_fix_trial.loc[closest_word_row["number"], "fix_distance"].append(
                distance
            )
            words_fix_trial.loc[closest_word_row["number"], "fix_duration"] += (
                fixation_row["duration"]
            )
            # we save first fixations duration
            if words_fix_trial.loc[closest_word_row["number"], "fix_number"] == 0:
                words_fix_trial.loc[
                    closest_word_row["number"], "first_fix_duration"
                ] += fixation_row["duration"]

            words_fix_trial.loc[closest_word_row["number"], "fix_number"] += 1
            words_fix_trial.loc[closest_word_row["number"], "pupil_l"] += fixation_row[
                "pupil_l"
            ]
            words_fix_trial.loc[closest_word_row["number"], "pupil_r"] += fixation_row[
                "pupil_r"
            ]
            # coment this line. not sure why I was computing again the closets row.
            # closest_word_row = words_fix_trial.loc[words_fix_trial["distance"].idxmin()]
            # asign closest_word_row to fixations_trial['word'] to this row
            fixations_trial.loc[fixation_idx, "word"] = closest_word_row["text"]
            fixations_trial.loc[fixation_idx, "word_number"] = closest_word_row[
                "number"
            ]
            fixations_trial.loc[fixation_idx, "distance"] = distance
            word_line_prev = closest_word_row["line"]
            fixation_x_prev = fixation_row["x"]

        words_fix_trial["pupil_l"] = (
            words_fix_trial["pupil_l"] / words_fix_trial["fix_number"]
        )
        words_fix_trial["pupil_l"] = words_fix_trial["pupil_l"].apply(
            lambda x: round(x, 4)
        )
        words_fix_trial["pupil_r"] = (
            words_fix_trial["pupil_r"] / words_fix_trial["fix_number"]
        )
        words_fix_trial["pupil_r"] = words_fix_trial["pupil_r"].apply(
            lambda x: round(x, 4)
        )
        words_fix_trial["pupil"] = round(
            (words_fix_trial["pupil_l"] + words_fix_trial["pupil_r"]) / 2, 2
        )
        words_fix_trial["fix_duration_avg"] = (
            words_fix_trial["fix_duration"] / words_fix_trial["fix_number"]
        )
        words_fix_trial["fix_duration_avg"] = words_fix_trial["fix_duration_avg"].apply(
            lambda x: round(x, 4)
        )
        words_fix_trial["fix_duration"] = words_fix_trial["fix_duration"].apply(
            lambda x: round(x, 4)
        )
        words_fix_trial["first_fix_duration"] = words_fix_trial[
            "first_fix_duration"
        ].apply(lambda x: round(x, 4))
        return words_fix_trial, fixations_trial, total_distance

    def _filter_fixations_proportion(self, fixations_trial_original, proportion=0.1):
        #remove from dataset the 10% with high values in distance column
        fixations_trial = copy.deepcopy(fixations_trial_original)
        fixations_trial = fixations_trial[(fixations_trial["distance"].isna()) | (fixations_trial["distance"] < fixations_trial["distance"].quantile(1-proportion))]
        return list(set(fixations_trial_original["ID"].values.tolist()) - set(fixations_trial["ID"].values.tolist()) )

    def _filter_fixations_value(self, fixations_trial_original, value=100):
        #remove from dataset the 10% with high values in distance column
        fixations_trial = copy.deepcopy(fixations_trial_original)
        fixations_trial = fixations_trial[(fixations_trial["distance"].isna()) | (fixations_trial["distance"] < value)]
        return list(set(fixations_trial_original["ID"].values.tolist()) - set(fixations_trial["ID"].values.tolist()) )


    def asign_fixations_words_trial(self, trial, check_calibration=False, filter_value=0, filter_proportion=0,  return_notcalibrated=True, return_removed=True):
        """
        Asign fixations to words of trial. change fixations_trial and data_trial dataframes to add this information
        """
        words_fix_trial_original = None
        if hasattr(self, "coordinates"):
            if trial in self.coordinates:
                words_fix_trial_original = self.coordinates[trial]
        # words_fix_trial_original = None
        if not isinstance(words_fix_trial_original, pd.DataFrame):
            # if the the coordenates of words of trial are not in self.coordinates we try to load the csv
            words_fix_trial_original = self._read_csv_trial(trial)
            # words_fix_trial_original = None
            if not isinstance(words_fix_trial_original, pd.DataFrame):
                # If there we no csv with the coordenate of words of trial we read the image and extract them
                words_fix_trial_original = self._read_image_trial(trial)

        fixations_trial_remove = []
        fixations_trial_original = self._get_fixations_trial(trial)
        words_fix_trial, fixations_trial, total_distance = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial_original)
        
        if filter_value > 0:
                fixations_trial_remove.extend(self._filter_fixations_value(fixations_trial, filter_value))
        if filter_proportion > 0:
            fixations_trial_remove.extend(self._filter_fixations_proportion(fixations_trial, filter_proportion))

        if check_calibration:
            fixations_trial_remove_cal = []
            calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial(
                trial
            )
            fixations_trial_original["x"] = fixations_trial_original["x"] - calibrate_x
            fixations_trial_original["y"] = fixations_trial_original["y"] - calibrate_y
            words_fix_trial_cal, fixations_trial_cal, total_distance_cal = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial_original)
            
            if filter_value > 0:
                fixations_trial_remove_cal.extend(self._filter_fixations_value(fixations_trial_cal, filter_value))
            if filter_proportion > 0:
                fixations_trial_remove_cal.extend(self._filter_fixations_proportion(fixations_trial_cal, filter_proportion))
                
            if return_notcalibrated:
                if return_removed:
                    return words_fix_trial, fixations_trial, total_distance, fixations_trial_remove, words_fix_trial_cal, fixations_trial_cal, total_distance_cal, fixations_trial_remove_cal
                return words_fix_trial, fixations_trial, total_distance, words_fix_trial_cal, fixations_trial_cal, total_distance_cal
        
            if total_distance_cal < total_distance:
                fixations_trial, words_fix_trial, total_distance, fixations_trial_remove = (
                    fixations_trial_cal,
                    words_fix_trial_cal,
                    total_distance_cal,
                    fixations_trial_remove_cal
                )
        if return_removed:
            return words_fix_trial, fixations_trial, total_distance, fixations_trial_remove
        return fixations_trial, words_fix_trial, total_distance
    
    def asign_fixations_process_words_trial(self, trial, filter_value=100, filter_proportion=0.1, show=False):
        """
        Asign fixations to words of trial. change fixations_trial and data_trial dataframes to add this information
        """
        info = {"trial": trial, "discarted" : False, "calibrate": False, "mean_distance": None, "mean_distance_cal": None, "mean_distance_10": None, "mean_distance_cal_10": None, "mean_distance_nr": None, "fix_removed": 0}
        words_fix_trial_original = None
        if hasattr(self, "coordinates"):
            if trial in self.coordinates:
                words_fix_trial_original = self.coordinates[trial]
        # words_fix_trial_original = None
        if not isinstance(words_fix_trial_original, pd.DataFrame):
            # if the the coordenates of words of trial are not in self.coordinates we try to load the csv
            words_fix_trial_original = self._read_csv_trial(trial)
            # words_fix_trial_original = None
            if not isinstance(words_fix_trial_original, pd.DataFrame):
                # If there we no csv with the coordenate of words of trial we read the image and extract them
                words_fix_trial_original = self._read_image_trial(trial)
            #if its a datafarme but its empty, we try to obtain the words from the image
            if isinstance(words_fix_trial_original, pd.DataFrame):
                if  words_fix_trial_original.empty:
                    words_fix_trial_original = self._read_image_trial(trial)


        #compute mean distance of fixations without calibration
        fixations_trial_original = self._get_fixations_trial(trial)
        if fixations_trial_original.empty:
            info["discarted"] = True
            return pd.DataFrame(), words_fix_trial_original, 0, info
        words_fix_trial, fixations_trial, total_distance = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial_original)
        #in case there are no fixations
        if "distance" not in fixations_trial.columns:
            info["discarted"] = True
            return pd.DataFrame(), words_fix_trial_original, 0, info
        mean_distance = round(fixations_trial["distance"].mean(), 2)
        info["mean_distance"] = mean_distance
       
        #compute mean distance in calibrated fixations
        calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial(
            trial
        )
        fixations_trial_original["x"] = fixations_trial_original["x"] - calibrate_x
        fixations_trial_original["y"] = fixations_trial_original["y"] - calibrate_y
        words_fix_trial_cal, fixations_trial_cal, total_distance_cal = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial_original)
        #in case there are no fixations

        mean_distance_cal = round(fixations_trial_cal["distance"].mean(), 2)
        info["mean_distance_cal"] = mean_distance_cal
        #if both mean distances are more than 100 we try removing thw 10% worse in both
        if mean_distance > 100 and mean_distance_cal > 100:
            #compute mean distance removing 10% worse without calibration
            fixations_trial_remove = self._filter_fixations_proportion(fixations_trial, filter_proportion)
            fixations_trial_notremoved = fixations_trial[~fixations_trial['ID'].isin(fixations_trial_remove)]
            mean_distance_notremoved = round(fixations_trial_notremoved["distance"].mean(), 2)
            info["mean_distance_10"] = mean_distance_notremoved
            #compute mean distance removing 10% worse in calibrated fixations
            fixations_trial_remove_cal = self._filter_fixations_proportion(fixations_trial_cal, filter_proportion)     
            fixations_trial_cal_notremoved = fixations_trial_cal[~fixations_trial_cal['ID'].isin(fixations_trial_remove_cal)]
            mean_distance_cal_notremoved = round(fixations_trial_cal_notremoved["distance"].mean(), 2)
            info["mean_distance_cal_10"] = mean_distance_cal_notremoved
            if mean_distance_notremoved > 100 and mean_distance_cal_notremoved > 100:
                #if still both mean distances are more than 100 we remove this trial 
                info["discarted"] = True
                return pd.DataFrame(), words_fix_trial_original, 0, info
            else:
                mean_distance, mean_distance_cal = mean_distance_notremoved, mean_distance_cal_notremoved
               
        #we take the better between the calibrated and not calibrated fixations
        if mean_distance_cal < mean_distance:
            info["calibrate"] = True
            fixations_trial, words_fix_trial, total_distance = fixations_trial_cal, words_fix_trial_cal, total_distance_cal

        #we filter the final fixations of all more than 100
        fixations_trial_remove = self._filter_fixations_value(fixations_trial, filter_value)    
        
        self.plot_image_trial_colors(
                trial,
                fixations_trial=fixations_trial,
                words_fix_trial=words_fix_trial,
                fixations=True,
                coordinates=True,
                calibrate=False,
                ids_removed=fixations_trial_remove,
                print_fix_distance=True,
                save=True,
            ) 
        if len(fixations_trial_remove) > 0:
            #if in the last version I need to remove some fixations, i reasign the ones not remove to the words
            info["fix_removed"] = len(fixations_trial_remove)
            fixations_trial = fixations_trial[~fixations_trial['ID'].isin(fixations_trial_remove)]
            words_fix_trial, fixations_trial, total_distance = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial)
            info["mean_distance_nr"] = round(fixations_trial["distance"].mean(), 2)

        return fixations_trial, words_fix_trial, total_distance, info
    



    def asign_fixations_process_words_trial_prompt(self, trial, filter_value=100, filter_proportion=0.1, show=False):
        """
        Asign fixations to words of trial. change fixations_trial and data_trial dataframes to add this information
        """
        info = {"trial": trial, "discarted" : False, "calibrate": False, "mean_distance": None, "mean_distance_cal": None, "mean_distance_10": None, "mean_distance_cal_10": None, "mean_distance_nr": None, "fix_removed": 0}
        words_fix_trial_original = None
        if hasattr(self, "coordinates"):
            if trial in self.coordinates:
                words_fix_trial_original = self.coordinates[trial]
        # words_fix_trial_original = None
        if not isinstance(words_fix_trial_original, pd.DataFrame):
            # if the the coordenates of words of trial are not in self.coordinates we try to load the csv
            words_fix_trial_original = self._read_csv_trial_prompt(trial)
            # words_fix_trial_original = None
           

            if not isinstance(words_fix_trial_original, pd.DataFrame):
                # If there we no csv with the coordenate of words of trial we read the image and extract them
                words_fix_trial_original = self._read_image_trial_prompt(trial)
            #if its a datafarme but its empty, we try to obtain the words from the image
            if isinstance(words_fix_trial_original, pd.DataFrame):
                if  words_fix_trial_original.empty:
                    words_fix_trial_original = self._read_image_trial_prompt(trial)
                    

        #compute mean distance of fixations without calibration
        
        fixations_trial_original = self._get_fixations_trial_prompt(trial)
        if fixations_trial_original.empty:
            info["discarted"] = True
            return pd.DataFrame(), words_fix_trial_original, 0, info
        words_fix_trial, fixations_trial, total_distance = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial_original)
        #in case there are no fixations
        if "distance" not in fixations_trial.columns:
            info["discarted"] = True
            return pd.DataFrame(), words_fix_trial_original, 0, info
        mean_distance = round(fixations_trial["distance"].mean(), 2)
        info["mean_distance"] = mean_distance
       
        #compute mean distance in calibrated fixations
        calibrate_x, calibrate_y = self._compute_calibration_coordinates_trial_prompt(
            trial
        )
        if calibrate_x is not None and calibrate_y is not None:
            fixations_trial_original["x"] = fixations_trial_original["x"] - calibrate_x
            fixations_trial_original["y"] = fixations_trial_original["y"] - calibrate_y
            words_fix_trial_cal, fixations_trial_cal, total_distance_cal = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial_original)
            #in case there are no fixations

            mean_distance_cal = round(fixations_trial_cal["distance"].mean(), 2)
            info["mean_distance_cal"] = mean_distance_cal
            #if both mean distances are more than 100 we try removing thw 10% worse in both
            if mean_distance > 100 and mean_distance_cal > 100:
                #compute mean distance removing 10% worse without calibration
                fixations_trial_remove = self._filter_fixations_proportion(fixations_trial, filter_proportion)
                fixations_trial_notremoved = fixations_trial[~fixations_trial['ID'].isin(fixations_trial_remove)]
                mean_distance_notremoved = round(fixations_trial_notremoved["distance"].mean(), 2)
                info["mean_distance_10"] = mean_distance_notremoved
                #compute mean distance removing 10% worse in calibrated fixations
                fixations_trial_remove_cal = self._filter_fixations_proportion(fixations_trial_cal, filter_proportion)     
                fixations_trial_cal_notremoved = fixations_trial_cal[~fixations_trial_cal['ID'].isin(fixations_trial_remove_cal)]
                mean_distance_cal_notremoved = round(fixations_trial_cal_notremoved["distance"].mean(), 2)
                info["mean_distance_cal_10"] = mean_distance_cal_notremoved
                if mean_distance_notremoved > 100 and mean_distance_cal_notremoved > 100:
                    #if still both mean distances are more than 100 we remove this trial 
                    info["discarted"] = True
                    return pd.DataFrame(), words_fix_trial_original, 0, info
                else:
                    mean_distance, mean_distance_cal = mean_distance_notremoved, mean_distance_cal_notremoved
                
            #we take the better between the calibrated and not calibrated fixations
            if mean_distance_cal < mean_distance:
                info["calibrate"] = True
                fixations_trial, words_fix_trial, total_distance = fixations_trial_cal, words_fix_trial_cal, total_distance_cal

            #we filter the final fixations of all more than 100
            fixations_trial_remove = self._filter_fixations_value(fixations_trial, filter_value)    
            
            self.plot_image_trial_prompt_colors(
                    trial,
                    fixations_trial=fixations_trial,
                    words_fix_trial=words_fix_trial,
                    fixations=True,
                    coordinates=True,
                    calibrate=False,
                    ids_removed=fixations_trial_remove,
                    print_fix_distance=True,
                    save=True,
                ) 
            if len(fixations_trial_remove) > 0:
                #if in the last version I need to remove some fixations, i reasign the ones not remove to the words
                info["fix_removed"] = len(fixations_trial_remove)
                fixations_trial = fixations_trial[~fixations_trial['ID'].isin(fixations_trial_remove)]
                words_fix_trial, fixations_trial, total_distance = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial)
                info["mean_distance_nr"] = round(fixations_trial["distance"].mean(), 2)

            return fixations_trial, words_fix_trial, total_distance, info
        else:
            info["calibrate"] = False
            fixations_trial_remove = self._filter_fixations_value(fixations_trial, filter_value)
            self.plot_image_trial_prompt_colors(
                    trial,
                    fixations_trial=fixations_trial,
                    words_fix_trial=words_fix_trial,
                    fixations=True,
                    coordinates=True,
                    calibrate=False,
                    ids_removed=fixations_trial_remove,
                    print_fix_distance=True,
                    save=True,
                ) 
            if len(fixations_trial_remove) > 0:
                #if in the last version I need to remove some fixations, i reasign the ones not remove to the words
                info["fix_removed"] = len(fixations_trial_remove)
                fixations_trial = fixations_trial[~fixations_trial['ID'].isin(fixations_trial_remove)]
                words_fix_trial, fixations_trial, total_distance = self._asign_fixations_words_trial_1(words_fix_trial_original, fixations_trial)
                info["mean_distance_nr"] = round(fixations_trial["distance"].mean(), 2)

            return fixations_trial, words_fix_trial, total_distance, info
        
    def compute_go_past_time_per_word(self, fixations_trial: pd.DataFrame, debug=False):
        """
        Compute go-past time for each word in a trial
        Returns: dict {word_number: go_past_time}
        """
        go_past_times = {}
        if "word_number" not in fixations_trial.columns:
            if debug:
                print("[DEBUG] 'word_number' column not found in fixations.")
            return go_past_times
        
        fixations_sorted = fixations_trial.sort_index()

        word_numbers = fixations_sorted["word_number"].dropna().unique()
        
        if debug:
            print(f"[DEBUG] Starting go-past computation for {len(word_numbers)} words...")

        for wnum in word_numbers:
            try:
                wnum = int(wnum)
                go_past_time = 0
                entered = False
                if debug:
                    print(f"\n[DEBUG] → Word #{wnum}")

                for i, row in fixations_sorted.iterrows():
                    row_wnum = row.get("word_number")
                    if pd.isna(row_wnum):
                        continue
                    row_wnum = int(row_wnum)

                    if row_wnum == wnum:
                        if not entered and debug:
                            print(f"  [ENTER] Fixation ID {row['ID']} enters word {wnum}")
                        entered = True
                        go_past_time += row["duration"]
                        if debug:
                            print(f"   +{row['duration']}ms (on word {wnum})")
                    elif entered and row_wnum > wnum:
                        if debug:
                            print(f"  [EXIT] Fixation ID {row['ID']} on word {row_wnum} → stop")
                        break
                    elif entered:
                        go_past_time += row["duration"]
                        if debug:
                            print(f"   +{row['duration']}ms (regression or same)")

                go_past_times[wnum] = round(go_past_time, 2)

                if debug:
                    print(f"  [TOTAL] go-past time for word {wnum}: {go_past_times[wnum]} ms")
            except Exception as e:
                print(f"[ERROR] Computing go-past for word {wnum}: {e}")
                continue

        return go_past_times


    def _asign_fixations_words_trial_1(self, words_fix_trial, fixations_trial):
        # asign a line to each word
        words_fix_trial = copy.deepcopy(words_fix_trial)
        fixations_trial = copy.deepcopy(fixations_trial)
        words_fix_trial = self._search_lines_trial(words_fix_trial)
        words_fix_trial, fixations_trial, total_distance = (
            self._asign_fixations_words_trial(words_fix_trial, fixations_trial)
        )
        # compute go past time per word
        go_past_times = self.compute_go_past_time_per_word(fixations_trial, debug=False)
        words_fix_trial["go_past_time"] = words_fix_trial["number"].map(go_past_times)
        return words_fix_trial, fixations_trial, total_distance

    def compute_closets_word_row(self, words_fix_trial, fixation_row, word_line_prev, fixation_x_prev):
        """
        Compute the closest word row
        """
        '''
        if words_fix_trial["distance"].isna().all():
            print("No valid distances found")
            # Non c'è nessuna distanza valida → nessuna parola vicina
              # oppure puoi segnalare None
            return None, None
        '''

        idx = words_fix_trial["distance"].idxmin()
        closest_word_row = words_fix_trial.loc[idx]
        distance = words_fix_trial["distance"].min()

        # Logica di controllo linea precedente (come già c'è)
        if (
            closest_word_row["line"] != word_line_prev
            and fixation_row["x"] > fixation_x_prev
        ):
            words_fix_trial_new = copy.deepcopy(words_fix_trial)
            y_mean_line_previous_fix = words_fix_trial_new[
                words_fix_trial_new["line"] == word_line_prev
            ]["y"].mean()
            if abs(fixation_row["y"] - y_mean_line_previous_fix) < 110:
                words_fix_trial_new["distance"] = words_fix_trial_new.apply(
                    self.euclidean_distance,
                    args=(fixation_row["x"], y_mean_line_previous_fix),
                    axis=1,
                )
                if words_fix_trial_new["distance"].isna().all():
                    return closest_word_row, distance
                idx = words_fix_trial_new["distance"].idxmin()
                closest_word_row = words_fix_trial_new.loc[idx]
                distance = words_fix_trial["distance"].loc[idx]
        
        return closest_word_row, distance
    
    def aggregate_word_fixprop_across_users(self, users_list, sessions_list):
        """
        Calcola la proporzione di utenti che hanno fissato ogni parola per ogni trial,
        includendo anche le parole mai fissate (fixprop = 0.0).
        Salva il risultato in words_fixprop.csv dentro self.path.
        """
        from collections import defaultdict
        import pandas as pd
        import numpy as np
        import re
        import traceback

        word_user_map = defaultdict(set)  # (trial, word_number, text) → set(usernames)
        all_words = set()  # tutte le parole (anche mai fissate)

        for user in users_list:
            for session in sessions_list:
                try:
                    etdi = EyeTrackingDataImage(
                        user=user,
                        session=session,
                        user_set=0,
                        x_screen=self.x_screen,
                        y_screen=self.y_screen,
                        path=self.path
                    )

                    for trial in etdi.trials:
                        df = etdi.load_words_fixations_trial(trial)
                        if df is None or df.empty:
                            continue

                        for _, row in df.iterrows():
                            try:
                                key = (trial, int(row["number"]), row["text"])
                                all_words.add(key)

                                fix_ids_raw = row["fixations"]

                                fix_ids = []
                                if isinstance(fix_ids_raw, str):
                                    # Estrai float anche da stringhe tipo '[np.float64(1234.0)]'
                                    matches = re.findall(r"[-+]?\d*\.\d+|\d+", fix_ids_raw)
                                    fix_ids = [float(m) for m in matches]
                                else:
                                    fix_ids = fix_ids_raw

                                if isinstance(fix_ids, list) and len(fix_ids) > 0:
                                    word_user_map[key].add(user)

                            except Exception as row_error:
                                print(f"[ROW ERROR] Trial {trial}, user {user}: {row_error}")
                                continue

                except Exception as e:
                    print(f"[WARN] Skipping user {user}, session {session}: {e}")
                    traceback.print_exc()
                    continue

        # Costruisci il DataFrame finale
        total_users = len(users_list)
        fixprop_data = []

        for key in sorted(all_words):
            trial, word_number, text = key
            users = word_user_map.get(key, set())
            fixprop_data.append({
                "trial": trial,
                "word_number": word_number,
                "text": text,
                "fixprop": round(len(users) / total_users, 4)
            })

        fixprop_df = pd.DataFrame(fixprop_data)
        fixprop_df.sort_values(by=["trial", "word_number"], inplace=True)

        output_path = self.path + "/words_fixprop.csv"
        fixprop_df.to_csv(output_path, index=False)

        print(f"[OK] File salvato: {output_path}")
        return fixprop_df

    @staticmethod
    def average_word_length(df):
        total_letters = sum(
            len(''.join(char for char in word if char.isalpha()))
            for word in df['text'] if isinstance(word, str)
        )
        total_words = df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0).sum()
        avg = total_letters / total_words if total_words > 0 else float('nan')
        return round(avg, 2)