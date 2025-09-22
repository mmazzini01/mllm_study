from eyetrackpy_new.data_processor.models.eye_tracking_data_image import EyeTrackingDataImage

import pytesseract
import pandas as pd 
import os
import shutil
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import glob

#users_list = ['tomas','sebastian','mohammed','alessio','antonella','lorenzo_ciocca','mireia','lorenzo_f','noemi','ralitsa']
users_list = ['1','2','3','4','5','6','7','8','9','10','11','13','14','15','16']
sessions_list = [1]
for user in users_list:
    for session in sessions_list:
        etdi = EyeTrackingDataImage(
        user=user,
        session=session,
        user_set=0,
        x_screen=1536,
        y_screen=864,
        path="./users"
        )
        fixations, words_fix, info = etdi.asign_fixations_process_words_all()
        etdi.save_fixations(words_fix, fixations_all=fixations, info=info)
        features = pd.DataFrame(etdi.compute_entropy_all())
        etdi.save_features(features)
        print(f"finished for {user} in session {session}")


etdi.aggregate_word_fixprop_across_users(
    users_list=users_list,
    sessions_list=sessions_list
)