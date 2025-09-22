import os
import sys
cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cwd)
cwd = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cwd)
cwd = os.path.abspath(__file__)
sys.path.append(cwd)
# Set the timeout to 5 seconds
os.environ["PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT"] = "5.0"

from utils.data_loader import ETDataLoader
from attention.models.model_visual_att import ModelVisualAttentionExtractor
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--method",
        default='normal',
    )
    args = parser.parse_args()
    models = {
        # -----------------------------------------------
        "llava-hf/llava-1.5-7b-hf": "causalLM",
        "llava-hf/llava-1.5-13b-hf": "causalLM", 
    }
    method = args.method
    if method == 'rollout':
            folder = 'normal/'
    else:
        folder = 'attention/'
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_data_path = cwd + "/data/raw/et"
    process_data_path = cwd + "/data/processed/et"
    save_path= cwd + '/results/et/'

    responses, images, prompts, prompts_screenshots = ETDataLoader().load_data(raw_data_path=raw_data_path)
    path_images = raw_data_path + "/images/"
    images_trials_paths = {trial: path_images + "img_prompt_" + str(trial) + ".jpg" for trial, image in prompts.items()}
    words = ETDataLoader().load_texts_responses(raw_data_path=raw_data_path)
    responses_words  ={}
    prompts_words = {}
    for trial, list_text in words.items():
        if not '.' in str(trial):
            prompts_words[trial] = list_text
        else:
            responses_words[trial] = list_text

    for model_name, model_type in models.items():
        # Load the model
        model_name.replace("/", "_")
        
        folder_path_attention = (
            save_path
            + folder
            + model_name.split("/")[1]
            + "/"
        )
        print(f"Folder path attention: {folder_path_attention} for model {model_name} and method {method}")
       
        
        att_extractor = ModelVisualAttentionExtractor(model_name, model_type, folder_path_attention)
        # prompts_words = {'20': prompts_words['20']}
        #compute attention for prompts
        attention_trials_image, attention_trials_text, info = att_extractor.extract_attention(
            prompts_words, images_trials_paths=images_trials_paths,
            attention_method=method
        )

        att_extractor.save_attention_trials_image(images_trials_paths, attention_trials_image, info, folder_path_attention + "saliency/")
        att_extractor.save_attention_df(attention_trials_text, prompts_words, folder_path_attention)

        #compute attention for responses
        attention_trials_text = att_extractor.extract_attention_only_text(
            responses_words, attention_method=method
        )
        att_extractor.save_attention_df(attention_trials_text, responses_words, folder_path_attention)

        