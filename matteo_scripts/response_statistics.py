import os
import warnings
import math
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr


class ResponseStatisticsProcessor:
    """
    A class to process response statistics and create analysis files.
    
    This class handles:
    1. Creating trial_responses.csv from user data
    2. Creating response_model.csv from model response files
    3. Creating response_summary.csv with model performance statistics
    """
    
    def __init__(self, input_users_dir="./users", input_responses_dir="responses_files", 
                 output_dir="./users/response_statistics"):
        """
        Initialize the processor with input and output directories.
        
        Args:
            input_users_dir (str): Directory containing user participant data
            input_responses_dir (str): Directory containing model response Excel files
            output_dir (str): Directory where output CSV files will be saved
        """
        self.input_users_dir = input_users_dir
        self.input_responses_dir = input_responses_dir
        self.output_dir = output_dir
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data structures
        self.trial_dict = {trial: {"response_1": 0, "response_2": 0} for trial in range(1, 31)}
        self.model_choices_by_trial = defaultdict(list)
        self.model_choices_details = defaultdict(lambda: defaultdict(list))
        
    def _get_preferred(self, row):
        """Helper function to determine preferred response."""
        if row['response_1'] >= row['response_2']:
            return f"{row['trial']}.1"
        else:
            return f"{row['trial']}.2"
    
    def _calculate_entropy(self, row):
        """Calculate entropy for a trial row."""
        p1 = row['prop_response_1']
        p2 = row['prop_response_2']
        ent = 0
        if p1 > 0:
            ent -= p1 * math.log2(p1)
        if p2 > 0:
            ent -= p2 * math.log2(p2)
        return ent
    
    def create_trial_responses(self):
        """
        Create trial_responses.csv by processing user participant data.
        
        Returns:
            pd.DataFrame: The trial responses dataframe
        """
        # Process user data
        for folder in os.listdir(self.input_users_dir):
            if "participant" in folder:
                info_path = os.path.join(self.input_users_dir, folder, "session_1", "info_extended.csv")
                if os.path.exists(info_path):
                    info_df = pd.read_csv(info_path)
                    for _, row in info_df.iterrows():
                        trial_value = str(row['trial'])
                        if '.' in trial_value:
                            base, suffix = trial_value.split('.')
                            trial_num = int(base)

                            # Count user preference
                            if row.get('chosen_user') == True:
                                if suffix == '1':
                                    self.trial_dict[trial_num]['response_1'] += 1
                                else:
                                    self.trial_dict[trial_num]['response_2'] += 1

                            # Collect model preference
                            if row.get('chosen_model') == True:
                                normalized_choice = f"{trial_num}.{suffix}"
                                self.model_choices_by_trial[trial_num].append(normalized_choice)
                                self.model_choices_details[trial_num][normalized_choice].append(folder)

        # Create dataframe
        rows = []
        for trial, counts in self.trial_dict.items():
            total = counts["response_1"] + counts["response_2"]
            if total != 14:
                if counts["response_1"] < counts["response_2"]:
                    counts["response_2"] += 1
                else:
                    counts["response_1"] += 1
            rows.append({"trial": trial, "response_1": counts["response_1"], "response_2": counts["response_2"]})
        
        output_df = pd.DataFrame(rows)
        output_df['chosen_user'] = output_df.apply(self._get_preferred, axis=1)
        
        # Calculate proportions
        output_df['prop_response_1'] = output_df.apply(
            lambda row: row['response_1'] / (row['response_1'] + row['response_2']) if (row['response_1'] + row['response_2']) > 0 else 0,
            axis=1
        )
        output_df['prop_response_2'] = output_df.apply(
            lambda row: row['response_2'] / (row['response_1'] + row['response_2']) if (row['response_1'] + row['response_2']) > 0 else 0,
            axis=1
        )
        
        # Calculate entropy
        output_df['entropy'] = output_df.apply(self._calculate_entropy, axis=1)
        output_df['entropy'] = output_df['entropy'].round(3)
        
        # Determine model preferences
        preferred_model_by_trial = {}
        for trial in range(1, 31):
            values = self.model_choices_by_trial.get(trial, [])
            if not values:
                preferred_model_by_trial[trial] = ''
                continue
            unique_values = set(values)
            if len(unique_values) == 1:
                preferred_model_by_trial[trial] = values[0]
            else:
                detail = self.model_choices_details[trial]
                details_str = ", ".join([f"{val} (users: {', '.join(sorted(detail[val]))})" for val in sorted(detail.keys())])
                warnings.warn(
                    f"Inconsistent model preference detected for trial {trial}: {details_str}"
                )
                most_common_value, _ = Counter(values).most_common(1)[0]
                preferred_model_by_trial[trial] = most_common_value

        output_df['preferred_by_the_model'] = output_df['trial'].map(lambda t: preferred_model_by_trial.get(t, ''))
        
        # Save trial responses
        trial_responses_path = os.path.join(self.output_dir, "trial_responses.csv")
        output_df.to_csv(trial_responses_path, index=False)
        
        return output_df
    
    def create_response_model(self):
        """
        Create response_model.csv by processing model response Excel files.
        
        Returns:
            pd.DataFrame: The all models dataframe
        """
        all_models = []
        for filename in os.listdir(self.input_responses_dir):
            if filename.endswith('.xlsx') and 'prompt' in filename:
                csv_path = os.path.join(self.input_responses_dir, filename)
                try:
                    df = pd.read_excel(csv_path)
                    df.drop(columns=['rank','resp_text'], inplace=True)
                    all_models.append(df)
                except Exception as e:
                    # Silently ignore errors as per instruction to remove prints
                    pass

        all_models = pd.concat(all_models)
        all_models.sort_values(by='related_prompt', inplace=True)
        all_models['n_resp'] = all_models['n_resp'].astype(str)
        
        # Save response model
        response_model_path = os.path.join(self.output_dir, "response_model.csv")
        all_models.to_csv(response_model_path, index=False)
        
        return all_models
    
    def create_response_summary(self, trial_responses_df, all_models_df):
        """
        Create response_summary.csv with model performance statistics.
        
        Args:
            trial_responses_df (pd.DataFrame): Trial responses dataframe
            all_models_df (pd.DataFrame): All models dataframe
            
        Returns:
            pd.DataFrame: The model summary dataframe
        """
        # Create mapping for chosen human responses
        chosen_human_data = []
        for _, row in trial_responses_df.iterrows():
            trial = row['trial']
            response_1 = row['response_1']
            response_2 = row['response_2']
            preferred_by_model = str(row['preferred_by_the_model'])
            
            # Add data for response 1
            chosen_human_data.append({
                'related_prompt': int(trial),
                'n_resp': f"{int(trial)}.1",
                'number_of_human': int(response_1),
                'chosen_model': (preferred_by_model == f"{int(trial)}.1"),
                'chosen_user': (True if int(response_1) > 5 else (False if int(response_1) < 5 else np.nan))
            })
            # Add data for response 2
            chosen_human_data.append({
                'related_prompt': int(trial),
                'n_resp': f"{int(trial)}.2",
                'number_of_human': int(response_2),
                'chosen_model': (preferred_by_model == f"{int(trial)}.2"),
                'chosen_user': (True if int(response_2) > 5 else (False if int(response_2) < 5 else np.nan))
            })

        chosen_human_df = pd.DataFrame(chosen_human_data)
        chosen_human_df['n_resp'] = chosen_human_df['n_resp'].astype(str)

        # Merge dataframes
        merged_df = pd.merge(all_models_df, chosen_human_df, on=['related_prompt', 'n_resp'], how='inner')
        merged_df.sort_values(['related_prompt', 'n_resp'], inplace=True)

        # Save merged dataset
        merged_path = os.path.join(self.output_dir, "merged_models_responses.csv")
        merged_df.to_csv(merged_path, index=False)

        # Create summary table
        model_summary = merged_df.groupby('model').agg({
            'related_prompt': 'count',
            'chosen_user': lambda x: (x == True).sum(),
            'chosen_model': lambda x: (x == True).sum()
        }).rename(columns={
            'related_prompt': 'appearances',
            'chosen_user': 'chosen_by_user',
            'chosen_model': 'chosen_by_model'
        })

        # Add percentage columns
        model_summary['user_choice_rate'] = (model_summary['chosen_by_user'] / model_summary['appearances'] * 100).round(1)
        model_summary['model_choice_rate'] = (model_summary['chosen_by_model'] / model_summary['appearances'] * 100).round(1)
        model_summary = model_summary.sort_values('appearances', ascending=False)

        # Save summary
        response_summary_path = os.path.join(self.output_dir, "response_summary.csv")
        model_summary.to_csv(response_summary_path)

        # Create LaTeX table
        latex_table = model_summary.to_latex(
            index=True,
            float_format='%.1f',
            columns=['appearances', 'chosen_by_user', 'chosen_by_model', 'user_choice_rate', 'model_choice_rate'],
            header=['Appearances', 'Chosen by User', 'Chosen by Model', 'User Choice Rate (%)', 'Model Choice Rate (%)'],
            escape=False
        )

        latex_path = os.path.join(self.output_dir, 'model_summary_latex.txt')
        with open(latex_path, 'w') as f:
            f.write(latex_table)

        return model_summary
    
    def compute_agreement_metrics(self, trial_responses_path: str = None):
        """
        Compute agreement metrics using trial responses CSV.
        - Fleiss' kappa among users using counts per option (response_1, response_2)
        - Accuracy of model choice vs human majority (ties skipped)
        - Cohen's kappa between model labels and human majority labels
        """
        if trial_responses_path is None:
            trial_responses_path = os.path.join(self.output_dir, "trial_responses.csv")
            
        if not os.path.exists(trial_responses_path):
            return None

        df = pd.read_csv(trial_responses_path)
        
        # Fleiss' kappa: build matrix [n_items x n_categories]
        try:
            from statsmodels.stats.inter_rater import fleiss_kappa
            mat = df[["response_1", "response_2"]].to_numpy()
            kappa_umani = fleiss_kappa(mat, method='fleiss')
        except Exception as exc:
            kappa_umani = np.nan

        # Majority vs model accuracy and Cohen's kappa
        correct = 0
        total = 0
        majority_labels = []
        model_labels = []

        for _, row in df.iterrows():
            r1 = row.get('response_1')
            r2 = row.get('response_2')
            # Determine majority label (1 or 2), skip ties
            if pd.isna(r1) or pd.isna(r2):
                continue
            if r1 > r2:
                majority = 1
            elif r2 > r1:
                majority = 2
            else:
                continue

            # Parse model choice from pattern "<trial>.<variant>"
            model_field = str(row.get('preferred_by_the_model', ""))
            try:
                model_choice = int(model_field.split('.')[-1])
            except Exception:
                # If unparsable, skip this item
                continue

            # Accumulate accuracy
            if model_choice == majority:
                correct += 1
            total += 1

            # For Cohen's kappa lists
            majority_labels.append(majority)
            model_labels.append(model_choice)

        accuracy = (correct / total) if total > 0 else np.nan
        try:
            kappa_mod_vs_majority = cohen_kappa_score(model_labels, majority_labels) if len(model_labels) > 0 else np.nan
        except Exception as exc:
            kappa_mod_vs_majority = np.nan

        # Correlazione lineare (Pearson) tra entropia e errore modello (1 - correct)
        def _is_correct(row):
            try:
                if row['prop_response_1'] > 0.5:
                    majority = 1
                elif row['prop_response_2'] > 0.5:
                    majority = 2
                else:
                    return np.nan  # tie
                model_choice = int(str(row['preferred_by_the_model']).split('.')[1])
                return int(model_choice == majority)
            except Exception:
                return np.nan

        df['correct'] = df.apply(_is_correct, axis=1)
        df_clean = df.dropna(subset=['correct', 'entropy']).copy()
        if not df_clean.empty:
            rho_entropy_error, p_entropy_error = pearsonr(df_clean['entropy'], 1 - df_clean['correct'])
        else:
            rho_entropy_error, p_entropy_error = np.nan, np.nan

        # Calculate average trial entropy
        avg_trial_entropy = df['entropy'].mean() if not df['entropy'].isna().all() else np.nan

        return {
            'fleiss_kappa_users': kappa_umani,
            'model_vs_majority_accuracy': accuracy,
            'model_vs_majority_cohen_kappa': kappa_mod_vs_majority,
            'entropy_error_pearson_r': rho_entropy_error,
            'entropy_error_pearson_p': p_entropy_error,
            'avg_trial_entropy': avg_trial_entropy,
            'evaluated_trials': total
        }
    
    def process_all(self):
        """
        Process all files and create the complete analysis.
        
        Returns:
            tuple: (trial_responses_df, all_models_df, model_summary_df, agreement_metrics)
        """
        # Step 1: Create trial responses
        trial_responses_df = self.create_trial_responses()
        
        # Step 2: Create response model
        all_models_df = self.create_response_model()
        
        # Step 3: Create response summary
        model_summary_df = self.create_response_summary(trial_responses_df, all_models_df)
        
        # Step 4: Compute agreement metrics
        agreement_metrics = self.compute_agreement_metrics()
        
        # Step 5: Update summary text with agreement metrics
        self._update_summary_with_agreement_metrics(agreement_metrics)
        
        return trial_responses_df, all_models_df, model_summary_df, agreement_metrics
    
    def _update_summary_with_agreement_metrics(self, agreement_metrics):
        """Update the model summary LaTeX file with agreement metrics and save to CSV."""
        if agreement_metrics is None:
            return
            
        # Save agreement metrics to CSV
        agreement_csv_path = os.path.join(self.output_dir, 'agreement_metrics.csv')
        agreement_df = pd.DataFrame([agreement_metrics])
        agreement_df.to_csv(agreement_csv_path, index=False)
        
        # Update LaTeX summary
        summary_path = os.path.join(self.output_dir, 'model_summary_latex.txt')
        
        # Read existing content
        with open(summary_path, 'r') as f:
            content = f.read()
        
        # Add agreement metrics section
        agreement_section = f"""

% Agreement Metrics
\\section{{Agreement Metrics}}
\\begin{{itemize}}
    \\item Fleiss' Kappa (Users): {agreement_metrics['fleiss_kappa_users']:.3f}
    \\item Model vs Majority Accuracy: {agreement_metrics['model_vs_majority_accuracy']:.3f}
    \\item Model vs Majority Cohen's Kappa: {agreement_metrics['model_vs_majority_cohen_kappa']:.3f}
    \\item Entropy-Error Pearson r: {agreement_metrics['entropy_error_pearson_r']:.3f}
    \\item Entropy-Error Pearson p: {agreement_metrics['entropy_error_pearson_p']:.3f}
    \\item Average Trial Entropy: {agreement_metrics['avg_trial_entropy']:.3f}
    \\item Evaluated Trials: {agreement_metrics['evaluated_trials']}
\\end{{itemize}}
"""
        
        # Append agreement metrics to the content
        updated_content = content + agreement_section
        
        # Write back to file
        with open(summary_path, 'w') as f:
            f.write(updated_content)
    


# Example usage
if __name__ == "__main__":
    # Default usage
    processor = ResponseStatisticsProcessor()
    processor.process_all()
    
    # Custom directories usage
    # processor = ResponseStatisticsProcessor(
    #     input_users_dir="./custom_users",
    #     input_responses_dir="./custom_responses", 
    #     output_dir="./custom_output"
    # )
    # processor.process_all()

