import pandas as pd
import numpy as np
import os
import glob
from scipy import stats
from pathlib import Path
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

class SyntheticMetricsAnalyzer:
    def __init__(self, input_dir="users_synthetic", output_dir="users_synthetic"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.synthetic_data_path = input_dir  # Keep for backward compatibility
        self.aggregate_data = None
        self.participants = []
    
    def average_word_length(self, df):
        """Calculate average word length from text data"""
        total_letters = sum(
            len(''.join(char for char in word if char.isalpha()))
            for word in df['text'] if isinstance(word, str)
        )
        total_words = df['text'].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0).sum()
        avg = total_letters / total_words if total_words > 0 else float('nan')
        return round(avg, 2)
    
    def compute_general_features_trial(self, words_fix_trial, trial):
        """Compute general features for a single trial"""
        avg_word_length = self.average_word_length(words_fix_trial)
        number_words = len(words_fix_trial)
        if "first_fix_duration" in words_fix_trial.columns:
            first_fix_duration = round(
                words_fix_trial["first_fix_duration"].sum() / number_words, 4
            )
        else:
            first_fix_duration = 0
        fix_duration = round(words_fix_trial["fix_duration"].sum() / number_words, 4)
        fix_number = round(words_fix_trial["fix_number"].sum() / number_words, 4)
        if "GPT" in words_fix_trial.columns:
            go_past_time = round(words_fix_trial["GPT"].mean(), 4)
        else:
             go_past_time = 0
        total_reading_time = round(words_fix_trial["fix_duration"].sum(), 4)

        return {
            "trial": trial,
            "number_words_mean": number_words,
            "avg_word_length_mean": avg_word_length,
            "first_fix_duration_mean": first_fix_duration,
            "fix_duration_mean": fix_duration,
            "fix_number_mean": fix_number,
            "go_past_time_mean": go_past_time,
            "trt_trial": total_reading_time,
        }
    
    def get_participant_directories(self):
        """Get all participant directories"""
        participant_dirs = []
        for item in os.listdir(self.input_dir):
            item_path = os.path.join(self.input_dir, item)
            if os.path.isdir(item_path) and item.startswith('participant_'):
                participant_dirs.append(item)
        return sorted(participant_dirs)
    
    def read_participant_data(self, participant_dir):
        """Read general_features.csv and info_extended.csv for a participant"""
        session_path = os.path.join(self.input_dir, participant_dir, "session_1")
        
        # Read general features
        general_features_path = os.path.join(session_path, "general_features.csv")
        if not os.path.exists(general_features_path):
            return None
            
        general_features = pd.read_csv(general_features_path, sep=';')
        
        # Read info extended to get the 'chosen_user' and 'chosen_model' columns
        info_extended_path = os.path.join(session_path, "info_extended.csv")
        if not os.path.exists(info_extended_path):
            return None
            
        info_extended = pd.read_csv(info_extended_path)
        
        # Extract participant name
        participant_name = participant_dir.replace('participant_', '').replace('_', ' ')
        
        # Merge the dataframes on trial column
        merged_data = pd.merge(general_features, info_extended[['trial', 'chosen_user', 'chosen_model', 'discarted']], 
                              on='trial', how='inner')
        
        merged_data['participant'] = participant_name
        merged_data['participant_id'] = participant_dir

        return merged_data
    
    def aggregate_all_data(self):
        """Aggregate data from all participants"""
        participant_dirs = self.get_participant_directories()
        all_data = []
        
        for participant_dir in participant_dirs:
            data = self.read_participant_data(participant_dir)
            if data is not None:
                all_data.append(data)
                self.participants.append(participant_dir.replace('participant_', '').replace('_', ' '))
        
        if all_data:
            self.aggregate_data = pd.concat(all_data, ignore_index=True)
            
            # Filter out trials with .0 values (keep only .1 and .2 trials)
            # Convert trial to string to check for .0, .1, .2 endings
            self.aggregate_data['trial_str'] = self.aggregate_data['trial'].astype(str)
            
            # Keep only trials ending with .1 or .2
            self.aggregate_data = self.aggregate_data[
                self.aggregate_data['trial_str'].str.endswith('.1') | 
                self.aggregate_data['trial_str'].str.endswith('.2')
            ]
            
            # Remove discarded trials and ensure only complete pairs remain PER PARTICIPANT
            discarded_trials = self.aggregate_data[self.aggregate_data['discarted'] == True]
            discarded_trial_bases = set()
            
            for _, row in discarded_trials.iterrows():
                trial_str = str(row['trial'])
                if '.' in trial_str:
                    base = trial_str.split('.')[0]
                    discarded_trial_bases.add(base)
            
            # Remove all trials (both .1 and .2) for trial bases that have any discarded trials
            self.aggregate_data = self.aggregate_data[
                ~self.aggregate_data['trial_str'].str.split('.').str[0].isin(discarded_trial_bases)
            ]
            
            # Clean up temporary column
            self.aggregate_data = self.aggregate_data.drop('trial_str', axis=1)
            
            return self.aggregate_data
        else:
            return None
    
    def save_aggregate_data(self, filename=None):
        """Save the aggregated data to CSV"""
        if self.aggregate_data is None:
            return
        
        if filename is None:
            filename = os.path.join(self.output_dir, "aggregate_metrics.csv")
        
        self.aggregate_data.to_csv(filename, index=False)
    
    def create_synthetic_data_structure(self):
        """Create the synthetic data structure and aggregate metrics CSV"""
        # List of users to process
        users = ['1','2','3','4','5','6','7','8','9','10','11','13','14','15','16']
        
        for user_set in users:
            folder_syn_par = os.path.join(self.input_dir, "participant_" + str(user_set) +'_'+ str(user_set), 'session_1')
            folder_syn_par_vertices = os.path.join(folder_syn_par, "vertices")
            
            # Create general_features.csv if it doesn't exist
            general_features_path = os.path.join(folder_syn_par, "general_features.csv")
            if not os.path.exists(general_features_path):
                features_all = []
                if os.path.exists(folder_syn_par_vertices):
                    for file in os.listdir(folder_syn_par_vertices):
                        if file.startswith("word_gaze_synthetic_"):
                            trial_str = file[len("word_gaze_synthetic_"):-len(".csv")]
                            words_fix_trial = pd.read_csv(os.path.join(folder_syn_par_vertices, file), sep=";")
                            features = self.compute_general_features_trial(words_fix_trial, trial_str)
                            features_all.append(features)
                    features_all = pd.DataFrame(features_all)
                    features_all.to_csv(general_features_path, sep=";")
            
            # Create info_extended.csv
            folder_real_par = os.path.join("users", "participant_" + str(user_set) +'_'+ str(user_set), 'session_1', "info_extended.csv")
            if os.path.exists(folder_real_par):
                info_extended = pd.read_csv(folder_real_par, sep=",")
                info_extended = info_extended.copy()
                info_extended = info_extended[["trial", 'chosen_user','chosen_model']]
                
                if os.path.exists(general_features_path):
                    general_features = pd.read_csv(general_features_path, sep=";")
                    info_extended['trial'] = info_extended['trial'].astype(str)
                    general_features['trial'] = general_features['trial'].astype(str)
                    merged = pd.merge(info_extended, general_features, on='trial', how='right')
                    merged['discarted'] = False
                    if 'Unnamed: 0' in merged.columns:
                        merged = merged.drop(columns=['Unnamed: 0'])
                    merged['trial_sort'] = merged['trial'].astype(float)
                    merged.sort_values(by='trial_sort', inplace=True)
                    merged.drop(columns=['trial_sort'], inplace=True)
                    merged.to_csv(os.path.join(folder_syn_par, "info_extended.csv"), sep=",", index=False)
        
    def load_synthetic_data(self):
        """Load synthetic data from the aggregate metrics CSV"""
        aggregate_metrics_path = os.path.join(self.input_dir, "aggregate_metrics.csv")
        if os.path.exists(aggregate_metrics_path):
            self.aggregate_data = pd.read_csv(aggregate_metrics_path)
            
            # Get unique participants
            self.participants = self.aggregate_data['participant'].unique()
            
            # Filter out trials with .0 values (keep only .1 and .2 trials)
            
            # Convert trial to string to check for .0, .1, .2 endings
            self.aggregate_data['trial_str'] = self.aggregate_data['trial'].astype(str)
            
            # Keep only trials ending with .1 or .2
            self.aggregate_data = self.aggregate_data[
                self.aggregate_data['trial_str'].str.endswith('.1') | 
                self.aggregate_data['trial_str'].str.endswith('.2')
            ]
            
            # Remove discarded trials and ensure only complete pairs remain PER PARTICIPANT
            discarded_trials = self.aggregate_data[self.aggregate_data['discarted'] == True]
            discarded_trial_bases = set()
            
            for _, row in discarded_trials.iterrows():
                trial_str = str(row['trial'])
                if '.' in trial_str:
                    base = trial_str.split('.')[0]
                    discarded_trial_bases.add(base)
            
            # Remove all trials (both .1 and .2) for trial bases that have any discarded trials
            self.aggregate_data = self.aggregate_data[
                ~self.aggregate_data['trial_str'].str.split('.').str[0].isin(discarded_trial_bases)
            ]
            
            # Clean up temporary column
            self.aggregate_data = self.aggregate_data.drop('trial_str', axis=1)
            
            return self.aggregate_data
        else:
            return None
    
    
    
    def perform_reading_metrics_tests(self):
        """Perform one-sample t-tests on reading metrics (FFD, TRT, nFIX) using only Antonella's data per trial level"""
        if self.aggregate_data is None:
            return None
        
        from scipy.stats import ttest_1samp
        
        # Filter data for participant_antonella_antonella only
        antonella_data = self.aggregate_data[self.aggregate_data['participant_id'] == 'participant_7_7'].copy()
        
        if antonella_data.shape[0] == 0:
            return None
        
        # Define the reading metrics to analyze
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        
        # Define chooser columns for both user and model
        chooser_cols = [
            ('user', 'chosen_user'),
            ('model', 'chosen_model')
        ]
        
        results = {}
        
        for metric_name, column_name in metrics.items():
            for chooser_key, chooser_col in chooser_cols:
                label = f"{metric_name}_{chooser_key}"
                
                if column_name not in antonella_data.columns:
                    continue
                
                # Create trial-level analysis like in rm_analyzer.py
                antonella_data["trial_id"] = antonella_data["trial"].astype(str).str.split(".").str[0]
                
                # Create pivot table with trials as rows and chosen/rejected as columns
                pivot_data = antonella_data.pivot_table(
                    index='trial_id', 
                    columns=chooser_col, 
                    values=column_name,
                    aggfunc='mean'
                ).reset_index()
                
                # Rename columns for clarity
                pivot_data.columns = ['trial', 'Rejected', 'Preferred']
                
                # Calculate differences (Preferred - Rejected)
                pivot_data['Difference'] = pivot_data['Preferred'] - pivot_data['Rejected']
                
                # Save paired data in reading_metric_statistics directory
                output_dir = os.path.join(self.output_dir, "reading_metric_statistics")
                os.makedirs(output_dir, exist_ok=True)
                output_filename = os.path.join(output_dir, f"synthetic_paired_data_{metric_name}_{chooser_col}.csv")
                pivot_data.to_csv(output_filename, index=False)
                
                # Calculate statistics
                mean_diff = pivot_data['Difference'].mean()
                std_diff = pivot_data['Difference'].std()
                n_trials = len(pivot_data)
                
                if n_trials > 0:
                    # Perform one-sample t-test (testing if mean difference is different from 0)
                    t_stat, p_value = ttest_1samp(pivot_data['Difference'].dropna(), 0, alternative='two-sided')
                    
                    # Calculate effect size (Hedges' g for one-sample t-test)
                    cohens_d = mean_diff / std_diff if std_diff != 0 else 0
                    if not np.isnan(cohens_d) and n_trials > 1:
                        # Apply Hedges' g correction: g = d * J, where J = 1 - 3/(4*df-1), df = n-1
                        df = n_trials - 1
                        J = 1 - 3/(4*df - 1) if df > 0 else 1
                        effect_size = cohens_d * J
                        effect_size_name = "Hedges' g"
                    else:
                        effect_size = cohens_d
                        effect_size_name = "Cohen's d"
                    
                    # Determine effect size interpretation (using Hedges' g thresholds)
                    if abs(effect_size) < 0.2:
                        effect_interpretation = "negligible"
                    elif abs(effect_size) < 0.5:
                        effect_interpretation = "small"
                    elif abs(effect_size) < 0.8:
                        effect_interpretation = "medium"
                    else:
                        effect_interpretation = "large"
                    
                    # Store results
                    results[label] = {
                        'metric': metric_name,
                        'chooser': chooser_key,
                        'column': column_name,
                        'n_trials': n_trials,
                        'preferred_mean': pivot_data['Preferred'].mean(),
                        'preferred_std': pivot_data['Preferred'].std(),
                        'rejected_mean': pivot_data['Rejected'].mean(),
                        'rejected_std': pivot_data['Rejected'].std(),
                        'difference_mean': mean_diff,
                        'difference_std': std_diff,
                        'test_type': "One-sample t-test (synthetic)",
                        'test_statistic': t_stat,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'effect_size_name': effect_size_name,
                        'cohens_d': cohens_d,
                        'effect_interpretation': effect_interpretation,
                        'is_significant': p_value < 0.05
                    }
                else:
                    results[label] = None
        
        return results
    
    
    
    def apply_fdr_correction(self, results, alpha=0.05):
        """Apply False Discovery Rate correction to p-values"""
        if results is None:
            return None
        
        # Extract p-values and their labels
        p_values = []
        labels = []
        
        for label, result in results.items():
            if result is not None and 'p_value' in result:
                p_values.append(result['p_value'])
                labels.append(label)
        
        if not p_values:
            return results
        
        # Apply FDR correction
        from statsmodels.stats.multitest import multipletests
        rejected, pvals_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        
        # Update results with corrected p-values
        for i, label in enumerate(labels):
            if results[label] is not None:
                results[label]['p_value_fdr'] = pvals_corrected[i]
                results[label]['significant_fdr'] = rejected[i]
        
        return results
    
    def create_boxplots(self, results, output_dir=None):
        """Create boxplots for each metric using paired data CSV files with linked points between conditions"""
        if results is None:
            return
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define custom color palette: Orange for Rejected, Teal for Preferred
        custom_palette = ['#E69F00', '#009E73']  # Orange for Rejected, Teal for Preferred
        
        # Define the metrics to plot
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        
        # Define chooser columns for both user and model
        chooser_cols = [
            ('user', 'chosen_user'),
            ('model', 'chosen_model')
        ]
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        for metric_name, column_name in metrics.items():
            for chooser_key, chooser_col in chooser_cols:
                # Read the paired data CSV file from reading_metric_statistics directory
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"synthetic_paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                
                paired_df = pd.read_csv(csv_filename)
                
                if paired_df.shape[0] == 0:
                    continue

                # Prepare long-form data for seaborn
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Trial': list(paired_df['trial']) + list(paired_df['trial'])
                })

                # Sample sizes
                n_preferred = len(paired_df)
                n_rejected = len(paired_df)

                # Create figure and axis
                fig, ax = plt.subplots(1, 1, figsize=(8, 7))

                # Create boxplot
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, order=['Rejected', 'Preferred'])

                # Add individual points and connecting lines
                for i, row in paired_df.iterrows():
                    rejected_val = row['Rejected']
                    preferred_val = row['Preferred']
                    
                    # Plot individual points
                    ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                    ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                    
                    # Connect points with line - color based on direction
                    if preferred_val > rejected_val:
                        # Preferred > Rejected: green line
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=1)
                    else:
                        # Preferred < Rejected: red line
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=1)

                # Configure titles and labels
                ax.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)")
                ax.set_xlabel("Condition")
                ax.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})")
                ax.set_xticklabels([f"Rejected\n(n={n_rejected})", f"Preferred\n(n={n_preferred})"])

                plt.tight_layout()

                # Save the linked plot
                plot_filename = os.path.join(output_dir, f'{metric_name}_{chooser_key}_boxplot_linked.png')
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

                # Create and save non-linked version (boxplot only, no individual points or lines)
                fig_no_links, ax_no_links = plt.subplots(1, 1, figsize=(8, 7))
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax_no_links, palette=custom_palette, order=['Rejected', 'Preferred'])
                
                # Configure titles and labels for non-linked version
                ax_no_links.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)")
                ax_no_links.set_xlabel("Condition")
                ax_no_links.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})")
                ax_no_links.set_xticklabels([f"Rejected\n(n={n_rejected})", f"Preferred\n(n={n_preferred})"])
                
                plt.tight_layout()
                
                # Save the non-linked plot
                plot_filename_no_links = os.path.join(output_dir, f'{metric_name}_{chooser_key}_boxplot.png')
                fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
                plt.close(fig_no_links)
                
                plt.close(fig)

    def _get_metric_unit(self, metric_name):
        """Get the appropriate unit for each metric"""
        units = {
            'FFD': 'ms',
            'TRT': 'ms', 
            'nFIX': 'count',
            'GPT': 'ms'
        }
        return units.get(metric_name, '')
    
    def create_combined_boxplot(self, results, output_dir=None):
        """Create combined boxplots for all metrics using paired data CSV files with linked points"""
        if results is None:
            return
        
        # Set default output directory if not provided
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define custom color palette: Orange for Rejected, Teal for Preferred
        custom_palette = ['#E69F00', '#009E73']  # Orange for Rejected, Teal for Preferred
        
        # Define the metrics to plot
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        
        # Define chooser columns for both user and model
        chooser_cols = [
            ('user', 'chosen_user'),
            ('model', 'chosen_model')
        ]
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        for chooser_key, chooser_col in chooser_cols:
            # Create figure with subplots arranged horizontally
            fig, axes = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(f'{chooser_key.capitalize()}-Based Preferences (Synthetic)', fontsize=18, fontweight='bold')
            
            for idx, (metric_name, column_name) in enumerate(metrics.items()):
                # Read the paired data CSV file
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"synthetic_paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                
                paired_df = pd.read_csv(csv_filename)
                
                if paired_df.shape[0] == 0:
                    continue

                # Prepare long-form data for seaborn
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Trial': list(paired_df['trial']) + list(paired_df['trial'])
                })

                # Create boxplot on the current subplot
                ax = axes[idx]
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                           order=['Rejected', 'Preferred'], width=0.5)

                # Add individual points and connecting lines
                for i, row in paired_df.iterrows():
                    rejected_val = row['Rejected']
                    preferred_val = row['Preferred']
                    
                    # Plot individual points
                    ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=3)
                    ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=3)
                    
                    # Connect points with line - color based on direction
                    if preferred_val > rejected_val:
                        # Preferred > Rejected: green line
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=0.8)
                    else:
                        # Preferred < Rejected: red line
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=0.8)

                # Configure titles and labels with larger font sizes
                ax.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)", fontsize=14, fontweight='bold')
                ax.set_xlabel("", fontsize=12)
                ax.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})", fontsize=12)
                
                # Increase tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
                # Add padding to move x-axis labels further from the axes
                ax.tick_params(axis='x', pad=15)

            plt.tight_layout()

            # Save the linked plot
            plot_filename = os.path.join(output_dir, f'combined_reading_measures_{chooser_key}_boxplot_linked.png')
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

            # Create and save non-linked version (boxplots only, no individual points or lines)
            fig_no_links, axes_no_links = plt.subplots(1, 4, figsize=(16, 6))
            fig_no_links.suptitle(f'{chooser_key.capitalize()}-Based Preferences (Synthetic)', fontsize=18, fontweight='bold')
            
            for idx, (metric_name, column_name) in enumerate(metrics.items()):
                # Read the paired data CSV file
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"synthetic_paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                
                paired_df = pd.read_csv(csv_filename)
                
                if paired_df.shape[0] == 0:
                    continue

                # Prepare long-form data for seaborn
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Trial': list(paired_df['trial']) + list(paired_df['trial'])
                })

                # Create boxplot on the current subplot (no links)
                ax = axes_no_links[idx]
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                           order=['Rejected', 'Preferred'], width=0.5)

                # Configure titles and labels with larger font sizes
                ax.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)", fontsize=14, fontweight='bold')
                ax.set_xlabel("", fontsize=12)
                ax.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})", fontsize=12)
                
                # Increase tick label font sizes
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
                ax.tick_params(axis='x', pad=15)

            plt.tight_layout()
            
            # Save the non-linked plot
            plot_filename_no_links = os.path.join(output_dir, f'combined_reading_measures_{chooser_key}_boxplot.png')
            fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
            plt.close(fig_no_links)
            
            plt.close(fig)
        
    
    def save_summary_statistics(self, filename=None, results=None, words_results=None):
        """Save summary statistics to a text file"""
        if self.aggregate_data is None:
            return
        
        # Set default filename if not provided
        if filename is None:
            filename = os.path.join(self.output_dir, "reading_metric_statistics", "summary_statistics.txt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== SYNTHETIC DATA ANALYSIS SUMMARY ===\n")
            f.write(f"Total number of trials: {len(self.aggregate_data)}\n")
            f.write(f"Number of participants: {len(self.participants)}\n")
            f.write(f"User preferred responses: {len(self.aggregate_data[self.aggregate_data['chosen_user'] == True])}\n")
            f.write(f"User rejected responses: {len(self.aggregate_data[self.aggregate_data['chosen_user'] == False])}\n")
            f.write(f"Model preferred responses: {len(self.aggregate_data[self.aggregate_data['chosen_model'] == True])}\n")
            f.write(f"Model rejected responses: {len(self.aggregate_data[self.aggregate_data['chosen_model'] == False])}\n")
            
            # Write reading metrics statistical test results if available
            if results is not None:
                f.write("\n=== READING METRICS STATISTICAL TEST RESULTS (Trial-Level) ===\n")
                f.write("Comparison: Preferred vs. Rejected Responses (One-sample t-tests)\n\n")
                
                for metric_name, result in results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Number of trials: n={result['n_trials']}\n")
                        f.write(f"Preferred: mean={result['preferred_mean']:.2f}, std={result['preferred_std']:.2f}\n")
                        f.write(f"Rejected: mean={result['rejected_mean']:.2f}, std={result['rejected_std']:.2f}\n")
                        f.write(f"Mean difference (Preferred - Rejected): {result['difference_mean']:.4f}\n")
                        f.write(f"Analysis type: trial_level_one_sample_t_test\n")
                        f.write(f"{result['test_type']}: statistic={result['test_statistic']:.3f}, p={result['p_value']:.4f}\n")
                        if 'p_value_fdr' in result:
                            f.write(f"FDR-corrected p-value: {result['p_value_fdr']:.6f}\n")
                            f.write(f"Significant after FDR correction: {'Yes' if result.get('significant_fdr', False) else 'No'}\n")
                        if 'effect_size_name' in result and 'effect_size' in result:
                            f.write(f"{result['effect_size_name']}: {result['effect_size']:.3f} ({result['effect_interpretation']} effect)\n")
                        else:
                            f.write(f"Effect size: {result['effect_size']:.3f} ({result['effect_interpretation']} effect)\n")
                        f.write(f"Statistically significant: {'Yes' if result['is_significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient data for analysis.\n\n")
            
            
        


def main():
    """Main function to run the synthetic reading metrics analysis"""
    
    # Initialize analyzer
    analyzer = SyntheticMetricsAnalyzer(input_dir="users_synthetic", output_dir="users_synthetic")
    
    # Create synthetic data structure and aggregate metrics CSV
    analyzer.create_synthetic_data_structure()
    
    # Aggregate all data
    aggregate_data = analyzer.aggregate_all_data()
    
    if aggregate_data is not None:
        # Save aggregate data
        analyzer.save_aggregate_data()
        
        # Load synthetic data
        synthetic_data = analyzer.load_synthetic_data()
        
        if synthetic_data is not None:
            # Perform statistical analysis on reading metrics (Antonella only)
            results = analyzer.perform_reading_metrics_tests()
            
            # Create boxplots
            analyzer.create_boxplots(results)
            analyzer.create_combined_boxplot(results)
            
            # Apply FDR correction to each family separately (4 tests each)
            import copy
            user_results = {k: copy.deepcopy(v) for k, v in results.items() if k.endswith('_user')}
            user_results_fdr = analyzer.apply_fdr_correction(user_results, alpha=0.05)
            
            model_results = {k: copy.deepcopy(v) for k, v in results.items() if k.endswith('_model')}
            model_results_fdr = analyzer.apply_fdr_correction(model_results, alpha=0.05)
            
            # Combine all FDR-corrected results
            results_fdr = {}
            if user_results_fdr:
                results_fdr.update(user_results_fdr)
            if model_results_fdr:
                results_fdr.update(model_results_fdr)
            
            # Save summary statistics
            analyzer.save_summary_statistics(None, results_fdr, None)
        else:
            pass
    else:
        pass


if __name__ == "__main__":
    main()
