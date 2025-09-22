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

class MetricsAnalyzer:
    def __init__(self, input_dir="users", output_dir="users"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.users_dir = input_dir  # Keep for backward compatibility
        self.aggregate_data = None
        self.participants = []
        
    def get_participant_directories(self):
        """Get all participant directories"""
        participant_dirs = []
        for item in os.listdir(self.users_dir):
            item_path = os.path.join(self.users_dir, item)
            if os.path.isdir(item_path) and item.startswith('participant_'):
                participant_dirs.append(item)
        return sorted(participant_dirs)
    
    def read_participant_data(self, participant_dir):
        """Read general_features.csv and info_extended.csv for a participant"""
        session_path = os.path.join(self.users_dir, participant_dir, "session_1")
        
        # Read general features
        general_features_path = os.path.join(session_path, "general_features.csv")
        if not os.path.exists(general_features_path):
            return None
            
        general_features = pd.read_csv(general_features_path, sep=';')
        
        # Read info extended to get the 'best' and 'discarted' columns
        info_extended_path = os.path.join(session_path, "info_extended.csv")
        if not os.path.exists(info_extended_path):
            return None
            
        info_extended = pd.read_csv(info_extended_path)
        
        # Extract participant name
        participant_name = participant_dir.replace('participant_', '').replace('_', ' ')
        
        # Merge the dataframes on trial column
        merged_data = pd.merge(general_features, info_extended[['trial', 'chosen_user', 'chosen_model', 'discarted']], 
                              on='trial', how='inner')
        
        # Add participant information
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
            # First, identify which trial bases have discarded trials
            discarded_trials = self.aggregate_data[self.aggregate_data['discarted'] == True]
            discarded_trial_bases = set()
            
            for _, row in discarded_trials.iterrows():
                trial_str = str(row['trial'])
                if '.' in trial_str:
                    base = trial_str.split('.')[0]
                    discarded_trial_bases.add(base)
            
            # Remove all trials (both .1 and .2) for trial bases that have any discarded trials
            self.aggregate_data['trial_base'] = self.aggregate_data['trial_str'].str.split('.').str[0]
            self.aggregate_data = self.aggregate_data[~self.aggregate_data['trial_base'].isin(discarded_trial_bases)]
            
            # Now verify that each participant has both .1 and .2 trials for each trial base
            # Group by participant_id and trial_base, then check if each group has exactly 2 trials
            participant_trial_counts = self.aggregate_data.groupby(['participant_id', 'trial_base']).size().reset_index(name='count')
            incomplete_pairs = participant_trial_counts[participant_trial_counts['count'] < 2]
            
            if not incomplete_pairs.empty:
                # Create a mask to remove incomplete pairs
                mask = self.aggregate_data.set_index(['participant_id', 'trial_base']).index.isin(
                    incomplete_pairs.set_index(['participant_id', 'trial_base']).index
                )
                self.aggregate_data = self.aggregate_data[~mask]
            
            # Remove the temporary columns
            self.aggregate_data = self.aggregate_data.drop(['trial_str', 'trial_base'], axis=1)
            
            # Convert time-based metrics from milliseconds to seconds
            time_metrics = ['first_fix_duration_mean', 'fix_duration_mean', 'go_past_time_mean']
            for metric in time_metrics:
                if metric in self.aggregate_data.columns:
                    # Try to detect if the values are in milliseconds (e.g., most values > 20)
                    col = self.aggregate_data[metric]
                    # Heuristic: if median > 20, likely ms; if < 10, likely already seconds
                    median_val = col.median()
                    if median_val > 20:
                        self.aggregate_data[metric] = col / 1000.0
            return self.aggregate_data
        else:
            return None
    
    def save_aggregate_data(self, filename="users/aggregate_metrics.csv"):
        """Save the aggregated data to CSV - DISABLED"""
        # This function is disabled - we don't want to save aggregate data
        pass
    
    def perform_paired_t_tests(self):
        """Perform paired t-tests or Wilcoxon tests on aggregated data per user (removes trial independence assumption).
        For each metric (FFD, TRT, nFIX, GPT):
        1. Aggregate data per user and condition (chosen/rejected)
        2. Calculate difference Chosen-Rejected for each user
        3. Test normality of differences (Shapiro)
        4. If normal → paired t-test (+ Cohen's d)
           If not normal → Wilcoxon
        """
        if self.aggregate_data is None:
            return None
        
        from scipy.stats import shapiro, wilcoxon,norm
        
        # Define the metrics to analyze
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
                
                # STEP 1 – Aggregate data per user and condition
                df = self.aggregate_data.copy()
                df['trial_base'] = df['trial'].astype(str).str.split('.').str[0]
                
                # Aggregate by participant_id and chosen/rejected status
                aggregated_df = df.groupby(['participant_id', chooser_col])[column_name].mean().reset_index()
                
                # Pivot to create "Preferred" and "Rejected" columns per user
                user_paired_df = aggregated_df.pivot_table(
                    index='participant_id',
                    columns=chooser_col,
                    values=column_name,
                    aggfunc='mean'
                ).reindex(columns=[False, True]).dropna()
                
                user_paired_df.columns = ['Rejected', 'Preferred']
                # Create directory if it doesn't exist
                output_dir = os.path.join(self.output_dir, "reading_metric_statistics")
                os.makedirs(output_dir, exist_ok=True)
                user_paired_df.to_csv(os.path.join(output_dir, f"paired_data_{metric_name}_{chooser_col}.csv"), index=True)
                
                if len(user_paired_df) > 0:
                    # STEP 2 – Calculate difference Chosen-Rejected for each user
                    diff = user_paired_df['Preferred'] - user_paired_df['Rejected']
                    
                    # STEP 3 – Test normality of differences using Shapiro-Wilk
                    try:
                        shapiro_stat, shapiro_p = shapiro(diff)
                        is_normal = shapiro_p > 0.05  # If p > 0.05, we fail to reject normality
                    except Exception as e:
                        is_normal = False  # Default to non-parametric if test fails
                    
                    # STEP 4 – Choose appropriate test based on normality
                    if is_normal and len(diff) >= 3:  # Need at least 3 observations for t-test
                        # Use paired t-test
                        test_type = "Paired t-test (aggregated)"
                        t_stat, p_value = stats.ttest_rel(user_paired_df['Preferred'], user_paired_df['Rejected'])
                        test_statistic = t_stat
                    else:
                        # Use Wilcoxon signed-rank test
                        test_type = "Wilcoxon signed-rank test (aggregated)"
                        try:
                            wilcoxon_stat, p_value = wilcoxon(diff, alternative='two-sided')
                            test_statistic = wilcoxon_stat
                        except ValueError as e:
                            results[label] = None
                            continue
                    
                    # Calculate appropriate effect size based on test type
                    if test_type == "Paired t-test (aggregated)":
                        # For paired t-test: use Hedges' g correction
                        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
                        if not np.isnan(cohens_d):
                            # Apply Hedges' g correction: g = d * J, where J = 1 - 3/(4*df-1), df = n-1
                            df = len(diff) - 1
                            J = 1 - 3/(4*df - 1) if df > 0 else 1
                            effect_size = cohens_d * J
                            effect_size_name = "Hedges' g"
                        else:
                            effect_size = np.nan
                            effect_size_name = "Hedges' g"
                    else:
                        # For Wilcoxon: use Rosenthal's r
                        N = len(diff)
                        if p_value > 0 and N > 0:
                            Z = norm.isf(p_value / 2.0) * np.sign(diff.mean())
                            effect_size = Z / np.sqrt(N)
                        else:
                            effect_size = np.nan
                        effect_size_name = "Rosenthal's r"
                    # Determine effect size interpretation
                    if np.isnan(effect_size):
                        effect_size_interpretation = "undefined"
                    elif test_type == "Paired t-test (aggregated)":
                        # Hedges' g interpretation (same as Cohen's d)
                        if abs(effect_size) < 0.2:
                            effect_size_interpretation = "negligible"
                        elif abs(effect_size) < 0.5:
                            effect_size_interpretation = "small"
                        elif abs(effect_size) < 0.8:
                            effect_size_interpretation = "medium"
                        else:
                            effect_size_interpretation = "large"
                    else:
                        # Rosenthal's r interpretation
                        if abs(effect_size) < 0.1:
                            effect_size_interpretation = "negligible"
                        elif abs(effect_size) < 0.3:
                            effect_size_interpretation = "small"
                        elif abs(effect_size) < 0.5:
                            effect_size_interpretation = "medium"
                        else:
                            effect_size_interpretation = "large"
                    
                    results[label] = {
                        'preferred_mean': user_paired_df['Preferred'].mean(),
                        'preferred_std': user_paired_df['Preferred'].std(),
                        'rejected_mean': user_paired_df['Rejected'].mean(),
                        'rejected_std': user_paired_df['Rejected'].std(),
                        'paired_n': len(user_paired_df),
                        'test_type': test_type,
                        'test_statistic': test_statistic,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'effect_size_name': effect_size_name,
                        'cohens_d': cohens_d if test_type == "Paired t-test (aggregated)" else np.nan,
                        'effect_size_interpretation': effect_size_interpretation,
                        'significant': p_value < 0.05,
                        'mean_difference': diff.mean(),
                        'std_difference': diff.std(ddof=1),
                        'shapiro_stat': shapiro_stat if 'shapiro_stat' in locals() else np.nan,
                        'shapiro_p': shapiro_p if 'shapiro_p' in locals() else np.nan,
                        'is_normal': is_normal,
                        'analysis_type': 'aggregated_per_user'
                    }
                else:
                    results[label] = None
        
        return results



    def apply_fdr_correction(self, results, alpha=0.05):
        """Apply False Discovery Rate (FDR) correction using Benjamini-Hochberg procedure.
        
        Args:
            results (dict): Dictionary of test results with 'p_value' keys.
            alpha (float): Significance level (default 0.05).
        
        Returns:
            dict: Original results plus FDR-corrected p-values and significance flags.
        """
        if results is None:
            return None

        # Extract p-values and test names
        p_values, test_names = [], []
        for test_name, result in results.items():
            if result is not None and 'p_value' in result:
                p_values.append(result['p_value'])
                test_names.append(test_name)

        if not p_values:
            return results

        # Convert to numpy arrays
        p_values = np.array(p_values)
        test_names = np.array(test_names)

        # Sort p-values and get indices
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]
        sorted_names = test_names[sorted_idx]

        n_tests = len(sorted_p)
        ranks = np.arange(1, n_tests + 1)
        critical_values = (ranks / n_tests) * alpha

        # Determine the largest k meeting the BH condition
        meets_condition = sorted_p <= critical_values
        if np.any(meets_condition):
            k_max = np.max(np.where(meets_condition)[0])
            significant_mask = np.zeros_like(sorted_p, dtype=bool)
            significant_mask[:k_max + 1] = True
        else:
            significant_mask = np.zeros_like(sorted_p, dtype=bool)

        # Compute BH adjusted p-values (monotonic step-up)
        bh_adj = sorted_p * n_tests / ranks
        bh_adj = np.minimum.accumulate(bh_adj[::-1])[::-1]  # enforce monotonicity
        bh_adj = np.minimum(bh_adj, 1.0)

        # Map adjusted results back to original test names
        name_to_fdr = {
            name: {
                'adjusted_p_value': bh_adj[i],
                'significant_fdr': significant_mask[i],
                'rank': ranks[i],
                'critical_value': critical_values[i]
            }
            for i, name in enumerate(sorted_names)
        }

        # Apply FDR results to the original dictionary
        corrected_results = results.copy()
        for test_name, result in corrected_results.items():
            if result is not None and test_name in name_to_fdr:
                fdr_info = name_to_fdr[test_name]
                result['p_value_fdr_corrected'] = fdr_info['adjusted_p_value']
                result['significant_fdr'] = fdr_info['significant_fdr']
                result['fdr_rank'] = fdr_info['rank']
                result['fdr_critical_value'] = fdr_info['critical_value']

        return corrected_results




    def perform_paired_t_tests_words(self):
        """Perform paired t-tests or Wilcoxon tests on aggregated data per user for words metrics (WORDS, AWL).
        For each metric (WORDS, AWL):
        1. Aggregate data per user and condition (chosen/rejected)
        2. Calculate difference Chosen-Rejected for each user
        3. Test normality of differences (Shapiro)
        4. If normal → paired t-test (+ Cohen's d)
           If not normal → Wilcoxon
        """
        if self.aggregate_data is None:
            return None

        from scipy.stats import shapiro, wilcoxon,norm

        metrics = {
            'WORDS': 'number_words_mean',
            'AWL': 'avg_word_length_mean'
        }

        chooser_cols = [
            ('user', 'chosen_user')
        ]

        results = {}

        for metric_name, column_name in metrics.items():
            if column_name not in self.aggregate_data.columns:
                continue

            for chooser_key, chooser_col in chooser_cols:
                label = f"{metric_name}_{chooser_key}"

                # STEP 1 – Aggregate data per user and condition
                df = self.aggregate_data.copy()
                df['trial_base'] = df['trial'].astype(str).str.split('.').str[0]

                # Aggregate by participant_id and chosen/rejected status
                aggregated_df = df.groupby(['participant_id', chooser_col])[column_name].mean().reset_index()

                # Pivot to create "Preferred" and "Rejected" columns per user
                user_paired_df = aggregated_df.pivot_table(
                    index='participant_id',
                    columns=chooser_col,
                    values=column_name,
                    aggfunc='mean'
                ).reindex(columns=[False, True]).dropna()

                user_paired_df.columns = ['Rejected', 'Preferred']
                # Create directory if it doesn't exist
                output_dir = os.path.join(self.output_dir, "reading_metric_statistics")
                os.makedirs(output_dir, exist_ok=True)
                user_paired_df.to_csv(os.path.join(output_dir, f"paired_data_{metric_name}_{chooser_col}.csv"), index=True)

                if len(user_paired_df) > 0:
                    # STEP 2 – Calculate difference Chosen-Rejected for each user
                    diff = user_paired_df['Preferred'] - user_paired_df['Rejected']

                    # STEP 3 – Test normality of differences using Shapiro-Wilk
                    try:
                        shapiro_stat, shapiro_p = shapiro(diff)
                        is_normal = shapiro_p > 0.05  # If p > 0.05, we fail to reject normality
                    except Exception as e:
                        is_normal = False  # Default to non-parametric if test fails

                    # STEP 4 – Choose appropriate test based on normality
                    if is_normal and len(diff) >= 3:  # Need at least 3 observations for t-test
                        # Use paired t-test
                        test_type = "Paired t-test (aggregated)"
                        t_stat, p_value = stats.ttest_rel(user_paired_df['Preferred'], user_paired_df['Rejected'])
                        test_statistic = t_stat
                    else:
                        # Use Wilcoxon signed-rank test
                        test_type = "Wilcoxon signed-rank test (aggregated)"
                        try:
                            wilcoxon_stat, p_value = wilcoxon(diff)
                            test_statistic = wilcoxon_stat
                        except ValueError as e:
                            results[label] = None
                            continue

                    # Calculate appropriate effect size based on test type
                    if test_type == "Paired t-test (aggregated)":
                        # For paired t-test: use Hedges' g correction
                        cohens_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
                        if not np.isnan(cohens_d):
                            # Apply Hedges' g correction: g = d * J, where J = 1 - 3/(4*df-1), df = n-1
                            df = len(diff) - 1
                            J = 1 - 3/(4*df - 1) if df > 0 else 1
                            effect_size = cohens_d * J
                            effect_size_name = "Hedges' g"
                        else:
                            effect_size = np.nan
                            effect_size_name = "Hedges' g"
                    else:
                        # For Wilcoxon: use Rosenthal's r
                        # Rosenthal's r = Z / sqrt(N), where Z = W / sqrt(N*(N+1)*(2*N+1)/6)
                        N = len(diff)
                        if p_value > 0 and N > 0:
                            Z = norm.isf(p_value / 2.0) * np.sign(diff.mean())
                            effect_size = Z / np.sqrt(N)
                        else:
                            effect_size = np.nan
                            effect_size_name = "Rosenthal's r"

                    # Determine effect size interpretation
                    if np.isnan(effect_size):
                        effect_size_interpretation = "undefined"
                    elif test_type == "Paired t-test (aggregated)":
                        # Hedges' g interpretation (same as Cohen's d)
                        if abs(effect_size) < 0.2:
                            effect_size_interpretation = "negligible"
                        elif abs(effect_size) < 0.5:
                            effect_size_interpretation = "small"
                        elif abs(effect_size) < 0.8:
                            effect_size_interpretation = "medium"
                        else:
                            effect_size_interpretation = "large"
                    else:
                        # Rosenthal's r interpretation
                        if abs(effect_size) < 0.1:
                            effect_size_interpretation = "negligible"
                        elif abs(effect_size) < 0.3:
                            effect_size_interpretation = "small"
                        elif abs(effect_size) < 0.5:
                            effect_size_interpretation = "medium"
                        else:
                            effect_size_interpretation = "large"

                    results[label] = {
                        'preferred_mean': user_paired_df['Preferred'].mean(),
                        'preferred_std': user_paired_df['Preferred'].std(),
                        'rejected_mean': user_paired_df['Rejected'].mean(),
                        'rejected_std': user_paired_df['Rejected'].std(),
                        'paired_n': len(user_paired_df),
                        'test_type': test_type,
                        'test_statistic': test_statistic,
                        'p_value': p_value,
                        'effect_size': effect_size,
                        'effect_size_name': effect_size_name,
                        'cohens_d': cohens_d if test_type == "Paired t-test (aggregated)" else np.nan,
                        'effect_size_interpretation': effect_size_interpretation,
                        'significant': p_value < 0.05,
                        'mean_difference': diff.mean(),
                        'std_difference': diff.std(ddof=1),
                        'shapiro_stat': shapiro_stat if 'shapiro_stat' in locals() else np.nan,
                        'shapiro_p': shapiro_p if 'shapiro_p' in locals() else np.nan,
                        'is_normal': is_normal,
                        'analysis_type': 'aggregated_per_user'
                    }
                else:
                    results[label] = None

        return results
    

    

    
    def compute_word_count_correlations(self):
        """Compute Pearson and Spearman correlations between number of words and reading metrics.
        Prints r with 95% CI as ± margin instead of p-values.
        """
        if self.aggregate_data is None:
            return None

        from scipy.stats import pearsonr, spearmanr, norm

        word_col = 'number_words_mean'
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean',
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }

        results = {}

        for metric_name, column_name in metrics.items():
            if word_col in self.aggregate_data.columns and column_name in self.aggregate_data.columns:
                x = self.aggregate_data[word_col]
                y = self.aggregate_data[column_name]

                # Remove NaNs
                mask = x.notna() & y.notna()
                x_valid = x[mask]
                y_valid = y[mask]

                if len(x_valid) < 2:
                    results[metric_name] = None
                    continue

                # Pearson
                pearson_corr, pearson_p = pearsonr(x_valid, y_valid)
                # 95% CI via Fisher z-transform
                n = len(x_valid)
                if n > 3 and np.isfinite(pearson_corr) and abs(pearson_corr) < 1:
                    z = np.arctanh(pearson_corr)
                    se = 1.0 / np.sqrt(n - 3)
                    z_crit = norm.ppf(1 - 0.05 / 2)
                    z_lo, z_hi = z - z_crit * se, z + z_crit * se
                    pearson_ci_low, pearson_ci_high = np.tanh([z_lo, z_hi])
                    pearson_margin = (pearson_ci_high - pearson_ci_low) / 2.0
                else:
                    pearson_ci_low = pearson_ci_high = pearson_margin = np.nan

                # Spearman (approximate CI via Fisher z on rho)
                spearman_corr, spearman_p = spearmanr(x_valid, y_valid)
                if n > 3 and np.isfinite(spearman_corr) and abs(spearman_corr) < 1:
                    z = np.arctanh(spearman_corr)
                    se = 1.0 / np.sqrt(n - 3)
                    z_crit = norm.ppf(1 - 0.05 / 2)
                    z_lo, z_hi = z - z_crit * se, z + z_crit * se
                    spearman_ci_low, spearman_ci_high = np.tanh([z_lo, z_hi])
                    spearman_margin = (spearman_ci_high - spearman_ci_low) / 2.0
                else:
                    spearman_ci_low = spearman_ci_high = spearman_margin = np.nan

                results[metric_name] = {
                    'pearson_r': pearson_corr,
                    'pearson_ci_low': pearson_ci_low,
                    'pearson_ci_high': pearson_ci_high,
                    'spearman_rho': spearman_corr,
                    'spearman_ci_low': spearman_ci_low,
                    'spearman_ci_high': spearman_ci_high,
                    'n': n
                }
            else:
                results[metric_name] = None

        return results

    def compute_avg_word_length_correlations(self):
        """Compute Pearson and Spearman correlations between avg_word_length_mean and reading metrics (user-only data).
        Prints r with 95% CI as ± margin instead of p-values.
        """
        if self.aggregate_data is None:
            return None

        from scipy.stats import pearsonr, spearmanr, norm

        word_col = 'avg_word_length_mean'
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean',
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }

        results = {}

        for metric_name, column_name in metrics.items():
            if word_col in self.aggregate_data.columns and column_name in self.aggregate_data.columns:
                x = self.aggregate_data[word_col]
                y = self.aggregate_data[column_name]

                mask = x.notna() & y.notna()
                x_valid = x[mask]
                y_valid = y[mask]

                if len(x_valid) < 2:
                    results[metric_name] = None
                    continue

                pearson_corr, pearson_p = pearsonr(x_valid, y_valid)
                n = len(x_valid)
                if n > 3 and np.isfinite(pearson_corr) and abs(pearson_corr) < 1:
                    z = np.arctanh(pearson_corr)
                    se = 1.0 / np.sqrt(n - 3)
                    z_crit = norm.ppf(1 - 0.05 / 2)
                    z_lo, z_hi = z - z_crit * se, z + z_crit * se
                    pearson_ci_low, pearson_ci_high = np.tanh([z_lo, z_hi])
                    pearson_margin = (pearson_ci_high - pearson_ci_low) / 2.0
                else:
                    pearson_ci_low = pearson_ci_high = pearson_margin = np.nan

                spearman_corr, spearman_p = spearmanr(x_valid, y_valid)
                if n > 3 and np.isfinite(spearman_corr) and abs(spearman_corr) < 1:
                    z = np.arctanh(spearman_corr)
                    se = 1.0 / np.sqrt(n - 3)
                    z_crit = norm.ppf(1 - 0.05 / 2)
                    z_lo, z_hi = z - z_crit * se, z + z_crit * se
                    spearman_ci_low, spearman_ci_high = np.tanh([z_lo, z_hi])
                    spearman_margin = (spearman_ci_high - spearman_ci_low) / 2.0
                else:
                    spearman_ci_low = spearman_ci_high = spearman_margin = np.nan

                results[metric_name] = {
                    'pearson_r': pearson_corr,
                    'pearson_ci_low': pearson_ci_low,
                    'pearson_ci_high': pearson_ci_high,
                    'spearman_rho': spearman_corr,
                    'spearman_ci_low': spearman_ci_low,
                    'spearman_ci_high': spearman_ci_high,
                    'n': n
                }
            else:
                results[metric_name] = None

        return results

    
    
    def print_summary_statistics(self):
        """Print summary statistics for the aggregated data"""
        if self.aggregate_data is None:
            return
        
        # No print statements as per instruction
    
    def save_summary_statistics(self, filename=None, results=None, words_results=None, 
                               correlation_results=None, awl_correlation_results=None,
                               lme_llm_results=None, lme_words_results=None, clustering_results=None, antonella_model_results=None):
        """Save summary statistics to a text file (all output in English)"""
        if self.aggregate_data is None:
            return
        
        # Set default filename if not provided
        if filename is None:
            filename = os.path.join(self.output_dir, "reading_metric_statistics", "summary_statistics.txt")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=== SUMMARY STATISTICS ===\n")
            f.write(f"Total number of trials: {len(self.aggregate_data)}\n")
            f.write(f"Number of participants: {len(self.participants)}\n")
            
            # User choices
            f.write(f"User preferred responses: {len(self.aggregate_data[self.aggregate_data['chosen_user'] == True])}\n")
            f.write(f"User rejected responses: {len(self.aggregate_data[self.aggregate_data['chosen_user'] == False])}\n")
            
            # Model choices
            f.write(f"Model preferred responses: {len(self.aggregate_data[self.aggregate_data['chosen_model'] == True])}\n")
            f.write(f"Model rejected responses: {len(self.aggregate_data[self.aggregate_data['chosen_model'] == False])}\n")
            

            # Write correlation results if available
            if correlation_results is not None:
                f.write("\n=== CORRELATION RESULTS ===\n")
                f.write("Correlation between number_words_mean and reading metrics.\n\n")
                
                for metric_name, result in correlation_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Sample size: n={result['n']}\n")
                        
                        # Pearson correlation
                        if not np.isnan(result['pearson_r']):
                            pearson_margin = (result['pearson_ci_high'] - result['pearson_ci_low']) / 2.0 if not np.isnan(result['pearson_ci_high']) else np.nan
                            f.write(f"Pearson correlation: r = {result['pearson_r']:.3f}")
                            if not np.isnan(pearson_margin):
                                f.write(f" ± {pearson_margin:.3f} (95% CI)")
                            f.write("\n")
                        else:
                            f.write("Pearson correlation: insufficient data\n")
                        
                        # Spearman correlation
                        if not np.isnan(result['spearman_rho']):
                            spearman_margin = (result['spearman_ci_high'] - result['spearman_ci_low']) / 2.0 if not np.isnan(result['spearman_ci_high']) else np.nan
                            f.write(f"Spearman correlation: rho = {result['spearman_rho']:.3f}")
                            if not np.isnan(spearman_margin):
                                f.write(f" ± {spearman_margin:.3f} (95% CI)")
                            f.write("\n")
                        else:
                            f.write("Spearman correlation: insufficient data\n")
                        f.write("\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient data for correlation analysis.\n\n")
            
            # Write average word length correlation results if available
            if awl_correlation_results is not None:
                f.write("\n=== AVERAGE WORD LENGTH CORRELATION RESULTS ===\n")
                f.write("Correlation between avg_word_length_mean and reading metrics.\n\n")
                
                for metric_name, result in awl_correlation_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Sample size: n={result['n']}\n")
                        
                        # Pearson correlation
                        if not np.isnan(result['pearson_r']):
                            pearson_margin = (result['pearson_ci_high'] - result['pearson_ci_low']) / 2.0 if not np.isnan(result['pearson_ci_high']) else np.nan
                            f.write(f"Pearson correlation: r = {result['pearson_r']:.3f}")
                            if not np.isnan(pearson_margin):
                                f.write(f" ± {pearson_margin:.3f} (95% CI)")
                            f.write("\n")
                        else:
                            f.write("Pearson correlation: insufficient data\n")
                        
                        # Spearman correlation
                        if not np.isnan(result['spearman_rho']):
                            spearman_margin = (result['spearman_ci_high'] - result['spearman_ci_low']) / 2.0 if not np.isnan(result['spearman_ci_high']) else np.nan
                            f.write(f"Spearman correlation: rho = {result['spearman_rho']:.3f}")
                            if not np.isnan(spearman_margin):
                                f.write(f" ± {spearman_margin:.3f} (95% CI)")
                            f.write("\n")
                        else:
                            f.write("Spearman correlation: insufficient data\n")
                        f.write("\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient data for correlation analysis.\n\n")

            # Write statistical test results if available
            if results is not None:
                f.write("\n=== STATISTICAL TEST RESULTS ===\n")
                f.write("Test selection between paired t-test and Wilcoxon signed-rank test is based on the Shapiro-Wilk normality test.\n\n")
                f.write("Comparison: Preferred vs. Rejected Responses\n\n")
                
                for metric_name, result in results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Number of paired observations: n={result['paired_n']}\n")
                        f.write(f"Preferred: mean={result['preferred_mean']:.2f}, std={result['preferred_std']:.2f}")
                        if 'preferred_median' in result:
                            f.write(f", median={result['preferred_median']:.2f}")
                        f.write("\n")
                        f.write(f"Rejected: mean={result['rejected_mean']:.2f}, std={result['rejected_std']:.2f}")
                        if 'rejected_median' in result:
                            f.write(f", median={result['rejected_median']:.2f}")
                        f.write("\n")
                        f.write(f"Mean difference (Preferred - Rejected): {result['mean_difference']:.3f}")
                        if 'median_difference' in result:
                            f.write(f", Median difference: {result['median_difference']:.3f}")
                        f.write("\n")
                        f.write(f"Analysis type: {result['analysis_type']}\n")
                        f.write(f"Shapiro-Wilk normality test: W={result['shapiro_stat']:.3f}, p={result['shapiro_p']:.4f}\n")
                        f.write(f"Data normality: {'Normal' if result['is_normal'] else 'Non-normal'}\n")
                        f.write(f"{result['test_type']}: statistic={result['test_statistic']:.3f}, p={result['p_value']:.4f}\n")
                        if 'p_value_fdr_corrected' in result:
                            f.write(f"FDR-corrected p-value: {result['p_value_fdr_corrected']:.6f}\n")
                            f.write(f"Significant after FDR correction: {'Yes' if result.get('significant_fdr', False) else 'No'}\n")
                        if 'effect_size_name' in result and 'effect_size' in result:
                            f.write(f"{result['effect_size_name']}: {result['effect_size']:.3f} ({result['effect_size_interpretation']} effect)\n")
                        else:
                            f.write(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size_interpretation']} effect)\n")
                        f.write(f"Statistically significant: {'Yes' if result['significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient paired data for analysis.\n\n")
            
            # Write words statistical test results if available
            if words_results is not None:
                f.write("\n=== WORDS/AWL STATISTICAL TEST RESULTS ===\n")
                f.write("Comparison: Preferred vs. Rejected Responses (Automated Test Selection)\n\n")
                
                for metric_name, result in words_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Number of paired observations: n={result['paired_n']}\n")
                        f.write(f"Preferred: mean={result['preferred_mean']:.2f}, std={result['preferred_std']:.2f}")
                        if 'preferred_median' in result:
                            f.write(f", median={result['preferred_median']:.2f}")
                        f.write("\n")
                        f.write(f"Rejected: mean={result['rejected_mean']:.2f}, std={result['rejected_std']:.2f}")
                        if 'rejected_median' in result:
                            f.write(f", median={result['rejected_median']:.2f}")
                        f.write("\n")
                        f.write(f"Mean difference (Preferred - Rejected): {result['mean_difference']:.3f}")
                        if 'median_difference' in result:
                            f.write(f", Median difference: {result['median_difference']:.3f}")
                        f.write("\n")
                        f.write(f"Analysis type: {result['analysis_type']}\n")
                        f.write(f"Shapiro-Wilk normality test: W={result['shapiro_stat']:.3f}, p={result['shapiro_p']:.4f}\n")
                        f.write(f"Data normality: {'Normal' if result['is_normal'] else 'Non-normal'}\n")
                        f.write(f"{result['test_type']}: statistic={result['test_statistic']:.3f}, p={result['p_value']:.4f}\n")
                        if 'p_value_fdr_corrected' in result:
                            f.write(f"FDR-corrected p-value: {result['p_value_fdr_corrected']:.6f}\n")
                            f.write(f"Significant after FDR correction: {'Yes' if result.get('significant_fdr', False) else 'No'}\n")
                        if 'effect_size_name' in result and 'effect_size' in result:
                            f.write(f"{result['effect_size_name']}: {result['effect_size']:.3f} ({result['effect_size_interpretation']} effect)\n")
                        else:
                            f.write(f"Cohen's d: {result['cohens_d']:.3f} ({result['effect_size_interpretation']} effect)\n")
                        f.write(f"Statistically significant: {'Yes' if result['significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient paired data for analysis.\n\n")
            
            # Write Antonella model results if available
            if antonella_model_results is not None:
                f.write("\n=== ANTONELLA MODEL WORDS/AWL STATISTICAL TEST RESULTS ===\n")
                f.write("Comparison: Model Preferred vs. Rejected Responses (Antonella's Model)\n\n")
                
                for metric_name, result in antonella_model_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Number of trials: n={result['n_trials']}\n")
                        f.write(f"Preferred: mean={result['preferred_mean']:.2f}, std={result['preferred_std']:.2f}\n")
                        f.write(f"Rejected: mean={result['rejected_mean']:.2f}, std={result['rejected_std']:.2f}\n")
                        f.write(f"Mean difference (Preferred - Rejected): {result['difference_mean']:.4f}\n")
                        f.write(f"Analysis type: trial_level_one_sample_t_test\n")
                        f.write(f"{result['test_type']}: statistic={result['test_statistic']:.3f}, p={result['p_value']:.4f}\n")
                        if 'p_value_fdr_corrected' in result:
                            f.write(f"FDR-corrected p-value: {result['p_value_fdr_corrected']:.6f}\n")
                            f.write(f"Significant after FDR correction: {'Yes' if result.get('significant_fdr', False) else 'No'}\n")
                        if 'effect_size_name' in result and 'effect_size' in result:
                            f.write(f"{result['effect_size_name']}: {result['effect_size']:.3f} ({result['effect_interpretation']} effect)\n")
                        else:
                            f.write(f"Effect size: {result['effect_size']:.3f} ({result['effect_interpretation']} effect)\n")
                        f.write(f"Statistically significant: {'Yes' if result['is_significant'] else 'No'}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Insufficient data for analysis.\n\n")
            
            # Write linear mixed-effects results for LLM metrics if available
            if lme_llm_results is not None:
                f.write("\n=== LINEAR MIXED-EFFECTS MODELS (LLM METRICS) ===\n")
                f.write("Fixed effect: condition (preferred vs. rejected)\n")
                f.write("Random effect: subject\n\n")
                
                for metric_name, result in lme_llm_results.items():
                    if result is not None:
                        f.write(f"Condition coefficient: {result['coef_condition']:.4f}\n")
                        f.write(f"Standard error: {result['se_condition']:.4f}\n")
                        f.write(f"P-value: {result['p_value']:.6f}\n")
                        if 'p_value_fdr_corrected' in result:
                            f.write(f"FDR-corrected p-value: {result['p_value_fdr_corrected']:.6f}\n")
                            f.write(f"Significant after FDR correction: {'Yes' if result.get('significant_fdr', False) else 'No'}\n")
                        f.write(f"Effect size: {result['effect_size_d_resid']:.3f} ({result['effect_size_interpretation']})\n")
                        f.write(f"Model converged: {'Yes' if result['model_converged'] else 'No'}\n")
                        f.write(f"AIC: {result['aic']:.2f}, BIC: {result['bic']:.2f}\n")
                        f.write(f"Random effects variance: {result['random_effects_variance']:.4f}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Model fitting failed or insufficient data.\n\n")
            
            # Write linear mixed-effects results for words if available
            if lme_words_results is not None:
                f.write("\n=== LINEAR MIXED-EFFECTS MODELS (WORDS) ===\n")
                f.write("Fixed effect: condition (preferred vs. rejected)\n")
                f.write("Random effect: subject\n\n")
                
                for metric_name, result in lme_words_results.items():
                    if result is not None:
                        f.write(f"--- {metric_name} ---\n")
                        f.write(f"Condition coefficient: {result['coef_condition']:.4f}\n")
                        f.write(f"Standard error: {result['se_condition']:.4f}\n")
                        f.write(f"P-value: {result['p_value']:.6f}\n")
                        if 'p_value_fdr_corrected' in result:
                            f.write(f"FDR-corrected p-value: {result['p_value_fdr_corrected']:.6f}\n")
                            f.write(f"Significant after FDR correction: {'Yes' if result.get('significant_fdr', False) else 'No'}\n")
                        f.write(f"Effect size: {result['effect_size_d_resid']:.3f} ({result['effect_size_interpretation']})\n")
                        f.write(f"Model converged: {'Yes' if result['model_converged'] else 'No'}\n")
                        f.write(f"AIC: {result['aic']:.2f}, BIC: {result['bic']:.2f}\n")
                        f.write(f"Random effects variance: {result['random_effects_variance']:.4f}\n\n")
                    else:
                        f.write(f"--- {metric_name} ---\n")
                        f.write("Model fitting failed or insufficient data.\n\n")
            
            # Write clustering analysis results if available
            if clustering_results is not None:
                f.write("\n=== CLUSTERING ANALYSIS RESULTS ===\n")
                f.write("Strategy analysis based on metric differences (Rejected - Preferred).\n")
                f.write("Positive values = Rejected > Preferred (prefers model responses).\n")
                f.write("Negative values = Rejected < Preferred (prefers user responses).\n\n")
                
                # Overall best method
                f.write(f"Best overall clustering method: {clustering_results['best_overall']}\n")
                f.write(f"Best overall silhouette score: {clustering_results['all_scores'][clustering_results['best_overall']]:.3f}\n\n")
                
                # Silhouette scores comparison
                f.write("=== Silhouette Scores Comparison ===\n")
                for k in [2, 3]:
                    f.write(f"\n--- Results for k={k} ---\n")
                    f.write(f"KMeans: {clustering_results['silhouette_scores'][k]:.3f}\n")
                    f.write(f"Hierarchical Clustering: {clustering_results['hc_silhouette_scores'][k]:.3f}\n")
                    f.write(f"Gaussian Mixture Model: {clustering_results['gmm_silhouette_scores'][k]:.3f}\n")
                    
                    # Find best method for this k
                    best_method = max(clustering_results['silhouette_scores'][k], 
                                    clustering_results['hc_silhouette_scores'][k], 
                                    clustering_results['gmm_silhouette_scores'][k])
                    if best_method == clustering_results['silhouette_scores'][k]:
                        f.write(f"Best method for k={k}: KMeans (Silhouette = {best_method:.3f})\n")
                    elif best_method == clustering_results['hc_silhouette_scores'][k]:
                        f.write(f"Best method for k={k}: Hierarchical Clustering (Silhouette = {best_method:.3f})\n")
                    else:
                        f.write(f"Best method for k={k}: Gaussian Mixture Model (Silhouette = {best_method:.3f})\n")
                
                # Cluster assignments and centroids for best method
                f.write(f"\n=== Best Method Details: {clustering_results['best_overall']} ===\n")
                
                # Extract k value from best method name
                best_k = int(clustering_results['best_overall'].split('_k')[1])
                best_method_name = clustering_results['best_overall'].split('_k')[0]
                
                f.write(f"Number of clusters: {best_k}\n")
                f.write(f"Silhouette score: {clustering_results['all_scores'][clustering_results['best_overall']]:.3f}\n\n")
                
                # Get assignments for best method
                if best_method_name == "KMeans":
                    assignments = clustering_results['assignments'][best_k]
                    centroids = clustering_results['km_centroids'][best_k]
                elif best_method_name == "Hierarchical":
                    assignments = clustering_results['hc_assignments'][best_k]
                    # Hierarchical clustering doesn't have explicit centroids, so we'll calculate them
                    X = clustering_results['df'][clustering_results['metrics']].values
                    centroids = []
                    for cluster_id in range(best_k):
                        cluster_mask = assignments == cluster_id
                        cluster_centroid = X[cluster_mask].mean(axis=0)
                        centroids.append(cluster_centroid)
                    centroids = np.array(centroids)
                else:  # GMM
                    assignments = clustering_results['gmm_assignments'][best_k]
                    centroids = clustering_results['gmm_centroids'][best_k]
                
                # Write cluster details
                for cluster_id in range(best_k):
                    cluster_mask = assignments == cluster_id
                    cluster_users = clustering_results['df'].index[cluster_mask]
                    cluster_consistency = clustering_results['df'].iloc[cluster_mask]['consistency_%'].mean()
                    
                    f.write(f"Cluster {cluster_id}:\n")
                    f.write(f"  Number of users: {sum(cluster_mask)}\n")
                    f.write(f"  Users: {[u.replace('participant_', '').replace('_', ' ') for u in cluster_users]}\n")
                    f.write(f"  Average consistency: {cluster_consistency:.1f}%\n")
                    f.write(f"  Centroid values (Rejected - Preferred):\n")
                    for i, metric in enumerate(clustering_results['metrics']):
                        f.write(f"    {metric}: {centroids[cluster_id, i]:.4f}\n")
                    f.write("\n")
        
        # No print statement as per instruction
    

    def perform_linear_mixed_effects_llm_metrics(self):
        """Linear mixed-effects for FFD, TRT, nFIX, GPT with fixed effect=condition (preferred)
        and random intercept for participant. Applies FDR with apply_fdr_correction().
        """
        if self.aggregate_data is None:
            return None

        # --- imports local to keep this self-contained ---
        import numpy as np
        import pandas as pd
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            return None

        # metrics to analyze
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean',
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }

        # who defines "preferred"
        chooser_cols = [
            ('user', 'chosen_user'),
            ('model', 'chosen_model')
        ]

        results = {}

        for metric_name, column_name in metrics.items():
            if column_name not in self.aggregate_data.columns:
                continue

            for chooser_key, chooser_col in chooser_cols:
                if chooser_col not in self.aggregate_data.columns:
                    continue

                label = f"{metric_name}_{chooser_key}_LME"

                # prepare data
                df = self.aggregate_data[[column_name, chooser_col, 'participant_id']].dropna().copy()

                if df.empty:
                    results[label] = None
                    continue

                # robust boolean → int {0,1}
                if df[chooser_col].dtype != bool:
                    df['condition'] = df[chooser_col].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
                    # if still NaN (unexpected coding), coerce to 0/1 via truthiness
                    if df['condition'].isna().any():
                        df['condition'] = df[chooser_col].astype(bool).astype(int)
                else:
                    df['condition'] = df[chooser_col].astype(int)

                # keep only subjects with BOTH levels (0 and 1)
                has_both = df.groupby('participant_id')['condition'].nunique() == 2
                keep_ids = has_both[has_both].index
                n_drop = df['participant_id'].nunique() - len(keep_ids)
                df = df[df['participant_id'].isin(keep_ids)]

                if df.empty or df['participant_id'].nunique() < 2:
                    results[label] = None
                    continue

                # fit LMM: metric ~ condition + (1 | participant_id)
                try:
                    model = smf.mixedlm(
                        formula=f"{column_name} ~ condition",
                        data=df,
                        groups=df["participant_id"],
                        re_formula="~condition"   # << qui indichi che vuoi anche random slope
                    )

                    fit = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)

                    # named access (safer than [1])
                    beta = float(fit.fe_params.get('condition'))
                    se   = float(fit.bse.get('condition'))
                    pval = float(fit.pvalues.get('condition'))

                    # residual SD for standardization
                    # (statsmodels MixedLM: fit.scale = residual variance)
                    resid_sd = float(np.sqrt(fit.scale)) if np.isfinite(fit.scale) else np.nan
                    effect_size = beta / resid_sd if (resid_sd is not None and resid_sd > 0) else np.nan

                    if np.isnan(effect_size):
                        eff_interp = "undefined"
                    elif abs(effect_size) < 0.2:
                        eff_interp = "negligible"
                    elif abs(effect_size) < 0.5:
                        eff_interp = "small"
                    elif abs(effect_size) < 0.8:
                        eff_interp = "medium"
                    else:
                        eff_interp = "large"

                    # CI 95% (Wald)
                    ci = fit.conf_int().rename(columns={0: 'lo', 1: 'hi'})
                    lo = float(ci.loc['condition', 'lo'])
                    hi = float(ci.loc['condition', 'hi'])

                    # random-intercept variance
                    try:
                        re_var = float(fit.cov_re.iloc[0, 0])
                    except Exception:
                        re_var = np.nan

                    out = {
                        'metric': metric_name,
                        'chooser': chooser_key,
                        'column_name': column_name,
                        'n_observations': int(len(df)),
                        'n_participants': int(df['participant_id'].nunique()),
                        'n_preferred': int(df['condition'].sum()),
                        'n_rejected': int((df['condition'] == 0).sum()),
                        'coef_condition': beta,                  # Pref − Non-pref
                        'se_condition': se,
                        'z_condition': float(fit.tvalues.get('condition')),
                        'p_value': pval,                          # <-- key expected by apply_fdr_correction
                        'effect_size_d_resid': effect_size,       # standardized by residual SD
                        'effect_size_interpretation': eff_interp,
                        'ci95_lo_condition': lo,
                        'ci95_hi_condition': hi,
                        'aic': float(fit.aic),
                        'bic': float(fit.bic),
                        'log_likelihood': float(fit.llf),
                        'random_effects_variance': re_var,
                        'model_converged': bool(getattr(fit, "converged", True)),
                        'analysis_type': 'linear_mixed_effects'
                    }

                    results[label] = out
                except Exception as e:
                    results[label] = None

        # FDR correction across all models
        
        corrected = self.apply_fdr_correction(results, alpha=0.05)
        return corrected


    def perform_linear_mixed_effects_words(self):
        """Linear mixed-effects for WORDS & AWL with fixed effect=condition (preferred vs non)
        and random intercept for participant. Applies BH-FDR via apply_fdr_correction().
        """
        if self.aggregate_data is None:
            return None

        import numpy as np
        import pandas as pd
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            return None

        # metrics to analyze
        metrics = {
            'WORDS': 'number_words_mean',
            'AWL': 'avg_word_length_mean'
        }

        # "preference" source - only user choices
        chooser_cols = [
            ('user', 'chosen_user')
        ]

        results = {}

        for metric_name, column_name in metrics.items():
            if column_name not in self.aggregate_data.columns:
                continue

            for chooser_key, chooser_col in chooser_cols:
                if chooser_col not in self.aggregate_data.columns:
                    continue

                label = f"{metric_name}_{chooser_key}_LME"

                # prepare data
                df = self.aggregate_data[[column_name, chooser_col, 'participant_id']].dropna().copy()
                if df.empty:
                    results[label] = None
                    continue

                # robust boolean -> int {0,1}
                if df[chooser_col].dtype != bool:
                    df['condition'] = df[chooser_col].map({True: 1, False: 0, 'True': 1, 'False': 0, 1: 1, 0: 0})
                    if df['condition'].isna().any():
                        df['condition'] = df[chooser_col].astype(bool).astype(int)
                else:
                    df['condition'] = df[chooser_col].astype(int)

                # keep only subjects with BOTH levels (0 and 1)
                has_both = df.groupby('participant_id')['condition'].nunique() == 2
                keep_ids = has_both[has_both].index
                n_drop = df['participant_id'].nunique() - len(keep_ids)
                df = df[df['participant_id'].isin(keep_ids)]

                if df.empty or df['participant_id'].nunique() < 2:
                    results[label] = None
                    continue

                # fit LMM: metric ~ condition + (1 | participant_id)
                try:
                    model = smf.mixedlm(
                        formula=f"{column_name} ~ condition",
                        data=df,
                        groups=df["participant_id"]
                    )
                    fit = model.fit(reml=True, method="lbfgs", maxiter=200, disp=False)

                    beta = float(fit.fe_params.get('condition'))
                    se   = float(fit.bse.get('condition'))
                    pval = float(fit.pvalues.get('condition'))

                    # standardize by residual SD from the model
                    resid_sd = float(np.sqrt(fit.scale)) if np.isfinite(fit.scale) else np.nan
                    effect_size = beta / resid_sd if (resid_sd is not None and resid_sd > 0) else np.nan
                    if np.isnan(effect_size):
                        eff_interp = "undefined"
                    elif abs(effect_size) < 0.2:
                        eff_interp = "negligible"
                    elif abs(effect_size) < 0.5:
                        eff_interp = "small"
                    elif abs(effect_size) < 0.8:
                        eff_interp = "medium"
                    else:
                        eff_interp = "large"

                    ci = fit.conf_int().rename(columns={0: 'lo', 1: 'hi'})
                    lo = float(ci.loc['condition', 'lo'])
                    hi = float(ci.loc['condition', 'hi'])

                    try:
                        re_var = float(fit.cov_re.iloc[0, 0])
                    except Exception:
                        re_var = np.nan

                    out = {
                        'metric': metric_name,
                        'chooser': chooser_key,
                        'column_name': column_name,
                        'n_observations': int(len(df)),
                        'n_participants': int(df['participant_id'].nunique()),
                        'n_preferred': int(df['condition'].sum()),
                        'n_rejected': int((df['condition'] == 0).sum()),
                        'coef_condition': beta,                 # Pref − Non-pref
                        'se_condition': se,
                        'z_condition': float(fit.tvalues.get('condition')),
                        'p_value': pval,                        # <-- key used by apply_fdr_correction
                        'effect_size_d_resid': effect_size,     # standardized by residual SD
                        'effect_size_interpretation': eff_interp,
                        'ci95_lo_condition': lo,
                        'ci95_hi_condition': hi,
                        'aic': float(fit.aic),
                        'bic': float(fit.bic),
                        'log_likelihood': float(fit.llf),
                        'random_effects_variance': re_var,
                        'model_converged': bool(getattr(fit, "converged", True)),
                        'analysis_type': 'linear_mixed_effects_words'
                    }
                    results[label] = out

                except Exception as e:
                    results[label] = None

        corrected = self.apply_fdr_correction(results, alpha=0.05)
        return corrected

    def _get_metric_unit(self, metric_name):
        """Get the unit for a given metric name"""
        units = {
            'FFD': 'ms',
            'TRT': 'ms', 
            'nFIX': 'count',
            'GPT': 'ms',
            'WORDS': 'count',
            'AWL': 'chars'
        }
        return units.get(metric_name, '')

    def create_boxplots_from_paired_data(self, output_dir=None):
        """Create boxplots for each metric using paired data CSV files with linked points between conditions"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        custom_palette = ['#E69F00', '#009E73']
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        chooser_cols = [
            ('user', 'chosen_user'),
            ('model', 'chosen_model')
        ]
        plt.style.use('default')
        sns.set_palette("husl")
        for metric_name, column_name in metrics.items():
            for chooser_key, chooser_col in chooser_cols:
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                paired_df = pd.read_csv(csv_filename, index_col=0)
                if paired_df.shape[0] == 0:
                    continue
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Participant': list(paired_df.index) + list(paired_df.index)
                })
                n_preferred = len(paired_df)
                n_rejected = len(paired_df)
                fig, ax = plt.subplots(1, 1, figsize=(8, 7))
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, order=['Rejected', 'Preferred'])
                for i, participant in enumerate(paired_df.index):
                    rejected_val = paired_df.loc[participant, 'Rejected']
                    preferred_val = paired_df.loc[participant, 'Preferred']
                    ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                    ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                    if preferred_val > rejected_val:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=1)
                    else:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=1)
                ax.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)")
                ax.set_xlabel("Condition")
                ax.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})")
                ax.set_xticklabels([f"Rejected\n(n={n_rejected})", f"Preferred\n(n={n_preferred})"])
                plt.tight_layout()
                plot_filename = os.path.join(output_dir, f'{metric_name}_{chooser_key}_boxplot_linked.png')
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                fig_no_links, ax_no_links = plt.subplots(1, 1, figsize=(8, 7))
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax_no_links, palette=custom_palette, order=['Rejected', 'Preferred'])
                ax_no_links.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)")
                ax_no_links.set_xlabel("Condition")
                ax_no_links.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})")
                ax_no_links.set_xticklabels([f"Rejected\n(n={n_rejected})", f"Preferred\n(n={n_preferred})"])
                plt.tight_layout()
                plot_filename_no_links = os.path.join(output_dir, f'{metric_name}_{chooser_key}_boxplot.png')
                fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
                plt.close(fig_no_links)
                plt.close(fig)

    def create_combined_boxplots_from_paired_data(self, output_dir=None):
        """Create combined boxplots for all metrics using paired data CSV files with linked points"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        custom_palette = ['#E69F00', '#009E73']
        metrics = {
            'FFD': 'first_fix_duration_mean',
            'TRT': 'fix_duration_mean', 
            'nFIX': 'fix_number_mean',
            'GPT': 'go_past_time_mean'
        }
        chooser_cols = [
            ('user', 'chosen_user'),
            ('model', 'chosen_model')
        ]
        plt.style.use('default')
        sns.set_palette("husl")
        for chooser_key, chooser_col in chooser_cols:
            fig, axes = plt.subplots(1, 4, figsize=(16, 6))
            fig.suptitle(f'{chooser_key.capitalize()}-Based Preferences', fontsize=18, fontweight='bold')
            for idx, (metric_name, column_name) in enumerate(metrics.items()):
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                paired_df = pd.read_csv(csv_filename, index_col=0)
                if paired_df.shape[0] == 0:
                    continue
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Participant': list(paired_df.index) + list(paired_df.index)
                })
                ax = axes[idx]
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                           order=['Rejected', 'Preferred'], width=0.5)
                for i, participant in enumerate(paired_df.index):
                    rejected_val = paired_df.loc[participant, 'Rejected']
                    preferred_val = paired_df.loc[participant, 'Preferred']
                    ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=3)
                    ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=3)
                    if preferred_val > rejected_val:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=0.8)
                    else:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=0.8)
                ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')
                ax.set_xlabel("", fontsize=12)
                if metric_name in ['FFD', 'TRT', 'GPT']:
                    ax.set_ylabel(f"{metric_name} (seconds)", fontsize=12)
                elif metric_name == 'nFIX':
                    ax.set_ylabel(f"{metric_name} (count)", fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
                ax.tick_params(axis='x', pad=15)
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'combined_reading_measures_{chooser_key}_boxplot_linked.png')
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            fig_no_links, axes_no_links = plt.subplots(1, 4, figsize=(16, 6))
            fig_no_links.suptitle(f'{chooser_key.capitalize()}-Based Preferences', fontsize=18, fontweight='bold')
            for idx, (metric_name, column_name) in enumerate(metrics.items()):
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                paired_df = pd.read_csv(csv_filename, index_col=0)
                if paired_df.shape[0] == 0:
                    continue
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Participant': list(paired_df.index) + list(paired_df.index)
                })
                ax = axes_no_links[idx]
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                           order=['Rejected', 'Preferred'], width=0.5)
                ax.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)", fontsize=14, fontweight='bold')
                ax.set_xlabel("Condition", fontsize=12)
                ax.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})", fontsize=12)
                ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
                ax.tick_params(axis='x', pad=15)
            plt.tight_layout()
            plot_filename_no_links = os.path.join(output_dir, f'combined_reading_measures_{chooser_key}_boxplot.png')
            fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
            plt.close(fig_no_links)
            plt.close(fig)

    def create_boxplots_words_from_paired_data(self, output_dir=None):
        """Create boxplots for WORDS/AWL using paired data CSV files with linked points"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        custom_palette = ['#E69F00', '#009E73']
        metrics = {
            'WORDS': 'number_words_mean',
            'AWL': 'avg_word_length_mean'
        }
        chooser_cols = [
            ('user', 'chosen_user')
        ]
        plt.style.use('default')
        sns.set_palette("husl")
        for metric_name, column_name in metrics.items():
            for chooser_key, chooser_col in chooser_cols:
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                paired_df = pd.read_csv(csv_filename, index_col=0)
                if paired_df.shape[0] == 0:
                    continue
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Participant': list(paired_df.index) + list(paired_df.index)
                })
                n_preferred = len(paired_df)
                n_rejected = len(paired_df)
                fig, ax = plt.subplots(1, 1, figsize=(8, 7))
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, order=['Rejected', 'Preferred'])
                for i, participant in enumerate(paired_df.index):
                    rejected_val = paired_df.loc[participant, 'Rejected']
                    preferred_val = paired_df.loc[participant, 'Preferred']
                    ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                    ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                    if preferred_val > rejected_val:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=1)
                    else:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=1)
                ax.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)", fontsize=16, fontweight='bold')
                ax.set_xlabel("Condition", fontsize=14)
                ax.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})", fontsize=14)
                ax.set_xticklabels([f"Rejected\n(n={n_rejected})", f"Preferred\n(n={n_preferred})"], fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=12)
                plt.tight_layout()
                plot_filename = os.path.join(output_dir, f'{metric_name}_{chooser_key}_boxplot_linked.png')
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                fig_no_links, ax_no_links = plt.subplots(1, 1, figsize=(8, 7))
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax_no_links, palette=custom_palette, order=['Rejected', 'Preferred'])
                ax_no_links.set_title(f"{metric_name} ({chooser_key.capitalize()}-Based Preferences)", fontsize=16, fontweight='bold')
                ax_no_links.set_xlabel("Condition", fontsize=14)
                ax_no_links.set_ylabel(f"{metric_name} ({self._get_metric_unit(metric_name)})", fontsize=14)
                ax_no_links.set_xticklabels([f"Rejected\n(n={n_rejected})", f"Preferred\n(n={n_preferred})"], fontsize=12)
                ax_no_links.tick_params(axis='both', which='major', labelsize=12)
                plt.tight_layout()
                plot_filename_no_links = os.path.join(output_dir, f'{metric_name}_{chooser_key}_boxplot.png')
                fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
                plt.close(fig_no_links)
                plt.close(fig)

    def create_combined_words_boxplots_from_paired_data(self, output_dir=None):
        """Create combined boxplots for WORDS and AWL using paired data CSV files with linked points"""
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        custom_palette = ['#E69F00', '#009E73']
        metrics = {
            'WORDS': 'number_words_mean',
            'AWL': 'avg_word_length_mean'
        }
        chooser_cols = [
            ('user', 'chosen_user')
        ]
        plt.style.use('default')
        sns.set_palette("husl")
        for chooser_key, chooser_col in chooser_cols:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            fig.suptitle(f'{chooser_key.capitalize()}-Based Preferences', fontsize=18, fontweight='bold')
            for idx, (metric_name, column_name) in enumerate(metrics.items()):
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                paired_df = pd.read_csv(csv_filename, index_col=0)
                if paired_df.shape[0] == 0:
                    continue
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Participant': list(paired_df.index) + list(paired_df.index)
                })
                ax = axes[idx]
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                           order=['Rejected', 'Preferred'], width=0.6)
                for i, participant in enumerate(paired_df.index):
                    rejected_val = paired_df.loc[participant, 'Rejected']
                    preferred_val = paired_df.loc[participant, 'Preferred']
                    ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=3)
                    ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=3)
                    if preferred_val > rejected_val:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=0.8)
                    else:
                        ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=0.8)
                ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')
                ax.set_xlabel("", fontsize=12)
                if metric_name == 'WORDS':
                    ax.set_ylabel(f"{metric_name} (count)", fontsize=12)
                elif metric_name == 'AWL':
                    ax.set_ylabel(f"{metric_name} (characters)", fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
                ax.tick_params(axis='x', pad=15)
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'combined_words_{chooser_key}_boxplot_linked.png')
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            fig_no_links, axes_no_links = plt.subplots(1, 2, figsize=(12, 6))
            fig_no_links.suptitle(f'{chooser_key.capitalize()}-Based Preferences', fontsize=18, fontweight='bold')
            for idx, (metric_name, column_name) in enumerate(metrics.items()):
                csv_filename = os.path.join(self.output_dir, "reading_metric_statistics", f"paired_data_{metric_name}_{chooser_col}.csv")
                if not os.path.exists(csv_filename):
                    continue
                paired_df = pd.read_csv(csv_filename, index_col=0)
                if paired_df.shape[0] == 0:
                    continue
                df_plot = pd.DataFrame({
                    'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                    'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                    'Participant': list(paired_df.index) + list(paired_df.index)
                })
                ax = axes_no_links[idx]
                sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                           order=['Rejected', 'Preferred'], width=0.6)
                ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')
                ax.set_xlabel("", fontsize=12)
                if metric_name == 'WORDS':
                    ax.set_ylabel(f"{metric_name} (count)", fontsize=12)
                elif metric_name == 'AWL':
                    ax.set_ylabel(f"{metric_name} (characters)", fontsize=12)
                ax.tick_params(axis='both', which='major', labelsize=11)
                ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
                ax.tick_params(axis='x', pad=15)
            plt.tight_layout()
            plot_filename_no_links = os.path.join(output_dir, f'combined_words_{chooser_key}_boxplot.png')
            fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
            plt.close(fig_no_links)
            plt.close(fig)

    def create_model_unified_analysis(self, output_dir=None):
        """Unified model analysis: two-sided test, save to summary.txt, save pivot tables, create plots"""
        if self.aggregate_data is None:
            return None
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, "analysis_plots")
        os.makedirs(output_dir, exist_ok=True)
        reading_metrics_dir = os.path.join(self.output_dir, "reading_metric_statistics")
        os.makedirs(reading_metrics_dir, exist_ok=True)
        antonella_data = self.aggregate_data[self.aggregate_data['participant_id'] == 'participant_7_7'].copy()
        if antonella_data.shape[0] == 0:
            return None
        metrics = {
            'WORDS': 'number_words_mean',
            'AWL': 'avg_word_length_mean'
        }
        results_summary = []
        model_results = {}
        for metric_name, column_name in metrics.items():
            if column_name not in antonella_data.columns:
                continue
            antonella_data["trial_id"] = antonella_data["trial"].astype(str).str.split(".").str[0]
            pivot_data = antonella_data.pivot_table(
                index='trial_id', 
                columns='chosen_model', 
                values=column_name,
                aggfunc='mean'
            ).reset_index()
            pivot_data.columns = ['trial', 'Rejected', 'Preferred']
            pivot_data = pivot_data.dropna(subset=['Preferred', 'Rejected'])
            if len(pivot_data) == 0:
                continue
            pivot_data['Difference'] = pivot_data['Preferred'] - pivot_data['Rejected']
            mean_diff = pivot_data['Difference'].mean()
            std_diff = pivot_data['Difference'].std()
            n_trials = len(pivot_data)
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(pivot_data['Difference'].dropna(), 0, alternative='two-sided')
            cohens_d = mean_diff / std_diff if std_diff != 0 else 0
            if not np.isnan(cohens_d) and n_trials > 1:
                df = n_trials - 1
                J = 1 - 3/(4*df - 1) if df > 0 else 1
                effect_size = cohens_d * J
                effect_size_name = "Hedges' g"
            else:
                effect_size = cohens_d
                effect_size_name = "Cohen's d"
            if abs(effect_size) < 0.2:
                effect_interpretation = "negligible"
            elif abs(effect_size) < 0.5:
                effect_interpretation = "small"
            elif abs(effect_size) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            pivot_filename = os.path.join(reading_metrics_dir, f'paired_data_{metric_name}_chosen_model.csv')
            pivot_data.to_csv(pivot_filename, index=False)
            results_summary.append({
                'Metric': metric_name,
                'N_Trials': n_trials,
                'Mean_Difference': mean_diff,
                'Std_Difference': std_diff,
                't_statistic': t_stat,
                'p_value': p_value,
                'Cohens_d': cohens_d,
                'Effect_Size': effect_size,
                'Effect_Size_Name': effect_size_name,
                'Effect_Interpretation': effect_interpretation
            })
            model_results[f'{metric_name}_model'] = {
                'metric': metric_name,
                'chooser': 'model',
                'column': column_name,
                'n_trials': n_trials,
                'preferred_mean': pivot_data['Preferred'].mean(),
                'preferred_std': pivot_data['Preferred'].std(),
                'rejected_mean': pivot_data['Rejected'].mean(),
                'rejected_std': pivot_data['Rejected'].std(),
                'difference_mean': mean_diff,
                'difference_std': std_diff,
                'test_type': "One-sample t-test (Antonella model)",
                'test_statistic': t_stat,
                'p_value': p_value,
                'effect_size': effect_size,
                'effect_size_name': effect_size_name,
                'cohens_d': cohens_d,
                'effect_interpretation': effect_interpretation,
                'is_significant': p_value < 0.05
            }
        custom_palette = ['#E69F00', '#009E73']
        plt.style.use('default')
        sns.set_palette("husl")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Model-Based Preferences', fontsize=18, fontweight='bold')
        for idx, (metric_name, column_name) in enumerate(metrics.items()):
            csv_filename = os.path.join(reading_metrics_dir, f'paired_data_{metric_name}_chosen_model.csv')
            if not os.path.exists(csv_filename):
                continue
            paired_df = pd.read_csv(csv_filename, index_col=0)
            if paired_df.shape[0] == 0:
                continue
            df_plot = pd.DataFrame({
                'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                'Trial': list(paired_df.index) + list(paired_df.index)
            })
            ax = axes[idx]
            sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                       order=['Rejected', 'Preferred'], width=0.6)
            for i, trial_id in enumerate(paired_df.index):
                rejected_val = paired_df.loc[trial_id, 'Rejected']
                preferred_val = paired_df.loc[trial_id, 'Preferred']
                ax.plot(0, rejected_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                ax.plot(1, preferred_val, 'o', color='darkblue', alpha=0.6, markersize=4)
                if preferred_val > rejected_val:
                    ax.plot([0, 1], [rejected_val, preferred_val], '-', color='green', alpha=0.4, linewidth=1)
                else:
                    ax.plot([0, 1], [rejected_val, preferred_val], '-', color='red', alpha=0.4, linewidth=1)
            ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("", fontsize=12)
            if metric_name == 'WORDS':
                ax.set_ylabel(f"{metric_name} (count)", fontsize=12)
            elif metric_name == 'AWL':
                ax.set_ylabel(f"{metric_name} (characters)", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
            ax.tick_params(axis='x', pad=15)
        plt.tight_layout()
        plot_filename = os.path.join(output_dir, 'chosen_model_words_boxplot_linked.png')
        fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
        fig_no_links, axes_no_links = plt.subplots(1, 2, figsize=(12, 6))
        fig_no_links.suptitle('Model-Based Preferences (Antonella)', fontsize=18, fontweight='bold')
        for idx, (metric_name, column_name) in enumerate(metrics.items()):
            csv_filename = os.path.join(reading_metrics_dir, f'paired_data_{metric_name}_chosen_model.csv')
            if not os.path.exists(csv_filename):
                continue
            paired_df = pd.read_csv(csv_filename, index_col=0)
            if paired_df.shape[0] == 0:
                continue
            df_plot = pd.DataFrame({
                'Value': pd.concat([paired_df['Rejected'], paired_df['Preferred']]),
                'Condition': ['Rejected'] * len(paired_df) + ['Preferred'] * len(paired_df),
                'Trial': list(paired_df.index) + list(paired_df.index)
            })
            ax = axes_no_links[idx]
            sns.boxplot(data=df_plot, x='Condition', y='Value', ax=ax, palette=custom_palette, 
                       order=['Rejected', 'Preferred'], width=0.6)
            ax.set_title(f"{metric_name}", fontsize=14, fontweight='bold')
            ax.set_xlabel("", fontsize=12)
            if metric_name == 'WORDS':
                ax.set_ylabel(f"{metric_name} (count)", fontsize=12)
            elif metric_name == 'AWL':
                ax.set_ylabel(f"{metric_name} (characters)", fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=11)
            ax.set_xticklabels(["Rejected", "Preferred"], fontsize=14)
            ax.tick_params(axis='x', pad=15)
        plt.tight_layout()
        plot_filename_no_links = os.path.join(output_dir, 'chosen_model_words_boxplot.png')
        fig_no_links.savefig(plot_filename_no_links, dpi=300, bbox_inches='tight')
        plt.close(fig_no_links)
        plt.close(fig)
        return model_results

    def create_summary_words_awl(self, trial_responses_dir=None, model_summary_dir=None, output_dir=None):
        """
        Create a new dataframe for Antonella with model data joined from trial_responses and model_summary files
        
        Args:
            trial_responses_dir (str): Directory containing trial_responses.csv (defaults to self.input_dir)
            model_summary_dir (str): Directory containing model_summary.csv (defaults to self.input_dir)
            output_dir (str): Output directory path (defaults to self.output_dir)
            
        Returns:
            pd.DataFrame: Merged dataframe with  data and model statistics
        """
        if self.aggregate_data is None:
            return None
            
        if trial_responses_dir is None:
            trial_responses_dir = self.input_dir
        if model_summary_dir is None:
            model_summary_dir = self.input_dir
        if output_dir is None:
            output_dir = self.output_dir
            
        # Get Antonella data
        antonella_data = self.aggregate_data[self.aggregate_data['participant_id'] == 'participant_7_7'].copy()
        if antonella_data.shape[0] == 0:
            return None
            
        # Select relevant columns
        columns_to_keep = ['trial', 'avg_word_length_mean', 'number_words_mean']
        antonella_data = antonella_data[columns_to_keep]
        
        # Load response model data (contains model info for each trial)
        response_model_path = os.path.join(trial_responses_dir, 'response_model.csv')
        if not os.path.exists(response_model_path):
            return None
            
        try:
            response_model_df = pd.read_csv(response_model_path)
        except Exception as e:
            return None
            
        # Load response summary data
        response_summary_path = os.path.join(model_summary_dir, 'response_summary.csv')
        if not os.path.exists(response_summary_path):
            return None
            
        try:
            response_summary_df = pd.read_csv(response_summary_path)
        except Exception as e:
            return None
            
        # Merge Antonella data with response model data
        try:
            antonella_data['trial'] = antonella_data['trial'].astype(str)
            response_model_df['n_resp'] = response_model_df['n_resp'].astype(str)
            
            # First merge: Antonella data with response model data
            merged_df = antonella_data.merge(
                response_model_df,
                left_on="trial",
                right_on="n_resp",
                how="inner"
            )
            
            if merged_df.empty:
                return None
                
            # Remove related_prompt column if it exists
            if 'related_prompt' in merged_df.columns:
                merged_df = merged_df.drop('related_prompt', axis=1)
                
            # Group by model and calculate averages
            if 'model' in merged_df.columns:
                model_averages = merged_df.groupby('model').agg({
                    'avg_word_length_mean': 'mean',
                    'number_words_mean': 'mean'
                }).round(3)
                
                # Merge with response summary
                final_merged_df = model_averages.merge(
                    response_summary_df,
                    left_index=True,
                    right_on='model',
                    how='inner'
                )
            
                # Organize columns in desired order
                desired_column_order = [
                    'model', 'appearances', 'avg_word_length_mean', 'number_words_mean',
                    'chosen_by_user', 'chosen_by_model', 'user_choice_rate', 'model_choice_rate'
                ]
                available_columns = [col for col in desired_column_order if col in final_merged_df.columns]
                if available_columns:
                    final_merged_df = final_merged_df[available_columns]
                    
                # Save the result
                final_output_filename = os.path.join(output_dir, 'model_response_awl_words_summary.csv')
                final_merged_df.to_csv(final_output_filename, index=False)
                
                return final_merged_df
            else:
                return merged_df
                
        except Exception as e:
            return None


def main():
    """Main function to run the analysis"""
    
    # Initialize analyzer
    analyzer = MetricsAnalyzer(input_dir="users", output_dir="users")
    
    # Aggregate all data
    aggregate_data = analyzer.aggregate_all_data()
    
    if aggregate_data is not None:
        # Save aggregate data
        analyzer.save_aggregate_data("users/aggregate_metrics.csv")
        
        # Perform statistical analysis (aggregated per user)
        results = analyzer.perform_paired_t_tests()
        
        # Apply FDR correction to reading metrics results (separate family of 8 tests: 4 metrics × 2 choosers)
        results_fdr = analyzer.apply_fdr_correction(results, alpha=0.05)
        
        # Additional paired t-tests for words and average word length (user only)
        words_results = analyzer.perform_paired_t_tests_words()
        
        # Apply FDR correction to words results (separate family of 2 tests)
        words_results_fdr = analyzer.apply_fdr_correction(words_results, alpha=0.05)
        
        # Create boxplots from paired data
        analyzer.create_boxplots_from_paired_data()
        analyzer.create_combined_boxplots_from_paired_data()
        analyzer.create_boxplots_words_from_paired_data()
        analyzer.create_combined_words_boxplots_from_paired_data()

        # Get model awl and wordsn analysis results
        model_results = analyzer.create_model_unified_analysis()
        
        # Apply FDR correction to model awl and words results
        model_results_fdr = analyzer.apply_fdr_correction(model_results, alpha=0.05)
        
        # Save summary statistics
        analyzer.save_summary_statistics(None, results_fdr, words_results_fdr, model_results=model_results_fdr)
        
        
        
        # Create model awl and words dataframe
        antonella_models_df = analyzer.create_summary_words_awl(
            trial_responses_dir="./users/response_statistics",
            model_summary_dir="./users/response_statistics",
            output_dir="./users/response_statistics"
        )
        
    else:
        pass

if __name__ == "__main__":
    main() 