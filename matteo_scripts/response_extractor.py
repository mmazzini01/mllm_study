import os
import pandas as pd
from typing import Set, List, Optional


class ResponseExtractor:
    """
    A class to extract and process response data from experiment files.
    Creates info_extended.csv files with user and model choices for each participant.
    """
    
    def __init__(self, response_folder: str, model_response_folder: str, users_base_path: str = None):
        """
        Initialize the ResponseExtractor.
        
        Args:
            response_folder: Path to the folder containing participant response data
            model_response_folder: Path to the folder containing model response files
            users_base_path: Base path for users folder (defaults to current working directory)
        """
        self.response_folder = response_folder
        self.model_response_folder = model_response_folder
        self.users_base_path = users_base_path or os.getcwd()
        self.data_folders = self._get_data_folders()
        self.model_chosen_trials = self._collect_model_chosen_trials()
    
    def _get_data_folders(self) -> List[str]:
        """Get all data folders from the response folder."""
        folders = os.listdir(self.response_folder)
        data_folders = []
        for folder in folders:
            folder_path = os.path.join(self.response_folder, folder)
            if os.path.isdir(folder_path) and '_data' in folder:
                data_folders.append(folder_path)
        return data_folders
    
    def _normalize_column_name(self, col_name: str) -> str:
        """Normalize column names by removing non-alphanumeric characters and converting to lowercase."""
        return "".join(ch for ch in str(col_name).lower() if ch.isalnum())
    
    def _process_ranking_responses(self, df: pd.DataFrame) -> List[Optional[str]]:
        """
        Process ranking responses from the dataframe.
        
        Args:
            df: DataFrame containing response data
            
        Returns:
            List of n_resp values based on ranking choices
        """
        results = []
        # Process every group of 3 rows (i.e., 0-2, 3-5, 6-8, ...)
        for i in range(4, len(df), 3):
            ranking_better_value = df.loc[i, 'ranking_better.response']
            if 'n_resp' not in df.columns:
                print("Colonna 'n_resp' non trovata in response_df.")
                results.append(None)
                continue
            if ranking_better_value == 1:
                n_resp_value = df.loc[i-2, 'n_resp']
            elif ranking_better_value == 2:
                n_resp_value = df.loc[i-1, 'n_resp']
            else:
                print(f"Valore di ranking_better.response non gestito: {ranking_better_value}")
                n_resp_value = None
            results.append(n_resp_value)
        
        if len(results) != 30:
            print(f"ERROR: {len(results)} TRIALS INSTEAD OF 30")
        return results
    
    def _collect_model_chosen_trials(self) -> Set[str]:
        """
        Collect trials chosen by models from Excel files.
        
        Returns:
            Set of trial IDs chosen by models
        """
        chosen_trials: Set[str] = set()
        if not os.path.isdir(self.model_response_folder):
            print(f"Model responses folder non trovato: {self.model_response_folder}")
            return chosen_trials
        
        for file_name in os.listdir(self.model_response_folder):
            name_l = file_name.lower()
            if not name_l.endswith('.xlsx') or not name_l.startswith('prompt'):
                continue
            
            file_path = os.path.join(self.model_response_folder, file_name)
            try:
                df = pd.read_excel(file_path)
            except Exception as exc:
                print(f"Impossibile leggere '{file_name}': {exc}")
                continue
            
            if df is None or df.empty:
                continue
            
            chosen_trial = self._process_model_file(df, file_name)
            if chosen_trial:
                chosen_trials.add(chosen_trial)
        
        if not chosen_trials:
            print("Nessun trial selezionato trovato nei file modello.")
        else:
            print(f"Trovati {len(chosen_trials)} trial scelti dal modello.")
        return chosen_trials
    
    def _process_model_file(self, df: pd.DataFrame, file_name: str) -> Optional[str]:
        """
        Process a single model file to extract the chosen trial.
        
        Args:
            df: DataFrame from the model file
            file_name: Name of the file being processed
            
        Returns:
            The chosen trial ID or None if processing failed
        """
        normalized_map = {self._normalize_column_name(c): c for c in df.columns}
        
        # Get column references
        col_related = normalized_map.get('relatedprompt')
        col_rank = normalized_map.get('rank')
        col_model = (
            normalized_map.get('model') or
            normalized_map.get('modelname') or
            normalized_map.get('model_id') or
            normalized_map.get('modelid')
        )
        col_nresp = normalized_map.get('nresp') or normalized_map.get('n_resp')
        
        if col_rank is None or col_nresp is None:
            print(f"Colonne mancanti in '{file_name}'. Richieste almeno 'rank' e 'n_resp'. Colonne presenti: {list(df.columns)}")
            return None
        
        trial_id_info = None
        if col_related is not None and not df[col_related].empty:
            trial_id_info = df[col_related].iloc[0]
        
        # Determine the row with maximum rank; if ties, prefer model 'gpt4-v'
        chosen_row = self._get_best_ranked_row(df, col_rank, col_model)
        if chosen_row is None:
            return None
        
        n_resp_value = chosen_row.get(col_nresp)
        if pd.isna(n_resp_value):
            print(f"n_resp nullo in '{file_name}' (trial {trial_id_info})")
            return None
        
        return str(n_resp_value).strip()
    
    def _get_best_ranked_row(self, df: pd.DataFrame, col_rank: str, col_model: Optional[str]) -> Optional[pd.Series]:
        """
        Get the best ranked row from the dataframe.
        
        Args:
            df: DataFrame to process
            col_rank: Column name for ranking
            col_model: Column name for model (optional)
            
        Returns:
            The best ranked row or None if processing failed
        """
        try:
            rank_series = pd.to_numeric(df[col_rank], errors='coerce')
            max_rank = rank_series.max()
            candidates = df[rank_series == max_rank]
            if candidates.empty:
                raise ValueError("Nessun candidato con rank valido")
            
            # Tie-breaker on model column if available
            if col_model is not None:
                try:
                    mask_gpt4v = candidates[col_model].astype(str).str.lower().str.contains('gpt-4v')
                    if mask_gpt4v.any():
                        return candidates[mask_gpt4v].iloc[0]
                except Exception:
                    pass
            
            return candidates.iloc[0]
            
        except Exception:
            try:
                sorted_df = df.sort_values(by=col_rank, ascending=False)
                if col_model is not None:
                    top_rank_value = sorted_df.iloc[0][col_rank]
                    top_candidates = sorted_df[sorted_df[col_rank] == top_rank_value]
                    mask_gpt4v = top_candidates[col_model].astype(str).str.lower().str.contains('gpt-4v') if not top_candidates.empty else []
                    if not top_candidates.empty and getattr(mask_gpt4v, 'any', lambda: False)():
                        return top_candidates[mask_gpt4v].iloc[0]
                    else:
                        return sorted_df.iloc[0]
                else:
                    return sorted_df.iloc[0]
            except Exception as exc:
                print(f"Impossibile determinare il max rank: {exc}")
                return None
    
    def _process_user_data(self, data_folder: str) -> None:
        """
        Process user data from a specific data folder.
        
        Args:
            data_folder: Path to the data folder containing user files
        """
        for file in os.listdir(data_folder):
            if not (file.endswith('.csv') and 'FINAL' in file):
                continue
            
            user = file.split('_TEST')[0]
            response_df = pd.read_csv(os.path.join(data_folder, file), sep=';')
            
            # Build user path
            user_folder = f"participant_{user.lower()}_{user.lower()}"
            user_path = os.path.join(self.users_base_path, user_folder, "session_1")
            
            # Read info.csv
            info_df = pd.read_csv(os.path.join(user_path, "info.csv"), sep=';')
            info_extended = info_df.copy()
            
            # Process ranking responses
            n_resp_values = self._process_ranking_responses(response_df)
            
            # Create extended info with user and model choices
            info_extended['trial'] = info_extended['trial'].astype(str).str.strip()
            n_resp_values_str = set(str(v).strip() for v in n_resp_values if v is not None)
            info_extended['chosen_user'] = info_extended['trial'].apply(lambda x: x in n_resp_values_str)
            info_extended['chosen_model'] = info_extended['trial'].apply(lambda x: x in self.model_chosen_trials)
            
            # Save info_extended.csv
            info_extended.to_csv(os.path.join(user_path, "info_extended.csv"), index=False)
            print(f"info_extended.csv saved for {user}")
    
    def process_all_users(self) -> None:
        """Process all user data folders and create info_extended.csv files."""
        print(f"Trial scelti finali dal modello: {sorted(self.model_chosen_trials)}")
        
        for data_folder in self.data_folders:
            self._process_user_data(data_folder)


# Example usage
if __name__ == "__main__":
    response_folder_users = r"C:\Users\matte\OneDrive - Università degli Studi di Padova\Desktop\msc_experiment"
    model_response_folder = r"C:\Users\matte\OneDrive - Università degli Studi di Padova\Desktop\fixation_exp\responses_files"
    users_base_path = r"C:\Users\matte\OneDrive - Università degli Studi di Padova\Desktop\fixation_exp\users"
    
    extractor = ResponseExtractor(response_folder_users, model_response_folder, users_base_path)
    extractor.process_all_users()



