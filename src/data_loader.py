import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataLoader:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        
    def _find_files(self):
        files = []
        for p in self.input_dir.glob("*.xlsx"):
            if not p.name.startswith("~$"):
                files.append(p)
        return sorted(files)
        
    def load_data(self) -> pd.DataFrame:
        files = self._find_files()
        dfs = []
        
        # Load Excel files
        for f in files:
            try:
                df_temp = pd.read_excel(f, sheet_name=0, engine="openpyxl")
                df_temp["filename"] = f.name
                if len(df_temp) > 1:
                    df_temp = df_temp[1:]
                dfs.append(df_temp)
            except Exception as e:
                logger.error(f"Could not read excel file {f.name}: {e}")
                
        # Load CSV files
        for f in self.input_dir.glob("*.csv"):
            try:
                df_temp = pd.read_csv(f, encoding="utf-8", sep=";", header=0)
                df_temp["filename"] = f.name
                dfs.append(df_temp)
            except Exception as e:
                logger.error(f"Could not read csv file {f.name}: {e}")
                
        if not dfs:
            raise ValueError("No valid files found to process.")
            
        df = pd.concat(dfs, ignore_index=True)
        return df

    def preprocess(self, df: pd.DataFrame, settings) -> pd.DataFrame:
        df[settings.columns.true_cluster_col] = -1
        
        # Create full timestamp
        df[settings.columns.timestamp_col] = pd.to_datetime(
            df[settings.columns.date_col].astype(str) + " " + df[settings.columns.time_col].astype(str),
            errors="coerce"
        )
        return df
