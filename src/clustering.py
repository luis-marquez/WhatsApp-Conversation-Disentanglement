import pandas as pd
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class TemporalClustering:
    def __init__(self, settings):
        self.settings = settings
        self.ten_min = pd.Timedelta(minutes=self.settings.clustering.time_delta_minutes)
        
    def _process_group(self, current_filename: str, g: pd.DataFrame, df: pd.DataFrame):
        g_sorted = g.sort_values(self.settings.columns.timestamp_col)
        if g_sorted.empty:
            return
            
        timestamp_col = self.settings.columns.timestamp_col
        out_col = self.settings.columns.out_col
        sender_col = self.settings.columns.sender_col
        
        gaps = g_sorted[timestamp_col].diff().dt.total_seconds()
        df.loc[g_sorted.index, "gap_prev"] = gaps
        df.loc[g_sorted.index[0], "gap_prev"] = 0
        
        gaps_no_na = gaps.dropna()
        q1 = gaps_no_na.quantile(0.25) if not gaps_no_na.empty else 0
        q3 = gaps_no_na.quantile(0.75) if not gaps_no_na.empty else 0
        iqr = q3 - q1
        irc = iqr * self.settings.clustering.iqr_multiplier
        
        cluster_id = 1
        prev_idx = None
        
        for row in g_sorted.itertuples(index=True):
            idx = row.Index
            t = getattr(row, timestamp_col)
            
            if pd.isna(t):
                df.at[idx, out_col] = cluster_id
                prev_idx = idx
                continue
            
            if prev_idx is None:
                df.at[idx, out_col] = cluster_id
                prev_idx = idx
                continue

            gap = df.at[idx, "gap_prev"]
            same_sender = False
            if sender_col is not None and prev_idx in df.index:
                same_sender = (df.at[idx, sender_col] == df.at[prev_idx, sender_col])

            if same_sender:
                if gap > self.ten_min.total_seconds():
                    cluster_id += 1
                elif 'MediaLink' in df.columns and gap > irc/2 and getattr(row, "MediaType", "") == "image":
                    cluster_id += 1
            else:
                if gap > irc:
                    cluster_id += 1
                elif 'MediaLink' in df.columns and gap > irc/2 and getattr(row, "MediaType", "") == "image":
                    cluster_id += 1
            
            df.at[idx, out_col] = cluster_id
            df.at[idx, "irc"] = irc
            prev_idx = idx
            
        n_clusters = df.loc[g_sorted.index, out_col].nunique()
        logger.info(f" Clusters (before merging by citation) in '{current_filename}': {n_clusters}")

    def apply_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        filename_col = self.settings.columns.filename_col
        
        logger.info("Starting initial temporal clustering...")
        for current_filename, g in df.groupby(filename_col, sort=False, dropna=False):
            self._process_group(current_filename, g, df)
            
        return df

    def handle_isolates(self, df: pd.DataFrame) -> pd.DataFrame:
        filename_col = self.settings.columns.filename_col
        timestamp_col = self.settings.columns.timestamp_col
        out_col = self.settings.columns.out_col
        
        logger.info("Handling isolated elements...")
        for current_filename, g in df.groupby(filename_col, sort=False, dropna=False):
            g_sorted = g.sort_values(timestamp_col)
            if g_sorted.empty:
                continue
            
            sizes = df.loc[g_sorted.index].groupby(out_col).size()
            isolates_idx = sizes[sizes == 1].index
            isolates = g_sorted[g_sorted[out_col].isin(isolates_idx)].index
            
            if len(isolates) == 0:
                continue
            
            times = df.loc[g_sorted.index, timestamp_col]
            
            for idx in isolates:
                t = df.at[idx, timestamp_col]
                if pd.isna(t):
                    continue
                
                diffs = (times - t).abs()
                diffs.loc[idx] = pd.NaT
                
                if diffs.notna().any():
                    nearest_idx = diffs.idxmin()
                    df.at[idx, out_col] = df.at[nearest_idx, out_col]

            unique_ids = pd.Index(df.loc[g_sorted.index, out_col].astype(int).unique()).sort_values()
            remap = {old: i+1 for i, old in enumerate(unique_ids)}
            df.loc[g_sorted.index, out_col] = df.loc[g_sorted.index, out_col].astype(int).map(remap)
            
        return df
