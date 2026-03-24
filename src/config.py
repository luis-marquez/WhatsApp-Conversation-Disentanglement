import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PathConfig:
    input_dir: Path = Path(os.getenv("INPUT_DIR", r"C:\Users\luism\Documents\GDrive Sync\conversaciones_whatsapp_privacidad\salida_muestreada_80-10-10\test"))
    output_dir: Path = Path(os.getenv("OUTPUT_DIR", r"C:\Users\luism\Documents\GDrive Sync\conversaciones_whatsapp_privacidad\separacion_cluster_tiempo"))

@dataclass
class ColumnConfig:
    filename_col: str = "filename"
    date_col: str = "Date"
    time_col: str = "Time"
    timestamp_col: str = "timestamp"
    true_cluster_col: str = "cluster_true"
    pred_cluster_col: str = "conversation"
    out_col: str = "cluster_true"
    sender_col: str = "UserPhone"
    sender_col_candidates: tuple = ("Author", "Sender", "From", "Nombre", "Name")
    message_col_candidates: tuple = ("Message", "Text", "Content", "Mensaje")

@dataclass
class ClusterConfig:
    time_delta_minutes: int = 10
    iqr_multiplier: int = 29

class Settings:
    def __init__(self):
        self.paths = PathConfig()
        self.columns = ColumnConfig()
        self.clustering = ClusterConfig()

settings = Settings()
