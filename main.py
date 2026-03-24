import logging
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from src.config import settings
from src.data_loader import DataLoader
from src.clustering import TemporalClustering
from src.evaluation import evaluate_preclustered

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initializing Data Loader pipeline...")
    loader = DataLoader(settings.paths.input_dir)
    try:
        df = loader.load_data()
    except ValueError as e:
        logger.error(f"Data loading failed: {e}")
        return
        
    df = loader.preprocess(df, settings)
    
    logger.info("Initializing Temporal Clustering engine...")
    engine = TemporalClustering(settings)
    df = engine.apply_clustering(df)
    df = engine.handle_isolates(df)
    
    logger.info("Exporting results...")
    settings.paths.output_dir.mkdir(parents=True, exist_ok=True)
    for current_filename, g in df.groupby(settings.columns.filename_col, sort=False, dropna=False):
        output_path = settings.paths.output_dir / (current_filename.replace(".xlsx", "").replace(".csv", "") + ".out.cluster_time.csv")
        g.to_csv(output_path, index=False, encoding="utf-8")
        
    logger.info("Proceeding to evaluation metrics calculation...")
    metrics_to_average = {
        "ARI": [], "NMI": [], "VI": [], "V-Measure": [], "Homogeneity": [],
        "Completeness": [], "1-1": [], "cluster_exact_F1": []
    }
    
    print("\n" + "="*50)
    print(" Metrics Breakdown per File")
    print("="*50)
    for current_filename, g in df.groupby(settings.columns.filename_col, sort=False):
        if g.empty:
            continue
        
        calculated_metrics = evaluate_preclustered(
            g, 
            true_cluster_col=settings.columns.true_cluster_col, 
            pred_cluster_col=settings.columns.pred_cluster_col,
            compute_link_metrics=False
        )
        
        print(f"\nFile: {current_filename}")
        print("-" * 30)
        for metric_name in metrics_to_average.keys():
            score = calculated_metrics.get(metric_name, 0.0)
            metrics_to_average[metric_name].append(score)
            print(f" {metric_name:<20}: {score:.4f}")
            
    print("\n" + "="*50)
    print(" FINAL SUMMARY (AVERAGE ACROSS FILES)")
    print("="*50)
    for metric_name, scores_list in metrics_to_average.items():
        if scores_list:
            mean_val = np.mean(scores_list)
            std_dev = np.std(scores_list, ddof=1)
            print(f"Metric: {metric_name:<18} | Mean: {mean_val:.4f} | Std. Dev: {std_dev:.4f}")
        else:
            print(f"No data for metric {metric_name}")

if __name__ == "__main__":
    main()
