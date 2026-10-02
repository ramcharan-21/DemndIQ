from rich.console import Console
from rich.table import Table

console = Console()

# ----------------- XGBoost Improvement Table -----------------
xgb_table = Table(title="🚀 XGBoost Demand Forecasting Improvement", show_header=True, header_style="bold cyan")

xgb_table.add_column("Metric", style="bold white", width=30)
xgb_table.add_column("Previous Model", justify="center", style="dim")
xgb_table.add_column("Optimized Ensemble", justify="center", style="bold green")

xgb_table.add_row("Forecast Accuracy", "96.80%", "98.45%")
xgb_table.add_row("Overall R² Score", "0.8945", "0.9706")
xgb_table.add_row("wMAPE", "3.20%", "1.55%")
xgb_table.add_row("MAE", "0.73", "0.36")
xgb_table.add_row("RMSE", "1.41", "0.74")

console.print("\n")
console.print(xgb_table)
console.print("\n")
