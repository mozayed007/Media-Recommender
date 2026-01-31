#!/usr/bin/env python3
"""
Dataset Validation Script

Validates the quality and completeness of the multi-media dataset.
"""

import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, Any
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def validate_dataset(parquet_path: str) -> Dict[str, Any]:
    """Validate dataset and return quality report.
    
    Args:
        parquet_path: Path to the parquet file
        
    Returns:
        Dictionary containing validation results
    """
    console.print(f"\n[bold cyan]Validating dataset: {parquet_path}[/bold cyan]\n")
    
    # Load dataset
    try:
        df = pd.read_parquet(parquet_path)
        console.print(f"✓ Loaded {len(df)} entries")
    except Exception as e:
        console.print(f"[bold red]✗ Failed to load dataset: {e}[/bold red]")
        return {"error": str(e)}
    
    report = {
        "total_entries": len(df),
        "columns": df.columns.tolist(),
        "media_types": {},
        "completeness": {},
        "quality_metrics": {},
        "issues": [],
    }
    
    # Check required columns
    required_cols = ["media_id", "title", "media_type"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        report["issues"].append(f"Missing required columns: {missing_cols}")
        console.print(f"[bold red]✗ Missing columns: {missing_cols}[/bold red]")
    else:
        console.print("✓ All required columns present")
    
    # Media type distribution
    if "media_type" in df.columns:
        media_type_counts = df["media_type"].value_counts().to_dict()
        report["media_types"] = media_type_counts
        
        table = Table(title="Media Type Distribution")
        table.add_column("Media Type", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        for media_type, count in media_type_counts.items():
            percentage = (count / len(df)) * 100
            table.add_row(str(media_type), str(count), f"{percentage:.1f}%")
        
        console.print(table)
    
    # Completeness checks
    completeness_cols = ["title", "synopsis", "genres", "score", "main_picture"]
    
    for col in completeness_cols:
        if col in df.columns:
            if col == "genres":
                # Check if genres list is non-empty
                non_empty = df[col].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
                completeness = (non_empty / len(df)) * 100
            else:
                completeness = (df[col].notna().sum() / len(df)) * 100
            
            report["completeness"][col] = completeness
    
    # Display completeness
    table = Table(title="Data Completeness")
    table.add_column("Field", style="cyan")
    table.add_column("Completeness", style="green")
    table.add_column("Status", style="yellow")
    
    for col, completeness in report["completeness"].items():
        status = "✓" if completeness >= 95 else "⚠" if completeness >= 80 else "✗"
        color = "green" if completeness >= 95 else "yellow" if completeness >= 80 else "red"
        table.add_row(
            col,
            f"{completeness:.1f}%",
            f"[{color}]{status}[/{color}]"
        )
    
    console.print(table)
    
    # Quality metrics
    if "score" in df.columns:
        scores = df["score"].dropna()
        if len(scores) > 0:
            report["quality_metrics"]["avg_score"] = float(scores.mean())
            report["quality_metrics"]["median_score"] = float(scores.median())
            report["quality_metrics"]["min_score"] = float(scores.min())
            report["quality_metrics"]["max_score"] = float(scores.max())
    
    if "synopsis" in df.columns:
        synopses = df["synopsis"].dropna()
        if len(synopses) > 0:
            avg_length = synopses.str.len().mean()
            report["quality_metrics"]["avg_synopsis_length"] = float(avg_length)
    
    # Check for duplicates
    if "media_id" in df.columns:
        duplicates = df["media_id"].duplicated().sum()
        report["quality_metrics"]["duplicates"] = int(duplicates)
        
        if duplicates > 0:
            report["issues"].append(f"Found {duplicates} duplicate media_ids")
            console.print(f"[bold red]✗ Found {duplicates} duplicates[/bold red]")
        else:
            console.print("✓ No duplicates found")
    
    # Check for invalid scores
    if "score" in df.columns:
        invalid_scores = df[(df["score"] < 0) | (df["score"] > 10)]["score"].count()
        if invalid_scores > 0:
            report["issues"].append(f"Found {invalid_scores} invalid scores (outside 0-10 range)")
            console.print(f"[bold yellow]⚠ Found {invalid_scores} invalid scores[/bold yellow]")
    
    # Display quality metrics
    if report["quality_metrics"]:
        table = Table(title="Quality Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in report["quality_metrics"].items():
            if isinstance(value, float):
                table.add_row(metric, f"{value:.2f}")
            else:
                table.add_row(metric, str(value))
        
        console.print(table)
    
    # Summary
    total_issues = len(report["issues"])
    
    if total_issues == 0:
        console.print(Panel(
            "[bold green]✓ Dataset validation passed![/bold green]\n"
            "No critical issues found.",
            title="Validation Summary",
            border_style="green"
        ))
    else:
        console.print(Panel(
            f"[bold yellow]⚠ Found {total_issues} issue(s)[/bold yellow]\n" +
            "\n".join(f"• {issue}" for issue in report["issues"]),
            title="Validation Summary",
            border_style="yellow"
        ))
    
    return report

def main():
    parser = argparse.ArgumentParser(description="Validate multi-media dataset")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the parquet file to validate"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Path to save validation report (JSON)"
    )
    
    args = parser.parse_args()
    
    # Validate dataset
    report = validate_dataset(args.dataset)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        console.print(f"\n[bold green]✓ Report saved to {output_path}[/bold green]")

if __name__ == "__main__":
    main()
