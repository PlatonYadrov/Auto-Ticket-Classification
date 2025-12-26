"""Comprehensive model evaluation script with metrics and visualizations."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from ticket_triage_ml.production.infer_onnx import ONNXInferenceEngine

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


def evaluate_model(test_size: int = 500):
    """Run comprehensive model evaluation.

    Args:
        test_size: Number of test samples to evaluate.
    """
    print("=" * 80)
    print("🎯 ОЦЕНКА МОДЕЛИ TICKET TRIAGE ML")
    print("=" * 80)
    print()

    # Load test data
    test_df = pd.read_parquet("data/processed/test_sample.parquet")
    test_df = test_df.head(test_size)
    print(f"📊 Тестовая выборка: {len(test_df)} записей")
    print()

    # Load model
    print("🔧 Загрузка модели...")

    model_path = Path("artifacts/model.onnx")
    tokenizer_path = Path("artifacts/tokenizer")
    label_maps_path = Path("artifacts/label_maps.json")

    engine = ONNXInferenceEngine(
        onnx_model_path=model_path,
        tokenizer_path=tokenizer_path,
        label_maps_path=label_maps_path,
        max_length=256,
    )
    print("✅ Модель загружена")
    print()

    # Run predictions
    print("🚀 Запуск предсказаний...")
    predictions = []
    for text in tqdm(test_df["text"].values, desc="Inference"):
        result = engine.predict_single(text)
        predictions.append(result)

    # Extract predictions
    pred_topics = [p.topic for p in predictions]
    pred_priorities = [p.priority for p in predictions]
    true_topics = test_df["topic"].values
    true_priorities = test_df["priority"].values

    # Calculate metrics
    print()
    print("=" * 80)
    print("📊 МЕТРИКИ МОДЕЛИ")
    print("=" * 80)
    print()

    # Topic metrics
    topic_acc = accuracy_score(true_topics, pred_topics)
    topic_f1_macro = f1_score(true_topics, pred_topics, average="macro", zero_division=0)
    topic_f1_weighted = f1_score(true_topics, pred_topics, average="weighted", zero_division=0)

    print("🎯 КЛАССИФИКАЦИЯ ТЕМЫ (TOPIC):")
    print(f"  Accuracy:        {topic_acc:.2%}")
    print(f"  F1 Macro:        {topic_f1_macro:.2%}")
    print(f"  F1 Weighted:     {topic_f1_weighted:.2%}")
    print()

    # Priority metrics
    priority_acc = accuracy_score(true_priorities, pred_priorities)
    priority_f1_macro = f1_score(true_priorities, pred_priorities, average="macro", zero_division=0)
    priority_f1_weighted = f1_score(
        true_priorities, pred_priorities, average="weighted", zero_division=0
    )

    print("⚡ КЛАССИФИКАЦИЯ ПРИОРИТЕТА (PRIORITY):")
    print(f"  Accuracy:        {priority_acc:.2%}")
    print(f"  F1 Macro:        {priority_f1_macro:.2%}")
    print(f"  F1 Weighted:     {priority_f1_weighted:.2%}")
    print()

    # Overall F1
    overall_f1 = (topic_f1_macro + priority_f1_macro) / 2
    print(f"🏆 ОБЩИЙ F1 MACRO: {overall_f1:.2%}")
    print()

    # Create visualizations
    print("=" * 80)
    print("📈 СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("=" * 80)
    print()

    plots_dir = Path("evaluation_results")
    plots_dir.mkdir(exist_ok=True)

    # 1. Confusion Matrix - Topic
    print("1️⃣ Confusion Matrix (Topic)...")
    cm_topic = confusion_matrix(true_topics, pred_topics)
    unique_topics = sorted(set(true_topics) | set(pred_topics))

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm_topic,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=unique_topics,
        yticklabels=unique_topics,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Topic Classification", fontsize=16, pad=20)
    plt.xlabel("Predicted Topic", fontsize=12)
    plt.ylabel("True Topic", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix_topic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Сохранено: evaluation_results/confusion_matrix_topic.png")

    # 2. Confusion Matrix - Priority
    print("2️⃣ Confusion Matrix (Priority)...")
    cm_priority = confusion_matrix(true_priorities, pred_priorities)
    unique_priorities = sorted(set(true_priorities) | set(pred_priorities))

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_priority,
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=unique_priorities,
        yticklabels=unique_priorities,
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Priority Classification", fontsize=16, pad=20)
    plt.xlabel("Predicted Priority", fontsize=12)
    plt.ylabel("True Priority", fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix_priority.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Сохранено: evaluation_results/confusion_matrix_priority.png")

    # 3. Per-class F1 scores - Topic
    print("3️⃣ Per-class F1 Scores (Topic)...")
    report_topic = classification_report(
        true_topics, pred_topics, output_dict=True, zero_division=0
    )
    topic_f1_scores = {
        topic: metrics["f1-score"]
        for topic, metrics in report_topic.items()
        if topic not in ["accuracy", "macro avg", "weighted avg"]
    }

    # Sort by F1 score
    sorted_topics = sorted(topic_f1_scores.items(), key=lambda x: x[1], reverse=True)
    topics_list = [t[0] for t in sorted_topics]
    f1_list = [t[1] for t in sorted_topics]

    plt.figure(figsize=(14, 10))
    colors = plt.cm.viridis(np.linspace(0, 1, len(topics_list)))
    bars = plt.barh(topics_list, f1_list, color=colors)
    plt.xlabel("F1 Score", fontsize=12)
    plt.title("F1 Score по категориям (Topic)", fontsize=16, pad=20)
    plt.xlim(0, 1)
    plt.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (bar, f1) in enumerate(zip(bars, f1_list)):
        plt.text(f1 + 0.01, i, f"{f1:.2%}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir / "f1_scores_topic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Сохранено: evaluation_results/f1_scores_topic.png")

    # 4. Metrics Summary Bar Chart
    print("4️⃣ Metrics Summary...")
    metrics_data = {
        "Topic\nAccuracy": topic_acc,
        "Topic\nF1 Macro": topic_f1_macro,
        "Priority\nAccuracy": priority_acc,
        "Priority\nF1 Macro": priority_f1_macro,
        "Overall\nF1 Macro": overall_f1,
    }

    plt.figure(figsize=(12, 6))
    colors_map = ["#3498db", "#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]
    bars = plt.bar(metrics_data.keys(), metrics_data.values(), color=colors_map, alpha=0.8)

    plt.ylabel("Score", fontsize=12)
    plt.title("Сводка метрик модели", fontsize=16, pad=20)
    plt.ylim(0, 1)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels
    for bar, value in zip(bars, metrics_data.values()):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{value:.2%}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(plots_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Сохранено: evaluation_results/metrics_summary.png")

    # 5. Class distribution comparison
    print("5️⃣ Class Distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Topic distribution
    topic_counts = pd.Series(true_topics).value_counts().head(10)
    axes[0].barh(range(len(topic_counts)), topic_counts.values, color="#3498db", alpha=0.7)
    axes[0].set_yticks(range(len(topic_counts)))
    axes[0].set_yticklabels([t[:40] for t in topic_counts.index])
    axes[0].set_xlabel("Count", fontsize=11)
    axes[0].set_title("Распределение тем (топ-10)", fontsize=13, pad=15)
    axes[0].grid(axis="x", alpha=0.3)

    # Priority distribution
    priority_counts = pd.Series(true_priorities).value_counts()
    colors_priority = ["#2ecc71", "#f39c12", "#e74c3c"]
    axes[1].bar(
        range(len(priority_counts)),
        priority_counts.values,
        color=colors_priority[: len(priority_counts)],
        alpha=0.7,
    )
    axes[1].set_xticks(range(len(priority_counts)))
    axes[1].set_xticklabels(priority_counts.index)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Распределение приоритетов", fontsize=13, pad=15)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("   ✅ Сохранено: evaluation_results/class_distribution.png")

    # Save detailed results
    print()
    print("=" * 80)
    print("💾 СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print()

    results = {
        "test_samples": len(test_df),
        "topic": {
            "accuracy": float(topic_acc),
            "f1_macro": float(topic_f1_macro),
            "f1_weighted": float(topic_f1_weighted),
            "per_class_f1": topic_f1_scores,
        },
        "priority": {
            "accuracy": float(priority_acc),
            "f1_macro": float(priority_f1_macro),
            "f1_weighted": float(priority_f1_weighted),
        },
        "overall_f1_macro": float(overall_f1),
    }

    with open(plots_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Результаты сохранены: evaluation_results/test_results.json")

    # Generate text report
    with open(plots_dir / "evaluation_report.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ОТЧЕТ ПО ОЦЕНКЕ МОДЕЛИ TICKET TRIAGE ML\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Тестовая выборка: {len(test_df)} записей\n")
        f.write("Датасет: Consumer Complaint Database (CFPB)\n\n")

        f.write("МЕТРИКИ:\n")
        f.write("-" * 80 + "\n\n")

        f.write("Классификация темы (Topic):\n")
        f.write(f"  • Accuracy:      {topic_acc:.2%}\n")
        f.write(f"  • F1 Macro:      {topic_f1_macro:.2%}\n")
        f.write(f"  • F1 Weighted:   {topic_f1_weighted:.2%}\n\n")

        f.write("Классификация приоритета (Priority):\n")
        f.write(f"  • Accuracy:      {priority_acc:.2%}\n")
        f.write(f"  • F1 Macro:      {priority_f1_macro:.2%}\n")
        f.write(f"  • F1 Weighted:   {priority_f1_weighted:.2%}\n\n")

        f.write(f"Общий F1 Macro:    {overall_f1:.2%}\n\n")

        f.write("=" * 80 + "\n")
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ ПО ТЕМАМ (топ-10):\n")
        f.write("=" * 80 + "\n\n")

        for topic, f1, support in sorted(
            [
                (t, m["f1-score"], m["support"])
                for t, m in report_topic.items()
                if t not in ["accuracy", "macro avg", "weighted avg"]
            ],
            key=lambda x: x[2],
            reverse=True,
        )[:10]:
            f.write(f"{topic[:50]:50s} F1={f1:.2%} (n={int(support)})\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("ДЕТАЛЬНЫЙ ОТЧЕТ ПО ПРИОРИТЕТАМ:\n")
        f.write("=" * 80 + "\n\n")
        f.write(classification_report(true_priorities, pred_priorities, zero_division=0))

    print("✅ Текстовый отчет: evaluation_results/evaluation_report.txt")

    print()
    print("=" * 80)
    print("✅ ОЦЕНКА ЗАВЕРШЕНА")
    print("=" * 80)
    print()
    print("📁 Результаты сохранены в: evaluation_results/")
    print("   • 5 графиков (PNG)")
    print("   • Метрики (JSON)")
    print("   • Текстовый отчет (TXT)")
    print()


if __name__ == "__main__":
    evaluate_model(test_size=500)
