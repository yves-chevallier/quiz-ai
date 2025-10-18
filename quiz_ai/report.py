def write_summary_csv(
    grades_path: Path, solution: Dict[str, Any], out_csv: Path
) -> None:
    grades = load_json(grades_path)
    got, tot = compute_points_from_grades(grades, solution)
    note = round(((got / tot) * 5.0 + 1.0), 1) if tot else None

    # minimal CSV (une ligne pour ce PDF)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "points_obtenus", "points_total", "note"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "file": grades.get("_source_pdf", ""),
                "points_obtenus": got,
                "points_total": tot,
                "note": note,
            }
        )
