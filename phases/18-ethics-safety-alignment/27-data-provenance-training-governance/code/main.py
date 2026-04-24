"""California AB 2013 dataset-summary scaffold — stdlib Python.

Generates the 12-field summary required by California AB 2013 for a toy
dataset. Identifies follow-on obligations triggered by specific fields
(personal-information flag -> CPRA; copyright-protected flag -> EU TDM
opt-out respect).

Usage: python3 code/main.py
"""

from __future__ import annotations


AB_2013_FIELDS = [
    "dataset_source_name",
    "source_url_or_description",
    "acquisition_mode (purchased / licensed / other)",
    "amount_paid",
    "contains_personal_information (Y/N)",
    "is_synthetic_data (Y/N)",
    "collection_time_period",
    "modification_or_curation_description",
    "contains_copyright_protected_material (Y/N)",
    "aggregation_level",
    "intended_purpose",
    "publication_date",
]


TOY_EXAMPLE = {
    "dataset_source_name": "ToyBinaryClassification-1.0",
    "source_url_or_description": "generated in-repo via Python random.gauss",
    "acquisition_mode (purchased / licensed / other)": "other (synthetic)",
    "amount_paid": "$0.00",
    "contains_personal_information (Y/N)": "N",
    "is_synthetic_data (Y/N)": "Y",
    "collection_time_period": "2026-04 (single run, fixed seed)",
    "modification_or_curation_description": "none (generated deterministically)",
    "contains_copyright_protected_material (Y/N)": "N",
    "aggregation_level": "per-example",
    "intended_purpose": "pedagogical demonstration in Phase 18",
    "publication_date": "2026-04-22",
}


def flag_followups(summary: dict) -> list[str]:
    flags = []
    if summary["contains_personal_information (Y/N)"] == "Y":
        flags.append("triggers CPRA obligations (California Privacy Rights Act)")
    if summary["contains_copyright_protected_material (Y/N)"] == "Y":
        flags.append("must respect EU TDM opt-out signals (EU Copyright Directive)")
    if summary["is_synthetic_data (Y/N)"] == "Y":
        flags.append("may still trigger obligations on the base model used for generation")
    if "other" in summary["acquisition_mode (purchased / licensed / other)"]:
        flags.append("document the provenance of 'other' acquisition mode")
    return flags


def render_markdown(summary: dict) -> str:
    lines = ["# Dataset Summary (AB 2013 12-field)", ""]
    for field in AB_2013_FIELDS:
        lines.append(f"- **{field}**: {summary.get(field, '(missing)')}")
    followups = flag_followups(summary)
    if followups:
        lines.append("")
        lines.append("## Follow-up obligations triggered")
        for f in followups:
            lines.append(f"- {f}")
    return "\n".join(lines)


def main() -> None:
    print("=" * 74)
    print("CALIFORNIA AB 2013 12-FIELD GENERATOR (Phase 18, Lesson 27)")
    print("=" * 74)
    print()
    print(render_markdown(TOY_EXAMPLE))
    print()
    print("=" * 74)
    print("TAKEAWAY: the 12 fields are the California baseline. fields 5 and 9")
    print("trigger cascading obligations (CPRA + EU TDM). EU AI Act GPAI")
    print("Code of Practice Copyright chapter requires opt-out respect. 2025")
    print("DPA convergence: legitimate interest + opt-out = lawful. compliance")
    print("window is at collection time; irreversibility precludes downstream fix.")
    print("=" * 74)


if __name__ == "__main__":
    main()
