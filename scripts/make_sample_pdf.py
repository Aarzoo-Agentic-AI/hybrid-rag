"""Generate data/samples/lumen_employee_handbook.pdf (fictional sample data).

Usage: uv run python scripts/make_sample_pdf.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.pdf_fixture import write_pdf  # noqa: E402

TITLE = "Lumen Robotics Employee Handbook"

PAGES = [
    [
        "Lumen Robotics Employee Handbook",
        "Version 3.1, effective 1 January 2026. Fictional sample document.",
        "",
        "1. Welcome",
        "This handbook explains the main policies for employees of Lumen Robotics.",
        "If a policy here differs from your employment contract, your contract applies.",
        "Questions about this handbook go to the People team at people@lumen.example.",
        "",
        "2. Working hours",
        "The standard working week is 38 hours, Monday to Friday. Core hours are",
        "10:00 to 15:00, when everyone should be available for meetings. Outside core",
        "hours you can choose when to work, as long as you agree it with your manager.",
        "",
        "3. Remote work",
        "Employees can work remotely for up to three days per week. Engineers who work",
        "with robot hardware must be on site at least two days per week, because the",
        "test lab cannot be used remotely. Fully remote arrangements need approval",
        "from both your manager and the People team.",
        "Lumen pays a home-office allowance of 600 USD once every two years, for a",
        "desk, chair or monitor. Keep the receipts and submit them as an expense.",
    ],
    [
        "4. Annual leave",
        "Full-time employees receive 25 days of paid annual leave per year, plus public",
        "holidays. Part-time employees receive leave in proportion to their hours.",
        "Up to 5 unused days can be carried over to the next year; any other unused",
        "days expire on 31 March of the following year.",
        "Request leave in the HR portal at least two weeks in advance for periods",
        "longer than three days.",
        "",
        "5. Sick leave",
        "Tell your manager before 10:00 on the first day you are sick. You need a",
        "doctor's note from the fourth consecutive sick day. Sick leave is paid in full",
        "for up to 30 days per year.",
        "",
        "6. Parental leave",
        "Primary caregivers receive 20 weeks of paid parental leave. Secondary",
        "caregivers receive 6 weeks. Parental leave can be taken in up to three blocks",
        "within 18 months of the birth or adoption.",
    ],
    [
        "7. Expenses and travel",
        "Expenses must be submitted within 60 days, with a receipt for every item over",
        "25 USD. Book travel through the company travel portal. Economy class is the",
        "standard for flights under 6 hours; premium economy is allowed for longer",
        "flights. The daily meal allowance during business travel is 75 USD.",
        "",
        "8. Safety in the test lab",
        "Everyone entering the robot test lab must complete safety training first and",
        "wear safety glasses. Never enter the marked safety zone of an LR-200 arm while",
        "it is powered. If a robot behaves unexpectedly, press the emergency stop and",
        "report the incident to the lab lead within 24 hours, even if nobody was hurt.",
        "",
        "9. Learning budget",
        "Each employee has a learning budget of 1,500 USD per year for courses,",
        "conferences and books. Unused budget does not carry over.",
    ],
]


def main() -> None:
    out = ROOT / "data" / "samples" / "lumen_employee_handbook.pdf"
    write_pdf(out, PAGES, title=TITLE)
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
