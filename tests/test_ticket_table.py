"""Risk boundaries, live state, and filtering for the ticket overview."""

import pandas as pd
from streamlit.testing.v1 import AppTest

from app.components.ticket_table import prepare_ticket_table


def sample_queue():
    return pd.DataFrame({
        "ticket_id": [101, 102, 103, 104, 105],
        "breach_risk": [0.70, 0.40, 0.399, 0.90, None],
        "priority": ["P3", "P1", "P4", "P2", "P1"],
        "is_overdue": [False, True, False, True, False],
        "is_due_soon": [True, True, False, False, False],
        "category": ["API"] * 5,
        "short_description": ["API ticket"] * 5,
        "remaining_display": ["1h", "Overdue", "8h", "Overdue", "10h"],
        "recommended_agent": ["Priya"] * 5,
        "recommended_action": ["Review ticket"] * 5,
    })


def test_classification_boundaries_and_live_state():
    queue = sample_queue()
    result = prepare_ticket_table(queue, {101: "Priya"}, {101: "Resolved"})
    assert result["Risk level"].tolist() == [
        "High risk", "Medium risk", "Low risk", "High risk", "Unknown",
    ]
    assert result["SLA status"].tolist() == [
        "Due soon", "Overdue", "Within SLA", "Overdue", "Within SLA",
    ]
    assert result.loc[0, "Priority"] == "P3"  # Risk doesn't change priority.
    assert result.loc[0, "Assigned agent"] == "Priya"
    assert result.loc[0, "Status"] == "Resolved"
    assert result.loc[1, "Assigned agent"] == "Unassigned"
    assert result.loc[0, "Breach risk (%)"] == 70.0
    assert "Risk level" not in queue.columns


def table_app():
    from tests.test_ticket_table import sample_queue
    from app.components.ticket_table import render_ticket_triage_table
    render_ticket_triage_table(sample_queue())


def test_table_filters_grouping_and_empty_results():
    at = AppTest.from_function(table_app).run()
    assert not at.exception
    assert at.dataframe[0].value["Ticket ID"].tolist() == ["104", "101", "102", "103", "105"]
    at.selectbox[0].select("Priority").run()
    assert not at.exception
    assert at.dataframe[0].value.columns[0] == "Priority"
    assert at.dataframe[0].value["Ticket ID"].tolist()[:2] == ["102", "105"]
    at.multiselect[0].select("High risk").run()
    assert len(at.dataframe[0].value) == 2
    at.multiselect[1].select("P1").run()
    assert not at.exception
    assert len(at.dataframe) == 0
    assert "No tickets match" in at.info[0].value
    at.multiselect[1].unselect("P1").run()
    at.multiselect[2].select("Overdue").run()
    assert at.dataframe[0].value["Ticket ID"].tolist() == ["104"]


def test_empty_queue():
    result = prepare_ticket_table(sample_queue().iloc[:0], {}, {})
    assert result.empty
    assert "Risk level" in result.columns


def test_supervisor_overview_shows_all_risk_groups():
    from pathlib import Path

    at = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app/main.py"), default_timeout=30,
    ).run()
    at.sidebar.radio[0].set_value("Supervisor View").run()
    assert not at.exception

    def overview_table():
        return next(df.value for df in at.dataframe if "Intake Time" in df.value.columns)

    all_tickets = overview_table()
    assert len(all_tickets) == 250
    assert all_tickets["Risk Tier"].nunique() == 3
    for label, lower, upper in [("High", 70, 100), ("Medium", 40, 70), ("Low", 0, 40)]:
        at.radio(key="overview_risk_group").set_value(label).run()
        assert not at.exception
        risks = overview_table()["Breach Risk (%)"]
        assert len(risks) > 0
        assert risks.ge(lower).all()
        assert risks.le(upper).all()
    at.radio(key="overview_risk_group").set_value("All").run()
    at.multiselect(key="overview_ticket_priorities").select("P1").run()
    assert not at.exception
    assert overview_table()["Priority"].eq("P1").all()
    at.multiselect(key="overview_ticket_priorities").unselect("P1").run()
    assert len(overview_table()) == 250
