from app.models.research import ResearchPlan, ResearchState, SubQuestion


def test_planner_output_schema_accepts_three_subquestions():
    plan = ResearchPlan(objective="Compare systems", subquestions=[
        SubQuestion(id=str(i), question=f"Question {i}?", search_queries=[f"query {i}"])
        for i in range(3)
    ])
    assert len(plan.subquestions) == 3
    assert plan.subquestions[0].search_queries == ["query 0"]


def test_research_state_has_observable_defaults():
    state = ResearchState(original_question="A sufficiently detailed research question")
    assert state.task_id
    assert state.current_round == 0
    assert state.evidence == []
    assert state.final_report is None

