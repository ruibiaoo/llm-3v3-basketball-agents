import pytest
from pydantic import ValidationError

from bball3x3.schemas import (
    Abilities,
    AgentOutput,
    BallState,
    CaptainCall,
    ContactEvent,
    CoordinationConfig,
    CSEntry,
    CSProposal,
    DecisionRecord,
    Directive,
    GameEndRecord,
    GameStartRecord,
    GameState,
    InvariantViolationRecord,
    MemoryEntry,
    MemoryProposal,
    Message,
    PlayerState,
    PossessionEndRecord,
    RefereeDecision,
    RefereeState,
    StateRecord,
)


def test_abilities_valid_and_bounds():
    # valid ratings 1 to 10
    ab = Abilities(
        shooting=7,
        passing=8,
        defense=6,
        stamina=8,
        ball_handling=7,
        rebounding=6,
    )
    assert ab.shooting == 7

    # invalid rating over 10
    with pytest.raises(ValidationError):
        Abilities(
            shooting=11,
            passing=8,
            defense=6,
            stamina=8,
            ball_handling=7,
            rebounding=6,
        )

    # invalid rating under 1
    with pytest.raises(ValidationError):
        Abilities(
            shooting=0,
            passing=8,
            defense=6,
            stamina=8,
            ball_handling=7,
            rebounding=6,
        )

def test_player_and_ball_state():
    p = PlayerState(
        id="A1",
        team="A",
        pos=(4, 6),
        prev_pos=(4, 5),
        movement="DRIVE_RIGHT",
        speed="FAST",
        has_ball=True,
        fatigue=2.0,
    )
    assert p.id == "A1"
    assert p.has_ball is True

    ball = BallState(status="HELD", pos=(4, 6), holder="A1")
    assert ball.holder == "A1"

def test_game_state_defaults():
    ball = BallState(status="HELD", pos=(4, 6), holder="A1")
    state = GameState(
        event_idx=0,
        game_clock_s=600.0,
        shot_clock_s=12.0,
        score={"A": 0, "B": 0},
        team_fouls={"A": 0, "B": 0},
        offense="A",
        needs_clear=False,
        phase="LIVE",
        players={},
        ball=ball,
        matchups={},
    )
    assert state.event_idx == 0
    assert state.phase == "LIVE"

def test_messages_and_captain_call():
    directive = Directive(to="A3", act="SCREEN", target="B1")
    msg = Message(
        sender="A1",
        is_captain_channel=True,
        kind="action",
        text="screen for me",
        directive=directive,
    )
    call = CaptainCall(maintain=False, tactic=None, directives=[msg])
    assert len(call.directives) == 1
    assert call.directives[0].directive.act == "SCREEN"

def test_memory_and_cs_proposals():
    mem = MemoryProposal(
        opponent="B1",
        category="defensive_tendency",
        tendency="gambles on steals",
    )
    assert mem.opponent == "B1"

    cs = CSProposal(
        opponent_strategy="switches screens",
        counter_strategy="slip screen to rim",
        side="offense",
    )
    assert cs.side == "offense"

    entry = MemoryEntry(
        id="M01",
        opponent="B1",
        category="defensive_tendency",
        tendency="gambles",
    )
    assert entry.id == "M01"

    cs_entry = CSEntry(
        key="01",
        side="offense",
        opponent_strategy="switch",
        counter_strategy="slip",
        created_by="A1",
    )
    assert cs_entry.key == "01"

def test_referee_decision_and_state():
    decision = RefereeDecision(
        call="FOUL",
        offender="B1",
        fouled_player="A1",
        foul_type="DEFENSIVE_FOUL",
        rule_ids_cited=["R12.3"],
    )
    assert decision.call == "FOUL"

    contact = ContactEvent(
        contact_detected=True,
        players_involved=["A1", "B1"],
        contact_location=(5, 7),
        contact_type="BODY_CONTACT",
    )
    assert contact.contact_detected is True

    ball = BallState(status="LIVE", pos=(5, 7), holder="A1")
    ref_state = RefereeState(
        event_idx=1,
        game_clock_s=580.0,
        shot_clock_s=8.0,
        possession="A",
        ball_handler="A1",
        ball=ball,
        contact_event=contact,
    )
    assert ref_state.event_idx == 1

def test_agent_output_and_coordination():
    coord = CoordinationConfig(comms=True, captain=True, memory_write=True)
    assert coord.comms is True

    out = AgentOutput(
        action="PASS",
        target="A2",
        memory_ref=["M01"],
        messages=[],
        rationale="open teammate",
        raw_text="<RESPONSE>...</RESPONSE>",
        parse_ok=True,
    )
    assert out.action == "PASS"

def test_log_records():
    # test creating sample log records
    gs = GameStartRecord(
        schema_version=1,
        game_id="game_1",
        experiment_id="schema_tests",
        replicate_id="A_starts",
        seed=101,
        first_possession="A",
        team_conditions={"A": "A_full", "B": "B_no_comm"},
        matchup={"home": "A_full", "away": "B_no_comm"},
        coordination={"A": CoordinationConfig(), "B": CoordinationConfig(comms=False)},
        config_hash="abc123",
        config_snapshot={},
        prompt_hashes={},
        model_slug="openai/gpt-4o-mini",
        code_commit=None,
        roster={},
        initial_state=GameState(ball=BallState(pos=(7, 1), holder="A1")),
        initial_possession_id="p0001",
    )
    assert gs.record_type == "game_start"

    dec = DecisionRecord(
        event_idx=1,
        player="A1",
        team="A",
        raw_output='<A1 action="HOLD"/>',
        parsed_output=AgentOutput(action="HOLD"),
        parse_ok=True,
        invalid_reason=None,
        executed_action="HOLD",
        executed_target=None,
    )
    assert dec.record_type == "decision"

    ball = BallState(status="HELD", pos=(7, 1))
    st = GameState(event_idx=1, offense="A", ball=ball)
    sr = StateRecord(
        event_idx=1,
        state=st,
        invariant_checks_completed=True,
        constraints_checked=list(range(1, 21)),
    )
    assert sr.record_type == "state"

    inv = InvariantViolationRecord(
        event_idx=1,
        constraint_id=1,
        constraint_name="unique_ball_possession",
        details="two players hold ball",
    )
    assert inv.constraint_id == 1

    pe = PossessionEndRecord(
        possession_id="p0001",
        team="A",
        points=2,
        points_by_team={"A": 2, "B": 0},
        end_reason="MADE_2PT",
        start_event_idx=1,
        end_event_idx=3,
    )
    assert pe.points == 2

    ge = GameEndRecord(
        last_event_idx=3,
        end_reason="WIN_SCORE",
        final_score={"A": 21, "B": 15},
        ot_flag=False,
        totals={
            "decisions": 18,
            "decisions_by_team": {"A": 9, "B": 9},
            "transitions": 3,
            "possessions_by_team": {"A": 1, "B": 1},
        },
    )
    assert ge.final_score["A"] == 21
