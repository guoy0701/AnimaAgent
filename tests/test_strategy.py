from anima.strategy import StrategyNetwork, TaskCategory, ActionType


class TestBoundedLearning:
    def test_preference_scores_stay_bounded(self):
        """After many positive feedbacks, scores should not grow past a reasonable bound."""
        net = StrategyNetwork()
        cat = TaskCategory.DATA_ANALYSIS

        for _ in range(200):
            net.learn_from_feedback(
                cat, ["decompose_first"], ["sql_query"], reward=1.0)

        profile = net.profiles[cat.value]
        score = profile.action_preferences.get("decompose_first", 0)
        assert score <= 1.5, f"Score grew unbounded to {score}"

    def test_negative_feedback_decreases_score(self):
        net = StrategyNetwork()
        cat = TaskCategory.CODE_WRITING

        # Positive first
        for _ in range(5):
            net.learn_from_feedback(cat, ["direct_execution"], [], reward=0.9)
        score_after_positive = net.profiles[cat.value].action_preferences.get("direct_execution", 0)

        # Then negative
        for _ in range(10):
            net.learn_from_feedback(cat, ["direct_execution"], [], reward=-0.8)
        score_after_negative = net.profiles[cat.value].action_preferences.get("direct_execution", 0)

        assert score_after_negative < score_after_positive


class TestExplorationLogic:
    def test_exploration_picks_least_tried_not_lowest_scored(self):
        net = StrategyNetwork(exploration_rate=1.0)  # always explore
        cat = TaskCategory.CODE_WRITING

        # Action A: tried 5 times with bad results
        for _ in range(5):
            net.learn_from_feedback(cat, ["direct_execution"], [], reward=-0.8)
        # Action B: tried 5 times with good results
        for _ in range(5):
            net.learn_from_feedback(cat, ["decompose_first"], [], reward=0.9)

        # Exploration should NOT repeatedly pick direct_execution (tried-and-failed)
        strategies = [net.decide_strategy(cat, {}, []) for _ in range(20)]
        action_lists = [s["actions"] for s in strategies]
        de_count = sum(1 for acts in action_lists if "direct_execution" in acts)
        assert de_count < 15, f"Exploration is re-trying failed action {de_count}/20 times"


class TestHistorySerialization:
    def test_history_survives_roundtrip(self):
        net = StrategyNetwork()
        net.learn_from_feedback(
            TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], reward=0.9)
        net.learn_from_feedback(
            TaskCategory.CODE_WRITING,
            ["direct_execution"], ["python_coding"], reward=0.7)

        data = net.to_dict()
        net2 = StrategyNetwork.from_dict(data)

        assert len(net2.history) == 2
        assert net2.history[0].reward == 0.9
        assert net2.history[1].task_category == TaskCategory.CODE_WRITING

    def test_attempt_counts_survive_roundtrip(self):
        net = StrategyNetwork()
        net.learn_from_feedback(
            TaskCategory.DATA_ANALYSIS,
            ["decompose_first"], ["sql_query"], reward=0.9)

        data = net.to_dict()
        net2 = StrategyNetwork.from_dict(data)

        profile = net2.profiles["data_analysis"]
        assert profile.action_attempt_counts.get("decompose_first", 0) == 1
