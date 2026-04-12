from __future__ import annotations

from gc_memory.graph import RelevanceGraph


class TestReinforce:
    def test_creates_edges(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b", "c"])
        assert "b" in g.neighbors("a")
        assert "c" in g.neighbors("a")
        assert "a" in g.neighbors("b")

    def test_accumulates_weight(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b"], weight=1.0)
        g.reinforce(["a", "b"], weight=1.0)
        # After two reinforcements, edge weight should be 2.0
        assert g._edges["a"]["b"] == 2.0

    def test_caps_neighbors(self) -> None:
        g = RelevanceGraph(max_neighbors=3)
        # Add many neighbors to "a"
        for i in range(10):
            g.reinforce(["a", f"n{i}"], weight=float(i))
        assert len(g._edges["a"]) <= 3


class TestWeaken:
    def test_reduces_weight(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b"], weight=1.0)
        g.weaken("a", amount=0.5)
        assert g._edges["a"]["b"] == 0.5

    def test_removes_zero_edges(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b"], weight=0.5)
        g.weaken("a", amount=1.0)
        assert "b" not in g._edges.get("a", {})


class TestDecay:
    def test_reduces_weights(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b"], weight=1.0)
        g.decay(factor=0.5)
        assert g._edges["a"]["b"] == 0.5

    def test_frozen_exempt(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b"], weight=1.0)
        g.decay(factor=0.5, frozen_ids={"a", "b"})
        assert g._edges["a"]["b"] == 1.0  # unchanged

    def test_removes_tiny_edges(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b"], weight=0.005)
        g.decay(factor=0.5)  # 0.005 * 0.5 = 0.0025 < 0.01
        assert "b" not in g._edges.get("a", {})


class TestExpand:
    def test_returns_neighbors_not_in_seeds(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b", "c"])
        result = g.expand(["a"])
        assert "a" not in result
        assert "b" in result or "c" in result

    def test_empty_for_isolated_node(self) -> None:
        g = RelevanceGraph()
        assert g.expand(["x"]) == []

    def test_deduplicates(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "c"], weight=1.0)
        g.reinforce(["b", "c"], weight=1.0)
        result = g.expand(["a", "b"])
        assert result.count("c") == 1


class TestProperties:
    def test_num_nodes_edges(self) -> None:
        g = RelevanceGraph()
        g.reinforce(["a", "b", "c"])
        assert g.num_nodes == 3
        assert g.num_edges == 3  # a-b, a-c, b-c
