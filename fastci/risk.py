from __future__ import annotations

from dataclasses import dataclass, field

OBSERVED_OUTCOMES = ("pass", "fail")


@dataclass
class RiskHistory:
    # file -> test -> [runs, failures]; only observed completed pass/fail runs.
    associations: dict[str, dict[str, list[int]]] = field(default_factory=dict)
    recent: dict[str, list[int]] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)
    window: int = 5
    gamma: float = 0.8
    smoothing: float = 5

    def __post_init__(self) -> None:
        if self.window < 1 or not 0 < self.gamma <= 1 or self.smoothing <= 0:
            raise ValueError("Invalid risk parameters")

    def score(self, test: str, files: list[str]) -> dict[str, float]:
        code, count = 0.0, 0
        # Paper does not specify f* tie resolution. Lexical file order is stable.
        for file in sorted(set(files)):
            n, e = self.associations.get(file, {}).get(test, [0, 0])
            if n and e / n > code:
                code, count = e / n, n
        alpha = count / (count + self.smoothing) if code > 0 else 0.0
        outcomes = self.recent.get(test, [])[-self.window :]
        weights = [self.gamma ** (len(outcomes) - 1 - i) for i in range(len(outcomes))]
        weighted = (
            sum(y * w for y, w in zip(outcomes, weights)) / sum(weights)
            if weights
            else 0.0
        )
        passes = 0
        for y in reversed(outcomes):
            if y:
                break
            passes += 1
        attenuation = 0.6 if passes >= 4 else 0.85 if passes >= 2 else 1.0
        instability = weighted * attenuation
        return {
            "code": code,
            "alpha": alpha,
            "instability": instability,
            "total": alpha * code + (1 - alpha) * instability,
        }

    def order(self, tests: list[str], files: list[str]) -> tuple[list[str], dict]:
        scores = {test: self.score(test, files) for test in tests}
        return sorted(tests, key=lambda test: -scores[test]["total"]), scores

    def observe(self, run_id: str, files: list[str], results: dict[str, str]) -> None:
        if run_id in self.seen:
            raise ValueError(f"Duplicate history update: {run_id}")
        changed_files = sorted(set(files))
        for test, status in results.items():
            if status not in OBSERVED_OUTCOMES:
                continue
            outcome = int(status == "fail")
            for file in changed_files:
                pair = self.associations.setdefault(file, {}).setdefault(test, [0, 0])
                pair[0] += 1
                pair[1] += outcome
            self.recent[test] = (self.recent.get(test, []) + [outcome])[-self.window :]
        self.seen.append(run_id)
