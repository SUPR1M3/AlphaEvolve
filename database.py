import hashlib
import json
import random
from dataclasses import asdict, dataclass, field


@dataclass
class Program:
    id: int
    code: str
    score: float
    feedback: str
    valid: bool
    parent_id: int | None
    generation: int
    island: int
    iteration: int


@dataclass
class Island:
    members: list[int] = field(default_factory=list)


def _code_key(code: str) -> str:
    return hashlib.sha1("\n".join(l.rstrip() for l in code.strip().splitlines()).encode()).hexdigest()


class ProgramDatabase:
    def __init__(self, n_islands: int = 4, exploit_ratio: float = 0.7, island_capacity: int = 50,
                 migration_interval: int = 20, path: str | None = None, rng: random.Random | None = None):
        self.programs: dict[int, Program] = {}
        self.islands = [Island() for _ in range(n_islands)]
        self.exploit_ratio = exploit_ratio
        self.island_capacity = island_capacity
        self.migration_interval = migration_interval
        self.rng = rng or random.Random()
        self._by_code: dict[str, int] = {}
        self._file = open(path, "a") if path else None

    # ---- storage -------------------------------------------------------------------------

    def add(self, code: str, score: float, feedback: str, valid: bool, parent_id: int | None,
            island: int, iteration: int) -> tuple[Program, bool]:
        """Store a program. Returns (program, is_new); duplicates of existing code are not re-added,
        but a valid one still joins the island that rediscovered it."""
        key = _code_key(code)
        if key in self._by_code:
            prog = self.programs[self._by_code[key]]
            if prog.valid:
                self._join(island, prog.id)
            return prog, False
        parent = self.programs.get(parent_id) if parent_id is not None else None
        prog = Program(id=len(self.programs), code=code, score=score, feedback=feedback, valid=valid,
                       parent_id=parent_id, generation=parent.generation + 1 if parent else 0,
                       island=island, iteration=iteration)
        self.programs[prog.id] = prog
        self._by_code[key] = prog.id
        if valid:
            self._join(island, prog.id)
        if self._file:
            self._file.write(json.dumps(asdict(prog)) + "\n")
            self._file.flush()
        return prog, True

    def add_seed(self, code: str, score: float, feedback: str) -> Program:
        """The seed starts every island."""
        prog, _ = self.add(code, score, feedback, True, None, island=0, iteration=0)
        for i in range(1, len(self.islands)):
            self._join(i, prog.id)
        return prog

    def _join(self, island: int, pid: int):
        members = self.islands[island].members
        if pid in members:
            return
        members.append(pid)
        if len(members) > self.island_capacity:  #evict the weakest
            members.remove(min(members, key=lambda m: self.programs[m].score))

    # ---- sampling ------------------------------------------------------------------------

    def island_best(self, island: int) -> Program:
        return max((self.programs[m] for m in self.islands[island].members), key=lambda p: p.score)

    def best(self) -> Program:
        return max((p for p in self.programs.values() if p.valid), key=lambda p: p.score)

    def sample_parent(self, island: int) -> tuple[Program, str]:
        """Exploit (island best) with probability exploit_ratio, otherwise explore (random member)."""
        if self.rng.random() < self.exploit_ratio:
            return self.island_best(island), "exploit"
        return self.programs[self.rng.choice(self.islands[island].members)], "explore"

    def sample_inspirations(self, island: int, parent: Program, k: int = 3) -> list[Program]:
        """Up to k other programs from the island for the prompt: the best one, then diverse ones
        (random members whose scores differ from the parent's and from each other's). If the island
        is too small (early in a run), the other islands' best programs fill the remaining slots."""
        if k <= 0:
            return []
        others = [self.programs[m] for m in self.islands[island].members if m != parent.id]
        if len(others) < k:
            have = {p.id for p in others} | {parent.id}
            borrowed = sorted({self.island_best(i).id for i in range(len(self.islands)) if i != island} - have,
                              key=lambda pid: -self.programs[pid].score)
            others += [self.programs[pid] for pid in borrowed[:k - len(others)]]
        if not others:
            return []
        picks = [max(others, key=lambda p: p.score)]
        seen = {round(parent.score, 4), round(picks[0].score, 4)}
        pool = [p for p in others if p.id != picks[0].id]
        self.rng.shuffle(pool)
        for p in pool:
            if len(picks) == k:
                break
            if round(p.score, 4) not in seen:
                picks.append(p)
                seen.add(round(p.score, 4))
        for p in pool: #Fill remaining
            if len(picks) == k:
                break
            if p not in picks:
                picks.append(p)
        return picks

    # ---- migration -----------------------------------------------------------------------

    def maybe_migrate(self, iteration: int) -> bool:
        """Every migration_interval iterations, copy each island's best into the next island (ring)."""
        if self.migration_interval <= 0 or iteration % self.migration_interval:
            return False
        bests = [self.island_best(i).id for i in range(len(self.islands))]
        for i, pid in enumerate(bests):
            self._join((i + 1) % len(self.islands), pid)
        return True

    def summary(self) -> str:
        return "  ".join(f"I{i}:{self.island_best(i).score:.4f}/{len(isl.members)}"
                         for i, isl in enumerate(self.islands))
