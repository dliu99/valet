"""What to build: the sandwich as an ordered list of layers, and the questions about each.

A coding model edits this file (or writes another recipe like it); the runtime only
follows it. Each question is one narrow property, so its probability can be read on
its own and thresholds tuned per question from the records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .judge import Locate, OneOf, Question, YesNo


@dataclass(frozen=True)
class Item:
    name: str  # unique id, used in question ids and records
    desc: str  # noun phrase without an article: "slice of cheese"
    thickness_m: float = 0.005  # how much it adds to the stack
    pick: str = "pick_flat"  # skill names, bound in the config
    place: str = "place_on_stack"


@dataclass(frozen=True)
class Recipe:
    name: str
    board: Item
    layers: list[Item] = field(default_factory=list)


SANDWICH = Recipe(
    name="sandwich",
    board=Item("board", "cutting board", thickness_m=0.0),
    layers=[
        Item("bread", "slice of bread", 0.012),
        Item("cheese", "slice of cheese", 0.003),
        Item("ham", "slice of ham", 0.003),
        Item("lettuce", "lettuce leaf", 0.004),
        Item("bread_top", "slice of bread", 0.012),
    ],
)


# Question ids follow "<kind>:<item>" so records group by kind and tests can answer them.


def find(item: Item) -> dict[str, Question]:
    """A layer is looked for only where it is still loose, so a retry never picks one off the sandwich."""
    which = item.desc if item.thickness_m == 0.0 else f"{item.desc} that is not stacked on the sandwich"
    return {
        f"present:{item.name}": YesNo(f"Is there any {which} in `wrist`?"),
        f"where:{item.name}": Locate(f"the {which} in `wrist`"),
    }


def holding(item: Item) -> dict[str, Question]:
    return {
        f"holding:{item.name}": YesNo(
            f"Is the gripper in `wrist` holding a {item.desc}?",
            yes=f"A {item.desc} is pinched between the gripper's fingers.",
            no="The fingers are empty or hold something else.",
        ),
    }


def placed(item: Item) -> dict[str, Question]:
    return {
        f"on_top:{item.name}": YesNo(
            f"Is a {item.desc} lying flat as the top layer of the sandwich in `wrist`?",
            yes="It rests on the layers below and is mostly over them.",
            no="It is not on the sandwich, hangs mostly off it, or is folded.",
        ),
        "stack_ok": YesNo(
            "Are all layers of the sandwich in `wrist` stacked on the cutting board, with none slid off or folded over?"
        ),
        f"after_miss:{item.name}": OneOf(
            f"Where is the {item.desc} in `wrist`, if it is not on top of the sandwich?",
            {
                "on_top": "It is on top of the sandwich.",
                "still_held": "It is still between the gripper's fingers.",
                "beside": "It lies on the board or table, off the sandwich.",
                "not_visible": "It cannot be seen in the picture.",
            },
        ),
    }
