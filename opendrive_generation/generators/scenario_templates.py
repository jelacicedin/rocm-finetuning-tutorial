import textwrap
import random


def generate_clothoid_scenario(
    start_x, start_y, end_x, end_y, lanes_left=2, lanes_right=2
):
    prompt_templates = [
        "Create a gentle road curve that smoothly connects two points.",
        "Design a road that eases into a curve, not sharp, just a natural bend.",
        "Build a smooth transition between two positions using a clothoid.",
        "Give me a curving road that doesn't feel sudden, something elegant.",
        "Simulate a road that connects two spots with a subtle, continuous curve.",
        "Imagine a road from point A to B that curves gradually, nothing jerky.",
        "Make a road that starts flat and blends into a nice, wide turn.",
        "Draw a road with a soft, sweeping curve between two coordinates.",
        "Use clothoid math to connect two locations with a smooth arc.",
        "Make a curvy road that feels like a proper on-ramp, gentle and flowing.",
    ]
    prompt = random.choice(prompt_templates)

    code = textwrap.dedent(
        f"""
        from scenariogeneration import xodr, prettyprint, ScenarioGenerator
        import pyclothoids as pcloth

        class Scenario(ScenarioGenerator):
            def __init__(self):
                super().__init__()

            def road(self):
                clothoids = pcloth.SolveG2(
                    {start_x}, {start_y}, 0,
                    xodr.STD_START_CLOTH,
                    {end_x}, {end_y}, 0,
                    xodr.STD_START_CLOTH,
                )
                roadgeoms = [xodr.Spiral(x.KappaStart, x.KappaEnd, length=x.length) for x in clothoids]
                road = xodr.create_road(roadgeoms, id=0, left_lanes={lanes_left}, right_lanes={lanes_right})
                odr = xodr.OpenDrive("myroad")
                odr.add_road(road)
                odr.adjust_roads_and_lanes()
                return odr

        if __name__ == "__main__":
            sce = Scenario()
            prettyprint(sce.road().get_element())
            sce.generate(".")
    """
    )

    return {"prompt": prompt.strip(), "response": code.strip()}


def generate_straight_road(length=100, road_id=1, lanes_left=1, lanes_right=1):
    prompt_templates = [
        f"Generate a straight road of {length} meters with {lanes_left} left lanes and {lanes_right} right lanes.",
        f"Create a {length}m long straight road. It should have {lanes_left} lanes on the left and {lanes_right} on the right.",
        f"A straight section of road, {length} meters, with symmetrical lane counts: {lanes_left} left and {lanes_right} right.",
        f"Make a basic straight OpenDRIVE road of {length} meters with {lanes_left} left and {lanes_right} right lanes.",
        f"Define a straight road geometry that is {length}m long and has {lanes_left}/{lanes_right} lanes.",
        f"Design a road segment that runs straight for {length}m with {lanes_left} lanes on one side, {lanes_right} on the other.",
        f"Add a simple straight road: {length} meters, {lanes_left} left lanes, {lanes_right} right lanes.",
        f"Construct a straight-line road for {length}m with {lanes_left}L/{lanes_right}R configuration.",
        f"Draw a flat road using OpenDRIVE that's {length} meters long and has {lanes_left} left and {lanes_right} right lanes.",
        f"Straight road, {length}m, {lanes_left}L and {lanes_right}R lanes — generate using OpenDRIVE syntax.",
    ]

    prompt = random.choice(prompt_templates)

    code = f"""
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        road = xodr.create_road(xodr.Line({length}), id={road_id}, left_lanes={lanes_left}, right_lanes={lanes_right})
        odr = xodr.OpenDrive("straight_road")
        odr.add_road(road)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
    """

    return {"prompt": prompt, "response": code.strip()}


def generate_line_spiral_combo(road_id=2):
    prompt_templates = [
        "Start with a straight road and gently curve it away using a spiral shape.",
        "Build a road that begins straight and then eases into a curve.",
        "Make a segment that transitions smoothly from a line into a spiral curve.",
        "I'd like a road that goes straight for a while, then bends naturally.",
        "Generate a road where drivers start on a straight stretch and then enter a spiral turn.",
        "Create a section that begins with a flat approach and curves outward using a spiral.",
        "Design a road with a calm straight lead-in followed by a gentle spiral arc.",
        "Simulate a highway piece that starts straight and gradually becomes a curve.",
        "A road that gives the feeling of a smooth transition from straight to curved.",
        "Start flat and end curving — like entering a long bend from a highway ramp.",
    ]
    prompt = random.choice(prompt_templates)

    code = f"""
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        road = xodr.create_road(
            [xodr.Line(30), xodr.Spiral(-0.00001, -0.035, 200)],
            id={road_id},
            left_lanes=2,
            right_lanes=2
        )
        odr = xodr.OpenDrive("line_spiral_combo")
        odr.add_road(road)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
    """

    return {"prompt": prompt, "response": code.strip()}


def generate_adjustable_planview_junction():
    prompt_templates = [
        "Build a loop that connects roads too complex to calculate by hand.",
        "Create a road network where the curves are tricky, and use something flexible to manage them.",
        "I need a junction that smoothly links multiple roads, even if they curve in weird ways.",
        "Design a layout where roads connect through a flexible, auto-adjusted segment.",
        "Give me a loop or connector between roads with hard-to-predict geometry.",
        "Use something like AdjustablePlanview to handle connections that don't fit neatly.",
        "Set up a connection where precision is tricky, and let the system figure it out.",
        "Make a multi-road junction with at least one difficult curve, solved programmatically.",
        "Handle road transitions that are too painful to do manually, let it auto-adjust.",
        "Let's connect complex segments with some smart interpolation, we don't need exact math here.",
    ]
    prompt = random.choice(prompt_templates)

    code = """\
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        road1 = xodr.create_road([xodr.Line(30), xodr.Spiral(-0.00001, -0.035, 200)], 1, 2, 2)
        road2 = xodr.create_road(xodr.Line(100), 2, 2, 2)
        road3 = xodr.create_road(xodr.Line(100), 3, 2, 2)

        jc = xodr.CommonJunctionCreator(100, "my junc")
        jc.add_incoming_road_cartesian_geometry(road1, 0, 0, 0, "successor")
        jc.add_incoming_road_cartesian_geometry(road2, 30, 0, -3.14, "predecessor")
        jc.add_incoming_road_cartesian_geometry(road3, 15, 15, -3.14 / 2, "successor")

        jc.add_connection(1, 2)
        jc.add_connection(3, 2)
        jc.add_connection(3, 1)

        road4 = xodr.create_road(xodr.AdjustablePlanview(100), 4, 2, 2)
        road4.add_predecessor(xodr.ElementType.road, 2, xodr.ContactPoint.end)
        road4.add_successor(xodr.ElementType.road, 1, xodr.ContactPoint.start)
        road2.add_successor(xodr.ElementType.road, 4, xodr.ContactPoint.start)
        road1.add_predecessor(xodr.ElementType.road, 4, xodr.ContactPoint.end)

        odr = xodr.OpenDrive("adjustable_planview_loop")
        odr.add_road(road1)
        odr.add_road(road2)
        odr.add_road(road3)
        odr.add_road(road4)
        odr.add_junction_creator(jc)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
    """
    return {"prompt": prompt, "response": code.strip()}


def generate_t_junction():
    prompt_templates = [
        "Make a basic T-junction where two roads meet a main road from either side.",
        "Create a T-shaped intersection with one road going straight, the others joining in from the sides.",
        "I'd like a simple three-way junction with a classic T-style layout.",
        "Generate a T-junction setup where traffic comes in from two directions into a central road.",
        "Simulate a crossroads with three arms, like a capital T.",
        "Design a road network where a side street joins the main road on both sides.",
        "Connect two side roads to a straight main road, forming a T.",
        "A junction shaped like the letter T. Simple and standard.",
        "Build a three-way road connection, one main route plus two merging from left and right.",
        "Think of a typical T-intersection you'd find in a suburb. That's what I need.",
    ]
    prompt = random.choice(prompt_templates)

    code = """
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        main_road = xodr.create_road(xodr.Line(100), 1, 2, 2)
        side_road_1 = xodr.create_road(xodr.Line(50), 2, 1, 1)
        side_road_2 = xodr.create_road(xodr.Line(50), 3, 1, 1)

        jc = xodr.CommonJunctionCreator(10, "t_junction")
        jc.add_incoming_road_cartesian_geometry(main_road, 0, 0, 0, "successor")
        jc.add_incoming_road_cartesian_geometry(side_road_1, 10, 0, 3.14 / 2, "predecessor")
        jc.add_incoming_road_cartesian_geometry(side_road_2, -10, 0, -3.14 / 2, "predecessor")

        jc.add_connection(2, 1)
        jc.add_connection(3, 1)

        odr = xodr.OpenDrive("t_junction")
        odr.add_road(main_road)
        odr.add_road(side_road_1)
        odr.add_road(side_road_2)
        odr.add_junction_creator(jc)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
    """
    return {"prompt": prompt, "response": code.strip()}


def generate_arc_road(length=80, curvature=0.01, road_id=5):
    prompt_templates = [
        "Create a simple curved road using a circular arc.",
        "Design a road that follows a consistent curve throughout.",
        "Make a road that bends evenly like a roundabout segment.",
        "A sweeping curve with constant radius all the way.",
        "Simulate a gently bending road using arc geometry.",
    ]
    prompt = random.choice(prompt_templates)

    code = f"""
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        road = xodr.create_road(
            xodr.Arc({curvature}, {length}),
            id={road_id},
            left_lanes=2,
            right_lanes=2
        )
        odr = xodr.OpenDrive("arc_road")
        odr.add_road(road)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
"""
    return {"prompt": prompt, "response": code.strip()}


def generate_composite_road(road_id=6):
    prompt_templates = [
        "Create a road that starts straight, curves in an arc, then spirals outward.",
        "Design a compound road segment with line, arc, and spiral sections.",
        "A mixed geometry road: line to arc to spiral.",
        "Build a stretch with varied shapes good for testing transitions.",
        "Generate a complex road path using multiple connected geometries.",
    ]
    prompt = random.choice(prompt_templates)

    code = f"""
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        road = xodr.create_road(
            [xodr.Line(50), xodr.Arc(0.01, 60), xodr.Spiral(0.01, 0.03, 80)],
            id={road_id},
            left_lanes=1,
            right_lanes=2
        )
        odr = xodr.OpenDrive("composite_road")
        odr.add_road(road)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
"""
    return {"prompt": prompt, "response": code.strip()}


def generate_cross_junction():
    prompt_templates = [
        "Design a four-way intersection with roads coming from all directions.",
        "Make a cross-junction that connects four roads at right angles.",
        "Simulate a traditional city intersection with four branches.",
        "Build a classic X-shaped road network using a junction.",
        "Create a busy crossroads using four roads meeting in the center.",
    ]
    prompt = random.choice(prompt_templates)

    code = """
from scenariogeneration import xodr, prettyprint, ScenarioGenerator

class Scenario(ScenarioGenerator):
    def __init__(self):
        super().__init__()

    def road(self):
        r1 = xodr.create_road(xodr.Line(100), 1, 2, 2)
        r2 = xodr.create_road(xodr.Line(100), 2, 2, 2)
        r3 = xodr.create_road(xodr.Line(100), 3, 2, 2)
        r4 = xodr.create_road(xodr.Line(100), 4, 2, 2)

        junc = xodr.CommonJunctionCreator(0, "crossroads")
        junc.add_incoming_road_cartesian_geometry(r1, 0, 0, 0, "successor")
        junc.add_incoming_road_cartesian_geometry(r2, 100, 0, 3.14, "predecessor")
        junc.add_incoming_road_cartesian_geometry(r3, 0, 100, -3.14/2, "successor")
        junc.add_incoming_road_cartesian_geometry(r4, 0, -100, 3.14/2, "predecessor")

        junc.add_connection(1, 2)
        junc.add_connection(3, 4)
        junc.add_connection(1, 3)
        junc.add_connection(2, 4)

        odr = xodr.OpenDrive("cross_junction")
        odr.add_road(r1)
        odr.add_road(r2)
        odr.add_road(r3)
        odr.add_road(r4)
        odr.add_junction_creator(junc)
        odr.adjust_roads_and_lanes()
        return odr

if __name__ == "__main__":
    sce = Scenario()
    prettyprint(sce.road().get_element())
    sce.generate(".")
"""
    return {"prompt": prompt, "response": code.strip()}
