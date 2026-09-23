from src.pipeline import ExperimentPipeline

p = ExperimentPipeline(use_mock_tts=True)


def make_frame(
    frame,
    red_x=240,
    yellow_x=340,
    plant_x=500,
    plant_y=650,
    spray_x=600,
    spray_y=650,
    white_hand=False,
    red_hand=False,
    yellow_hand=False,
    plant_hand=False,
    spray_hand=False,
):
    signals = {}

    if white_hand:
        signals["right_hand_near_white_container"] = True
    if red_hand:
        signals["right_hand_near_red_box"] = True
    if yellow_hand:
        signals["right_hand_near_yellow_box"] = True
    if plant_hand:
        signals["right_hand_near_plant"] = True
    if spray_hand:
        signals["right_hand_near_spray_bottle"] = True

    return {
        "frame": frame,
        "timestamp": float(frame),

        "objects": [
            {
                "class": "person",
                "track_id": 1,
                "confidence": 0.95,
                "bbox": [100, 100, 500, 900],
            },
            {
                "class": "white_container",
                "track_id": 2,
                "confidence": 0.90,
                "bbox": [200, 600, 600, 900],
            },
            {
                "class": "red_box",
                "track_id": 3,
                "confidence": 0.90,
                "bbox": [red_x, 650, red_x + 100, 750],
            },
            {
                "class": "yellow_box",
                "track_id": 4,
                "confidence": 0.90,
                "bbox": [yellow_x, 650, yellow_x + 100, 750],
            },
            {
                "class": "plant",
                "track_id": 5,
                "confidence": 0.90,
                "bbox": [
                    plant_x,
                    plant_y,
                    plant_x + 100,
                    plant_y + 150,
                ],
            },
            {
                "class": "spray_bottle",
                "track_id": 6,
                "confidence": 0.90,
                "bbox": [
                    spray_x,
                    spray_y,
                    spray_x + 80,
                    spray_y + 140,
                ],
            },
        ],

        "pose": {
            "person_track_id": 1,

            "left_wrist": {
                "x": spray_x if spray_hand else (
                    yellow_x if yellow_hand else plant_x
                ),
                "y": spray_y if spray_hand else (
                    700 if yellow_hand else plant_y + 50
                ),
                "confidence": 0.90,
            },

            "right_wrist": {
                "x": spray_x if spray_hand else (
                    yellow_x if yellow_hand else plant_x
                ),
                "y": spray_y if spray_hand else (
                    700 if yellow_hand else plant_y + 50
                ),
                "confidence": 0.90,
            },
        },

        "hand_object_interaction": {},
        "interaction_signals": signals,
    }


def show(frame, result):
    print(
        f"Frame {frame}: "
        f"{result[0].observed_action} | "
        f"state={result[0].current_state_id} | "
        f"status={result[0].status}"
    )


# ============================================================
# S01 - OPEN WHITE BOX
# ============================================================

print("\n=== TEST S01: OPEN_WHITE_BOX ===")

for i in range(1, 6):
    r = p.process_frame(
        make_frame(
            i,
            red_x=300,
            yellow_x=400,
            white_hand=True,
        )
    )
    show(i, r)

if p.state_machine.current_state.state_id != "S02":
    print("? S01 FAILED")
    p.close()
    raise SystemExit(1)

print("? S01 -> S02")


# ============================================================
# S02 - RETRIEVE RED BOX
# ============================================================

print("\n=== TEST S02: RETRIEVE_RED_BOX ===")

for i in range(6, 10):
    offset = (i - 5) * 15

    r = p.process_frame(
        make_frame(
            i,
            red_x=300 - offset,
            yellow_x=400,
            red_hand=True,
        )
    )
    show(i, r)

if p.state_machine.current_state.state_id != "S03":
    print("? S02 FAILED")
    p.close()
    raise SystemExit(1)

print("? S02 -> S03")


# ============================================================
# S03 - RETRIEVE YELLOW BOX
# ============================================================

print("\n=== TEST S03: RETRIEVE_YELLOW_BOX ===")

for i in range(10, 14):
    offset = (i - 9) * 15

    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=400 - offset,
            yellow_hand=True,
        )
    )
    show(i, r)

if p.state_machine.current_state.state_id != "S04":
    print("? S03 FAILED")
    p.close()
    raise SystemExit(1)

print("? S03 -> S04")


# ============================================================
# S04 FLUSH
# ============================================================

print("\n=== S04: FLUSH OLD MOVEMENT ===")

for i in range(14, 30):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
        )
    )
    show(i, r)


# ============================================================
# S04 - OPEN RED BOX
# ============================================================

print("\n=== TEST S04: OPEN_RED_BOX ===")

for i in range(30, 38):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            red_hand=True,
        )
    )
    show(i, r)

if p.state_machine.current_state.state_id != "S05":
    print("? S04 FAILED")
    p.close()
    raise SystemExit(1)

print("? S04 -> S05")


# ============================================================
# S05 - PLANT TO WORKPLACE
# ============================================================

print("\n=== TEST S05: PLANT_TO_WORKPLACE ===")

for i in range(38, 43):
    offset = (i - 37) * 15

    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=500 - offset,
            plant_y=650 + offset,
            plant_hand=True,
        )
    )
    show(i, r)

if p.state_machine.current_state.state_id != "S06":
    print("? S05 FAILED")
    p.close()
    raise SystemExit(1)

print("? S05 -> S06")


# ============================================================
# S06 PREPARATION
# Flush yellow-box movement from S03.
# ============================================================

print("\n=== S06: FLUSH OLD YELLOW-BOX MOVEMENT ===")

for i in range(43, 59):

    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=440,
            plant_y=710,
        )
    )
    show(i, r)


# ============================================================
# S06 - OPEN YELLOW BOX
# ============================================================

print("\n=== TEST S06: OPEN_YELLOW_BOX ===")

for i in range(59, 67):

    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=440,
            plant_y=710,
            yellow_hand=True,
        )
    )
    show(i, r)

print("\nSTATE AFTER S06:", p.state_machine.current_state.state_id)

if p.state_machine.current_state.state_id != "S07":
    print("? S06 FAILED")
    p.close()
    raise SystemExit(1)

print("? S06 -> S07")


# ============================================================
# S07 - SPRAY TO WORKPLACE
# ============================================================

print("\n=== TEST S07: SPRAY_TO_WORKPLACE ===")

for i in range(67, 75):

    offset = (i - 66) * 15

    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=440,
            plant_y=710,
            spray_x=650 - offset,
            spray_y=650,
        )
    )
    show(i, r)

print("\nSTATE AFTER S07:", p.state_machine.current_state.state_id)

if p.state_machine.current_state.state_id != "S08":
    print("? S07 FAILED")
    p.close()
    raise SystemExit(1)

print("? S07 -> S08")


# ============================================================
# S08 - PICK SPRAY
# ============================================================



print("\n=== S08: FLUSH OLD SPRAY MOVEMENT ===")
for i in range(75, 91):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=440,
            plant_y=710,
            spray_x=650,
            spray_y=650,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print("\n=== TEST S08: PICK_SPRAY ===")
for i in range(91, 99):
    spray_y = 650 - ((i - 90) * 15)
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=440,
            plant_y=710,
            spray_x=650,
            spray_y=spray_y,
            spray_hand=True,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print(f"\nFINAL STATE: {p.state_machine.current_state}")
if p.state_machine.current_state.state_id == "S09":
    print("? S08 PASSED")
else:
    print("? S08 FAILED")





print("\n=== S09: FLUSH PICK-SPRAY MOVEMENT ===")
for i in range(99, 115):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=500,
            plant_y=650,
            spray_x=600,
            spray_y=650,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print("\n=== TEST S09: SPRAY_PLANT ===")
for i in range(115, 123):
    spray_x = 500 + ((i - 115) * 8)
    spray_y = 650
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=500,
            plant_y=650,
            spray_x=spray_x,
            spray_y=spray_y,
            spray_hand=True,
            plant_hand=True,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print(f"\nFINAL STATE AFTER S09: {p.state_machine.current_state.state_id}")
if p.state_machine.current_state.state_id == "S10":
    print("? S09 PASSED")
else:
    print("? S09 FAILED")






# ============================================================
# TEST S10: SPRAY_TO_WORKPLACE_AGAIN
# ============================================================
print("\n=== TEST S10: SPRAY_TO_WORKPLACE_AGAIN ===")

# Flush previous S09 movement history
for i in range(123, 139):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=500,
            plant_y=650,
            spray_x=560,
            spray_y=650,
        )
    )

# Move spray bottle back toward workplace
for i in range(139, 147):
    spray_x = 560 - ((i - 139) * 8)
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=500,
            plant_y=650,
            spray_x=spray_x,
            spray_y=650,
            spray_hand=True,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print(f"FINAL STATE AFTER S10: {p.state_machine.current_state.state_id}")

if p.state_machine.current_state.state_id == "S11":
    print("? S10 PASSED")
else:
    print("? S10 FAILED")


# ============================================================
# TEST S11: PLANT_TO_RED_BOX
# ============================================================
print("\n=== TEST S11: PLANT_TO_RED_BOX ===")

# Flush previous spray movement
for i in range(147, 163):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=500,
            plant_y=650,
            spray_x=500,
            spray_y=650,
        )
    )

# Move plant toward red box
for i in range(163, 171):
    plant_x = 500 - ((i - 163) * 8)
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=plant_x,
            plant_y=650,
            spray_x=500,
            spray_y=650,
            plant_hand=True,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print(f"FINAL STATE AFTER S11: {p.state_machine.current_state.state_id}")

if p.state_machine.current_state.state_id == "S12":
    print("? S11 PASSED")
else:
    print("? S11 FAILED")


# ============================================================
# TEST S12: SPRAY_TO_YELLOW_BOX
# ============================================================
print("\n=== TEST S12: SPRAY_TO_YELLOW_BOX ===")

# Flush previous plant movement
for i in range(171, 187):
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=240,
            plant_y=650,
            spray_x=500,
            spray_y=650,
        )
    )

# Move spray bottle toward yellow box
for i in range(187, 195):
    spray_x = 500 - ((i - 187) * 8)
    r = p.process_frame(
        make_frame(
            i,
            red_x=240,
            yellow_x=340,
            plant_x=240,
            plant_y=650,
            spray_x=spray_x,
            spray_y=650,
            spray_hand=True,
            yellow_hand=True,
        )
    )
    print(f"Frame {i}: {r[2].action} | state={r[0].current_state_id} | status={r[0].status}")

print(f"FINAL STATE AFTER S12: {p.state_machine.current_state.state_id}")

if p.state_machine.current_state.state_id == "S13":
    print("? S12 PASSED")
else:
    print("? S12 FAILED")
