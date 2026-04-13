# backend/main.py
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import json
import os
from html import escape
import math
from sqlalchemy.orm import Session

# Import your ML pipeline modules
from backend.ml.matching import (
    find_optimal_roommates,
    persist_matching_results,

)
from backend.ml.model_loader import get_model
from backend.database import get_db, init_db, SessionLocal
from backend.models import FeedbackStaging, User
from backend.schemas import BatchJobResult, FeedbackCreate, FeedbackResponse
from backend.scheduler import start_scheduler, stop_scheduler
from backend.services.retraining import (
    get_current_assignment_record,
    get_feedback_for_cycle,
    get_active_students_csv,
    run_feedback_batch_job,
    sync_users_from_dataframe,
)

app = FastAPI(title="Hostel Roommate Matching System")

# 1. Setup CORS so React can communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with your React app's URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths for storing data locally
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")

# The static paths for your input and output files
STUDENTS_CSV_FILE = os.path.join(PROJECT_ROOT, "backend/training/data/students.csv")
PREDICT_CSV_FILE = os.path.join(DATA_DIR, "predict_data.csv")
ASSIGNMENTS_FILE = os.path.join(DATA_DIR, "assignments.json")


if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend-static")


@app.on_event("startup")
async def ensure_model_ready_on_startup():
    """Initialize persistence, users, model, and scheduler when the API starts."""
    init_db()

    try:
        df_students = pd.read_csv(get_active_students_csv())
        with SessionLocal() as db:
            sync_users_from_dataframe(db, df_students)
            db.commit()
    except Exception as exc:
        print(f"User bootstrap skipped: {exc}")

    print("Initializing ML model on startup...")
    get_model()
    print("ML model is ready.")
    start_scheduler()


@app.on_event("shutdown")
async def shutdown_event():
    stop_scheduler()


NUMERIC_GRAPH_FIELDS = [
    "sleep_time",
    "wake_up_time",
    "morning_productivity",
    "night_productivity",
    "cleanliness_score",
    "room_organization_level",
    "noise_tolerance",
    "daily_study_hours",
    "introvert_extrovert_score",
    "room_stay_duration",
]

NUMERIC_FIELDS = set(NUMERIC_GRAPH_FIELDS)

BINARY_GRAPH_FIELDS = [
    "alarm_usage",
    "smoking_drinking",
    "workout",
    "gaming",
    "anime",
]

BINARY_FIELDS = set(BINARY_GRAPH_FIELDS)

TIME_CYCLIC_FIELDS = {
    "sleep_time",
    "wake_up_time",
}

DISPLAY_NAMES = {
    "gender": "Gender",
    "year_of_study": "Year of Study",
    "department": "Department",
    "sleep_time": "Sleep Time",
    "wake_up_time": "Wake Up Time",
    "alarm_usage": "Alarm Usage (Y/N)",
    "morning_productivity": "Morning Productivity",
    "night_productivity": "Night Productivity",
    "cleanliness_score": "Cleanliness Score",
    "room_organization_level": "Room Organization Level",
    "noise_tolerance": "Noise Tolerance",
    "study_noise_preference": "Study Noise Preference",
    "fan_or_cooler_preference": "Fan/Cooler Preference",
    "study_habit": "Study Habit",
    "daily_study_hours": "Study Duration (Hours)",
    "food_preference": "Food Preference",
    "exam_preparation_style": "Exam Preparation Style",
    "introvert_extrovert_score": "Introvert Extrovert Score",
    "social_frequency": "Social Frequency",
    "relationship_status": "Relationship Status",
    "smoking_drinking": "Smoking Drinking (Y/N)",
    "workout": "Workout (Y/N)",
    "gaming": "Gaming (Y/N)",
    "anime": "Anime (Y/N)",
    "room_stay_duration": "Room Stay Duration",
    "career_interest": "Career Interest",
    "cult_sports": "Cult Sports",
    "language": "Language",
}

COMPARISON_FIELDS = list(DISPLAY_NAMES.keys())

OTHER_CATEGORICAL_FIELDS = [
    field for field in COMPARISON_FIELDS if field not in NUMERIC_FIELDS and field not in BINARY_FIELDS
]

RADAR_FIELDS = [
    "wake_up_time",
    "cleanliness_score",
    "room_organization_level",
    "daily_study_hours",
    "introvert_extrovert_score",
    "sleep_time",
    "noise_tolerance",
    "room_stay_duration",
]


def _load_frontend_template() -> str:
    template_path = os.path.join(FRONTEND_DIR, "index.html")
    if not os.path.exists(template_path):
        raise HTTPException(status_code=404, detail="Frontend is not available yet.")
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def _assignment_for_student(student_id: int):
    with SessionLocal() as db:
        record = get_current_assignment_record(db, student_id)
        if record:
            return {
                "your_id": student_id,
                "roommate_id": record.roommate_id,
                "compatibility_score": record.compatibility_score,
                "source": record.source,
                "matching_cycle": record.matching_cycle,
                "message": "No roommate assigned" if record.roommate_id is None else None,
            }

    if not os.path.exists(ASSIGNMENTS_FILE):
        return None

    with open(ASSIGNMENTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    for pair in data.get("assignments", []):
        if pair.get("student_1") == student_id:
            return {
                "your_id": student_id,
                "roommate_id": pair.get("student_2"),
                "compatibility_score": pair.get("compatibility_score"),
                "source": "json_fallback",
                "matching_cycle": data.get("matching_cycle"),
                "message": pair.get("message"),
            }
        if pair.get("student_2") == student_id:
            return {
                "your_id": student_id,
                "roommate_id": pair.get("student_1"),
                "compatibility_score": pair.get("compatibility_score"),
                "source": "json_fallback",
                "matching_cycle": data.get("matching_cycle"),
                "message": pair.get("message"),
            }

    return None


def _review_panel_html(student_id: int, assignment: dict) -> str:
    roommate_id = assignment.get("roommate_id")
    matching_cycle = assignment.get("matching_cycle")

    if assignment.get("source") == "json_fallback" or matching_cycle is None:
        return (
            '<article class="panel review-panel">'
            '<h3>Roommate Review</h3>'
            '<p class="panel-note">Feedback is enabled only for DB-saved assignments.</p>'
            '<div class="status error">'
            'Assignment exists in JSON only. Run matching once from backend/API so feedback can be validated and stored by cycle.'
            '</div>'
            '</article>'
        )

    if roommate_id is None:
        return (
            '<article class="panel review-panel">'
            '<h3>Roommate Review</h3>'
            '<p class="panel-note">Feedback is available only when a roommate is assigned.</p>'
            '</article>'
        )

    existing_feedback = None
    with SessionLocal() as db:
        existing_feedback = get_feedback_for_cycle(
            db,
            user_id=student_id,
            roommate_id=int(roommate_id),
            matching_cycle=int(matching_cycle),
        )

    if existing_feedback:
        return (
            '<article class="panel review-panel">'
            '<h3>Roommate Review</h3>'
            '<div class="status ok">'
            'Review already submitted for this roommate match.<br/>'
            f'Your score: {existing_feedback.feedback_score}<br/>'
            'This review is already stored for batch retraining.'
            '</div>'
            '</article>'
        )

    return (
        '<article class="panel review-panel">'
        '<h3>Roommate Review</h3>'
        '<p class="panel-note">You can review this roommate once. The feedback will be used later in batch retraining.</p>'
        '<div id="feedback-status" class="status placeholder">No review submitted yet for this roommate match.</div>'
        '<form id="feedback-form" class="review-form" '
        f'data-user-id="{student_id}" data-roommate-id="{roommate_id}" data-matching-cycle="{matching_cycle}">'
        '<label for="feedback_score">Compatibility review score</label>'
        '<div class="review-controls">'
        '<input id="feedback_score" name="feedback_score" type="number" min="0" max="100" step="1" placeholder="0 to 100" required />'
        '<button type="submit">Submit Review</button>'
        '</div>'
        '<p class="panel-note compact">Once submitted, this review stays locked and will wait in the staging queue until you run batch retraining.</p>'
        '</form>'
        '</article>'
    )


def _normalize_for_radar(df: pd.DataFrame, field: str, value):
    minimum = float(df[field].min())
    maximum = float(df[field].max())
    if maximum == minimum:
        return 0.5
    return (float(value) - minimum) / (maximum - minimum)


def _polygon_points(normalized_values, center=170.0, radius=128.0):
    total = len(normalized_values)
    points = []
    for index, value in enumerate(normalized_values):
        angle = -math.pi / 2 + (2 * math.pi * index / total)
        x = center + radius * value * math.cos(angle)
        y = center + radius * value * math.sin(angle)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _radar_label_positions(total=8, center=170.0, radius=154.0):
    positions = []
    for index in range(total):
        angle = -math.pi / 2 + (2 * math.pi * index / total)
        x = center + radius * math.cos(angle)
        y = center + radius * math.sin(angle)
        positions.append((x, y))
    return positions


def _radar_svg(student_row: pd.Series, roommate_row: pd.Series, df_students: pd.DataFrame):
    student_norm = [_normalize_for_radar(df_students, field, student_row[field]) for field in RADAR_FIELDS]
    mate_norm = [_normalize_for_radar(df_students, field, roommate_row[field]) for field in RADAR_FIELDS]

    rings = []
    for factor in [1.0, 0.8, 0.6, 0.4, 0.2]:
        ring_points = _polygon_points([factor] * len(RADAR_FIELDS))
        rings.append(f'<polygon points="{ring_points}" class="grid-ring" />')

    axes = []
    for x, y in _radar_label_positions(total=len(RADAR_FIELDS), center=170.0, radius=128.0):
        axes.append(f'<line x1="170" y1="170" x2="{x:.2f}" y2="{y:.2f}" class="grid-axis" />')

    labels = []
    for field, (x, y) in zip(RADAR_FIELDS, _radar_label_positions(total=len(RADAR_FIELDS), center=170.0, radius=158.0)):
        labels.append(
            f'<text x="{x:.2f}" y="{y:.2f}" class="axis-label">{escape(DISPLAY_NAMES[field])}</text>'
        )

    student_points = _polygon_points(student_norm)
    mate_points = _polygon_points(mate_norm)

    return (
        '<svg class="radar-svg" viewBox="0 0 340 340" role="img" aria-label="Student and roommate radar comparison">'
        + "".join(rings)
        + "".join(axes)
        + f'<polygon points="{student_points}" class="student-shape" />'
        + f'<polygon points="{mate_points}" class="mate-shape" />'
        + "".join(labels)
        + '</svg>'
    )


def _build_comparison(student_row: pd.Series, roommate_row: pd.Series):
    matches = []
    mismatches = []
    impacts = []

    for field in COMPARISON_FIELDS:
        label = DISPLAY_NAMES[field]
        student_value = student_row[field]
        mate_value = roommate_row[field]

        if field in BINARY_FIELDS:
            if int(student_value) == int(mate_value):
                matches.append(f"{label}")
            else:
                mismatches.append(f"{label}")
                impacts.append((label, 8.0))
            continue

        if field in NUMERIC_FIELDS:
            diff = abs(float(student_value) - float(mate_value))
            if diff <= 1:
                matches.append(
                    f"{label}"
                )
            else:
                mismatches.append(
                    f"{label}"
                )
                impacts.append((label, max(diff, 1.0)))
        else:
            if str(student_value) == str(mate_value):
                matches.append(f"{label}")
            else:
                mismatches.append(f"{label}")
                impacts.append((label, 6.0))

    if not matches:
        matches.append("No exact/close trait matches found in this pairing.")
    if not mismatches:
        mismatches.append("No major mismatches found.")

    impact_sorted = sorted(impacts, key=lambda item: item[1], reverse=True)[:10]
    return matches, mismatches, impact_sorted


def _binary_graph(student_row: pd.Series, roommate_row: pd.Series):
    details = []
    matched = 0

    for field in BINARY_GRAPH_FIELDS:
        left = int(student_row[field])
        right = int(roommate_row[field])
        is_match = left == right
        if is_match:
            matched += 1
        details.append((DISPLAY_NAMES[field], left, right, is_match))

    total = len(BINARY_GRAPH_FIELDS)
    match_pct = 0 if total == 0 else int(round((matched / total) * 100))
    mismatch_pct = 100 - match_pct

    circumference = 2 * math.pi * 44
    match_arc = circumference * (match_pct / 100)
    mismatch_arc = max(circumference - match_arc, 0.0)

    rows = []
    for label, left, right, is_match in details:
        state_class = "ok" if is_match else "bad"
        rows.append(
            '<div class="mini-row">'
            f'<p>{escape(label)}</p>'
            f'<span class="state {state_class}">{left} vs {right}</span>'
            '</div>'
        )

    return (
        '<div class="split-graph">'
        '<svg viewBox="0 0 140 140" class="ring" role="img" aria-label="Binary feature match ratio">'
        '<circle cx="70" cy="70" r="44" class="ring-bg" />'
        f'<circle cx="70" cy="70" r="44" class="ring-match" style="stroke-dasharray: {match_arc:.2f} {circumference:.2f}" />'
        f'<circle cx="70" cy="70" r="44" class="ring-miss" style="stroke-dasharray: {mismatch_arc:.2f} {circumference:.2f}; stroke-dashoffset: -{match_arc:.2f}" />'
        f'<text x="70" y="66" class="ring-score">{match_pct}%</text>'
        '<text x="70" y="82" class="ring-sub">match</text>'
        '</svg>'
        '<div class="split-meta">'
        f'<p><strong>{matched}</strong> matched, <strong>{total - matched}</strong> mismatched</p>'
        f'<p>Mismatch share: {mismatch_pct}%</p>'
        '<div class="mini-grid">'
        + "".join(rows)
        + '</div></div></div>'
    )


def _numerical_graph(student_row: pd.Series, roommate_row: pd.Series, df_students: pd.DataFrame):
    rows = []
    graph_data = []

    for field in NUMERIC_GRAPH_FIELDS:
        left = float(student_row[field])
        right = float(roommate_row[field])
        raw_diff = abs(left - right)
        if field in TIME_CYCLIC_FIELDS:
            # Wrap clock values across midnight for realistic time distance.
            diff = min(raw_diff, 24 - raw_diff)
            ratio = min(diff / 12.0, 1.0)
        else:
            diff = raw_diff
            range_span = float(df_students[field].max()) - float(df_students[field].min())
            ratio = 0.0 if range_span == 0 else min(diff / range_span, 1.0)

        graph_data.append((field, left, right, diff, ratio))

    for field, left, right, diff, ratio in graph_data:
        width = ratio * 100
        rows.append(
            '<div class="num-row">'
            f'<p>{escape(DISPLAY_NAMES[field])}</p>'
            '<div class="num-track">'
            f'<span style="--w: {width:.0f}%"></span>'
            '</div>'
            f'<small>{left:g} vs {right:g} (diff {diff:g})</small>'
            '</div>'
        )

    return '<div class="num-graph">' + "".join(rows) + '</div>'


def _categorical_graph(student_row: pd.Series, roommate_row: pd.Series):
    cards = []
    matched = 0

    for field in OTHER_CATEGORICAL_FIELDS:
        left = str(student_row[field])
        right = str(roommate_row[field])
        is_match = left == right
        if is_match:
            matched += 1
        state = "match" if is_match else "mismatch"
        cards.append(
            f'<div class="cat-card {state}">'
            f'<h4>{escape(DISPLAY_NAMES[field])}</h4>'
            f'<p>{escape(left)} vs {escape(right)}</p>'
            '</div>'
        )

    total = len(OTHER_CATEGORICAL_FIELDS)
    summary = f"{matched}/{total} categorical traits match"
    return '<div class="cat-graph"><p class="cat-summary">' + escape(summary) + '</p><div class="cat-grid">' + "".join(cards) + '</div></div>'


def _one_vs_one_rows(student_row: pd.Series, roommate_row: pd.Series, df_students: pd.DataFrame):
    rows = []
    fields = list(NUMERIC_GRAPH_FIELDS)

    for field in fields:
        minimum = float(df_students[field].min())
        maximum = float(df_students[field].max())
        if maximum == minimum:
            student_w = 50
            mate_w = 50
        else:
            student_w = ((float(student_row[field]) - minimum) / (maximum - minimum)) * 100
            mate_w = ((float(roommate_row[field]) - minimum) / (maximum - minimum)) * 100

        label = DISPLAY_NAMES.get(field, field.replace("_", " ").title())
        rows.append(
            '<div class="ovr-row">'
            f'<div class="ovr-track"><span class="ovr-fill student" style="--w: {student_w:.0f}%"></span></div>'
            '<div class="ovr-mid">'
            f"<strong>{escape(label)}</strong><br/>"
            f"{escape(str(student_row[field]))} vs {escape(str(roommate_row[field]))}"
            '</div>'
            f'<div class="ovr-track"><span class="ovr-fill mate" style="--w: {mate_w:.0f}%"></span></div>'
            '</div>'
        )

    return "".join(rows)


def _render_app(student_id: int | None = None, searched: bool = False) -> str:
    template = _load_frontend_template()
    students_csv_file = get_active_students_csv()

    default_status = ""

    replacements = {
        "__RESULTS_CLASS__": "hidden-presearch",
        "<!--__STATUS_BLOCK__-->": default_status,
        "<!--__ADMIN_ACTIONS__-->": "",
        "<!--__REVIEW_PANEL__-->": '<article class="panel review-panel"><h3>Roommate Review</h3><p class="panel-note">Search for a student to view or submit feedback.</p></article>',
        "<!--__MATCH_ITEMS__-->": '<li>No search yet.</li>',
        "<!--__MISMATCH_ITEMS__-->": '<li>No search yet.</li>',
        "<!--__ONE_VS_ONE_ROWS__-->": '<p class="panel-note">Search for a student to see 1v1 comparison bars.</p>',
        "<!--__RADAR_SVG__-->": '<p class="panel-note">Radar chart appears after a valid student search.</p>',
        "<!--__IMPACT_ROWS__-->": '<div class="bar-row"><p>No mismatch data yet.</p><span style="--w: 0%"></span></div>',
        "<!--__NUMERIC_GRAPH__-->": '<p class="panel-note">Numerical comparison graph appears after search.</p>',
        "<!--__BINARY_GRAPH__-->": '<p class="panel-note">Binary feature graph appears after search.</p>',
        "<!--__CATEGORICAL_GRAPH__-->": '<p class="panel-note">Categorical comparison graph appears after search.</p>',
    }

    if student_id is not None:
        template = template.replace(
            'id="student_id" name="student_id" type="number" placeholder="Eg: 1054" required',
            f'id="student_id" name="student_id" type="number" value="{student_id}" placeholder="Eg: 1054" required'
        )

    if not searched:
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    if student_id is None:
        replacements["<!--__STATUS_BLOCK__-->"] = (
            '<div class="status error">'
            'Please enter a valid student ID and search again.'
            '</div>'
        )
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    if not os.path.exists(students_csv_file):
        replacements["<!--__STATUS_BLOCK__-->"] = (
            '<div class="status error">'
            f"Data file missing at {escape(students_csv_file)}."
            '</div>'
        )
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    df_students = pd.read_csv(students_csv_file)
    student_data = df_students[df_students["student_id"] == student_id]

    if student_data.empty:
        replacements["<!--__STATUS_BLOCK__-->"] = (
            '<div class="status error">'
            f"Student ID {student_id} was not found in the CSV dataset."
            '</div>'
        )
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    assignment = _assignment_for_student(student_id)
    if not assignment:
        replacements["<!--__STATUS_BLOCK__-->"] = (
            '<div class="status error">'
            'Assignments are not ready yet. Run backend flow POST /api/admin/test-feedback-retrain (or POST /api/admin/run-matching) from API/admin side.'
            '</div>'
        )
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    if assignment["roommate_id"] is None:
        replacements["<!--__STATUS_BLOCK__-->"] = (
            '<div class="status error">'
            f"Student ID: {student_id}<br/>"
            'No roommate assigned for this student yet.'
            '</div>'
        )
        replacements["<!--__REVIEW_PANEL__-->"] = _review_panel_html(student_id, assignment)
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    roommate_id = int(assignment["roommate_id"])
    roommate_data = df_students[df_students["student_id"] == roommate_id]

    if roommate_data.empty:
        replacements["<!--__STATUS_BLOCK__-->"] = (
            '<div class="status error">'
            f"Roommate profile (ID {roommate_id}) not found in students.csv."
            '</div>'
        )
        html = template
        for key, value in replacements.items():
            html = html.replace(key, value)
        return html

    student_row = student_data.iloc[0]
    roommate_row = roommate_data.iloc[0]

    matches, mismatches, impacts = _build_comparison(student_row, roommate_row)

    replacements["<!--__STATUS_BLOCK__-->"] = (
        '<div class="status ok">'
        f"Student ID: {student_id}<br/>"
        f"Roommate ID: {roommate_id}<br/>"
        f"Compatibility: {assignment['compatibility_score']}<br/>"
        f"Gender: {escape(str(student_row['gender']))} vs {escape(str(roommate_row['gender']))}<br/>"
        f"Year of Study: {escape(str(student_row['year_of_study']))} vs {escape(str(roommate_row['year_of_study']))}"
        '</div>'
    )

    replacements["<!--__REVIEW_PANEL__-->"] = _review_panel_html(student_id, assignment)
    replacements["<!--__MATCH_ITEMS__-->"] = "".join(f"<li>{escape(text)}</li>" for text in matches)
    replacements["__RESULTS_CLASS__"] = ""
    replacements["<!--__MISMATCH_ITEMS__-->"] = "".join(f"<li>{escape(text)}</li>" for text in mismatches)
    replacements["<!--__ONE_VS_ONE_ROWS__-->"] = _one_vs_one_rows(student_row, roommate_row, df_students)
    replacements["<!--__RADAR_SVG__-->"] = _radar_svg(student_row, roommate_row, df_students)
    replacements["<!--__NUMERIC_GRAPH__-->"] = _numerical_graph(student_row, roommate_row, df_students)
    replacements["<!--__BINARY_GRAPH__-->"] = _binary_graph(student_row, roommate_row)
    replacements["<!--__CATEGORICAL_GRAPH__-->"] = _categorical_graph(student_row, roommate_row)

    if impacts:
        max_impact = max(score for _, score in impacts)
        rows = []
        for label, score in impacts:
            width = 0 if max_impact == 0 else (score / max_impact) * 100
            rows.append(
                '<div class="bar-row">'
                f"<p>{escape(label)}</p>"
                f'<span style="--w: {width:.0f}%"></span>'
                '</div>'
            )
        replacements["<!--__IMPACT_ROWS__-->"] = "".join(rows)
    else:
        replacements["<!--__IMPACT_ROWS__-->"] = '<div class="bar-row"><p>No mismatch data.</p><span style="--w: 0%"></span></div>'

    html = template
    for key, value in replacements.items():
        html = html.replace(key, value)
    return html


@app.get("/")
def read_root():
    return {"message": "Roommate Matching API is running."}


@app.get("/app", response_class=HTMLResponse)
def serve_frontend(
    student_id: int | None = Query(default=None),
    searched: int | None = Query(default=None)
):
    return HTMLResponse(content=_render_app(student_id, searched=bool(searched)))


# 2. ADMIN ENDPOINT: Trigger the matching process from the local CSV
@app.post("/api/admin/run-matching")
async def run_matching():
    students_csv_file = get_active_students_csv()

    # Verify you actually placed the file in the right spot before running
    if not os.path.exists(students_csv_file):
        raise HTTPException(
            status_code=404, 
            detail=(
                "Data file not found. Place 'predict_data.csv' (preferred) or "
                f"'students.csv' inside {DATA_DIR}."
            )
        )
    
    try:
        print(f"Reading student data from {students_csv_file}...")
        df_students = pd.read_csv(students_csv_file)
        
        # Load the trained model using your Singleton loader
        model = get_model()
        
        # Run the matching algorithm
        print("Starting the NetworkX matching pipeline...")
        matching_results = find_optimal_roommates(df_students, model)

        with SessionLocal() as db:
            sync_users_from_dataframe(db, df_students)
            persistence_result = persist_matching_results(db, matching_results, source="manual_matching")
            db.commit()
            
        return {
            "message": "Roommate matching completed successfully!",
            "summary": {
                "total_pairs": matching_results["total_pairs"],
                "average_hostel_score": matching_results["average_hostel_score"],
            },
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during matching: {str(e)}")


# 3. STUDENT ENDPOINT: Search for assigned roommate
@app.get("/api/student/match/{student_id}")
async def get_roommate(student_id: int):
    with SessionLocal() as db:
        stored_match = get_current_assignment_record(db, student_id)
        if stored_match:
            return {
                "your_id": student_id,
                "roommate_id": stored_match.roommate_id,
                "compatibility_score": stored_match.compatibility_score,
                "source": stored_match.source,
            }

    # Check if the admin has run the matching yet
    if not os.path.exists(ASSIGNMENTS_FILE):
        raise HTTPException(
            status_code=404, 
            detail="Roommate assignments have not been generated yet. Admin must run the matching pipeline first."
        )
        
    with open(ASSIGNMENTS_FILE, "r") as f:
        data = json.load(f)
        
    assignments = data.get("assignments", [])
    
    # Search for the student in the pairs
    for pair in assignments:
        if pair["student_1"] == student_id:
            return {"your_id": student_id, "roommate_id": pair["student_2"], "compatibility_score": pair["compatibility_score"]}
        elif pair["student_2"] == student_id:
            return {"your_id": student_id, "roommate_id": pair["student_1"], "compatibility_score": pair["compatibility_score"]}
            
    # If the loop finishes without returning, the student ID wasn't found
    raise HTTPException(status_code=404, detail=f"No roommate assignment found for student ID {student_id}.")


@app.post("/feedback", response_model=FeedbackResponse, status_code=201)
async def submit_feedback(payload: FeedbackCreate, db: Session = Depends(get_db)):
    if payload.user_id == payload.roommate_id:
        raise HTTPException(status_code=400, detail="user_id and roommate_id must be different.")

    users = db.query(User).filter(User.id.in_([payload.user_id, payload.roommate_id])).all()
    found_ids = {user.id for user in users}
    missing_ids = [user_id for user_id in [payload.user_id, payload.roommate_id] if user_id not in found_ids]
    if missing_ids:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown user IDs in feedback payload: {missing_ids}",
        )

    current_assignment = get_current_assignment_record(db, payload.user_id)
    if current_assignment is None:
        raise HTTPException(status_code=404, detail="No active roommate assignment found for this user.")
    if current_assignment.roommate_id != payload.roommate_id:
        raise HTTPException(
            status_code=400,
            detail="Feedback can be submitted only for the user's currently assigned roommate.",
        )

    active_cycle = int(current_assignment.matching_cycle)

    existing_feedback = get_feedback_for_cycle(
        db,
        user_id=payload.user_id,
        roommate_id=payload.roommate_id,
        matching_cycle=active_cycle,
    )
    if existing_feedback:
        raise HTTPException(
            status_code=409,
            detail=(
                "Feedback already submitted for this roommate match."
            ),
        )

    feedback = FeedbackStaging(
        user_id=payload.user_id,
        roommate_id=payload.roommate_id,
        matching_cycle=active_cycle,
        feedback_score=payload.feedback_score,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    return feedback


@app.post("/api/admin/run-feedback-batch", response_model=BatchJobResult)
async def run_feedback_batch(db: Session = Depends(get_db)):
    try:
        return run_feedback_batch_job(db)
    except Exception as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch retraining failed: {exc}") from exc


@app.post("/api/admin/test-feedback-retrain")
async def test_feedback_retrain(
    seed_feedback_count: int = Query(default=20, ge=1, le=500),
):
    """Backend-only QA helper: run matching, seed one-time feedback, then run batch retraining."""
    students_csv_file = get_active_students_csv()
    if not os.path.exists(students_csv_file):
        raise HTTPException(
            status_code=404,
            detail=f"Student data file not found at {students_csv_file}",
        )

    try:
        df_students = pd.read_csv(students_csv_file)
        model = get_model()
        matching_results = find_optimal_roommates(df_students, model)

        with SessionLocal() as db:
            sync_users_from_dataframe(db, df_students)
            persistence_result = persist_matching_results(
                db,
                matching_results,
                source="test_seed_matching",
            )
            db.commit()

            active_cycle = int(persistence_result["matching_cycle"])
            seeded_count = 0

            for assignment in matching_results.get("assignments", []):
                if seeded_count >= seed_feedback_count:
                    break

                roommate_id = assignment.get("student_2")
                if roommate_id is None:
                    continue

                user_id = int(assignment["student_1"])
                roommate_id = int(roommate_id)

                existing_feedback = get_feedback_for_cycle(
                    db,
                    user_id=user_id,
                    roommate_id=roommate_id,
                    matching_cycle=active_cycle,
                )
                if existing_feedback:
                    continue

                compatibility_score = assignment.get("compatibility_score")
                score_value = 75.0 if compatibility_score is None else float(compatibility_score)
                score_value = max(0.0, min(100.0, score_value))

                db.add(
                    FeedbackStaging(
                        user_id=user_id,
                        roommate_id=roommate_id,
                        matching_cycle=active_cycle,
                        feedback_score=score_value,
                    )
                )
                seeded_count += 1

            db.commit()
            batch_result = run_feedback_batch_job(db)

            return {
                "message": "Test retraining flow executed.",
                "seed_feedback_count": seeded_count,
                "seed_target": seed_feedback_count,
                "seed_cycle": active_cycle,
                "matching_summary": {
                    "total_pairs": matching_results.get("total_pairs"),
                    "average_hostel_score": matching_results.get("average_hostel_score"),
                },
                "batch_result": batch_result,
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Test retrain flow failed: {exc}") from exc
