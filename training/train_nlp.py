"""
train_nlp.py
------------
Trains a TF-IDF + LinearSVC text classifier for CivicTrack.

Run locally, in Google Colab, or on any machine with Python 3.8+:
  python train_nlp.py

Output: models/classifier.pkl

Categories (must match UI dropdown exactly):
  Roads | Garbage | Road Cracks | Other
"""

import os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
import nltk
nltk.download("stopwords", quiet=True)

warnings.filterwarnings("ignore", category=UserWarning)

# ── Training data (~60+ samples per category) ────────────────────────────────

RAW_DATA = [
    # ══ Roads ════════════════════════════════════════════════════════════════
    ("Large pothole on the main road near the market causing accidents", "Roads"),
    ("Road is severely damaged with multiple potholes after monsoon", "Roads"),
    ("Deep pothole near school zone is dangerous for children", "Roads"),
    ("Crater on the highway needs immediate repair", "Roads"),
    ("Road surface broken and uneven causing vehicle damage", "Roads"),
    ("Tar road has multiple potholes along the stretch", "Roads"),
    ("Speed breaker damaged and metal rods exposed", "Roads"),
    ("Multiple depressions on the road causing two-wheeler accidents", "Roads"),
    ("Road dug up by municipality and not repaired for weeks", "Roads"),
    ("Pothole filling done badly and road broken again", "Roads"),
    ("Potholes flooded with water making them invisible", "Roads"),
    ("Road cave-in near drainage line on main road", "Roads"),
    ("Broken road shoulder causing vehicles to fall into ditch", "Roads"),
    ("Pothole size is growing and nobody is repairing it", "Roads"),
    ("Road near my house has not been repaired for two years", "Roads"),
    ("Severe road damage due to heavy vehicle movement", "Roads"),
    ("Dangerous sinkhole appeared on colony road overnight", "Roads"),
    ("Bumpy road surface near hospital entrance causing patient distress", "Roads"),
    ("Road patches wearing off within a month of repair", "Roads"),
    ("Two-lane road narrowed due to craters on both sides", "Roads"),
    ("Pedestrian sidewalk cracked and uneven near shopping complex", "Roads"),
    ("Sharp-edged pothole burst my car tyre last night", "Roads"),
    ("Muddy patches on road after incomplete construction work", "Roads"),
    ("Huge road gap where concrete slabs have shifted apart", "Roads"),
    ("Colony internal road has never been asphalted properly", "Roads"),
    ("Newly laid road already showing potholes within weeks", "Roads"),
    ("Highway divider broken exposing iron rods on road surface", "Roads"),
    ("Road near railway crossing is extremely bumpy and uneven", "Roads"),
    ("Gravel road turning into slush during monsoon making it impassable", "Roads"),
    ("Main arterial road in city centre has deep craters everywhere", "Roads"),
    ("Service road riddled with potholes trucks avoid using it", "Roads"),
    ("Bridge approach road damaged badly needs urgent repair", "Roads"),
    ("Potholes near flyover ramp causing sudden braking hazards", "Roads"),
    ("Road subsidence beside new metro construction causing accidents", "Roads"),
    ("Continuous vibrations from cracked highway damaging nearby houses", "Roads"),
    ("School van gets stuck daily in pothole near gate", "Roads"),

    # ══ Garbage ══════════════════════════════════════════════════════════════
    ("Garbage not collected for three days in our area", "Garbage"),
    ("Waste dumped on roadside near the park entrance", "Garbage"),
    ("Overflowing dustbin outside market creating health hazard", "Garbage"),
    ("Illegal dumping of construction debris in open plot", "Garbage"),
    ("Sanitation workers not picking garbage regularly", "Garbage"),
    ("Garbage heap growing near the school boundary wall", "Garbage"),
    ("Stray dogs spreading garbage all over the street", "Garbage"),
    ("Waste bins full and not emptied since Monday", "Garbage"),
    ("Open dumping of household waste creating foul smell", "Garbage"),
    ("Littering near park creating mess and mosquito breeding", "Garbage"),
    ("Garbage collection vehicle has not come this week", "Garbage"),
    ("Municipality workers dumped garbage on road instead of landfill", "Garbage"),
    ("Burning of garbage causing smoke and health issues", "Garbage"),
    ("Abandoned garbage pile near water body", "Garbage"),
    ("Garbage chute blocked and waste overflowing on stairs", "Garbage"),
    ("Plastic waste clogging storm drain near building", "Garbage"),
    ("Dead animals not removed by sanitation department", "Garbage"),
    ("Bio-medical waste illegally dumped on roadside", "Garbage"),
    ("Fruit market vendors dumping waste near residential area", "Garbage"),
    ("Municipal van dumps garbage near school daily morning", "Garbage"),
    ("Heaps of trash piling up at community bin for a week", "Garbage"),
    ("Rotting food waste attracting rats near apartment complex", "Garbage"),
    ("Construction rubble left on footpath after building work", "Garbage"),
    ("Used syringes found among trash near hospital boundary", "Garbage"),
    ("Empty liquor bottles and litter at park every morning", "Garbage"),
    ("E-waste including old monitors dumped behind bus shelter", "Garbage"),
    ("Overflowing skip bin on main road stinking for days", "Garbage"),
    ("Roadside vendors leave food waste every night after closing", "Garbage"),
    ("Neighbourhood drain blocked because of dumped plastic bags", "Garbage"),
    ("Residential area dustbins not replaced after damage", "Garbage"),
    ("Industrial waste being secretly dumped in empty plot at night", "Garbage"),
    ("Wedding decoration waste left on road for three days", "Garbage"),
    ("Festival garbage not cleaned despite complaints to ward office", "Garbage"),
    ("Garbage truck leaking waste liquid on road during transit", "Garbage"),
    ("Open compost pit near houses causing unbearable stench", "Garbage"),
    ("Litter scattered across playground children cannot play safely", "Garbage"),
    ("Cloth and textile waste dumped near river bank polluting water", "Garbage"),
    ("Market area remains filthy despite daily sweeping schedule", "Garbage"),
    ("Old mattresses and furniture dumped on footpath blocking walk", "Garbage"),
    ("Segregated waste mixed again by collection truck defeating purpose", "Garbage"),

    # ══ Road Cracks ══════════════════════════════════════════════════════════
    ("Road cracks running across carriageway near bus stop", "Road Cracks"),
    ("Asphalt has wide cracks and loose gravel", "Road Cracks"),
    ("National highway has dangerous cracks near bridge", "Road Cracks"),
    ("Road has alligator cracking pattern near junction", "Road Cracks"),
    ("Deep cracks in the road surface causing vehicle damage", "Road Cracks"),
    ("Longitudinal crack on road getting wider every week", "Road Cracks"),
    ("Transverse crack across entire road width near school", "Road Cracks"),
    ("Crack in road has exposed the base layer of gravel", "Road Cracks"),
    ("Pavement cracking badly near residential colony entrance", "Road Cracks"),
    ("Multiple hairline cracks appearing on newly built road", "Road Cracks"),
    ("Road surface showing spider web cracks after heavy rain", "Road Cracks"),
    ("Crack along road edge causing shoulder to break away", "Road Cracks"),
    ("Thermal cracks appearing on highway during summer heat", "Road Cracks"),
    ("Block cracking on parking lot surface near mall", "Road Cracks"),
    ("Reflective crack from old road showing through new overlay", "Road Cracks"),
    ("Wide crack in concrete road panel near flyover", "Road Cracks"),
    ("Fatigue cracking in road due to overloaded trucks", "Road Cracks"),
    ("Road crack filled with water creating slippery hazard", "Road Cracks"),
    ("Surface crack on bridge deck needs immediate attention", "Road Cracks"),
    ("Crack in divider road extending to both lanes", "Road Cracks"),
    ("Cement road slab cracked and pieces coming loose", "Road Cracks"),
    ("Cracked road near construction site getting worse daily", "Road Cracks"),
    ("Major crack running along median of main road", "Road Cracks"),
    ("Crack in bitumen surface causing water seepage underneath", "Road Cracks"),
    ("Corner crack at road intersection expanding rapidly", "Road Cracks"),
    ("Edge crack along drainage cover causing road to split", "Road Cracks"),
    ("Freshly repaired road already showing cracks within a week", "Road Cracks"),
    ("Road surface crumbling and cracking near petrol pump", "Road Cracks"),
    ("Cracks in service road spreading to main carriageway", "Road Cracks"),
    ("Zigzag cracks appeared after water pipeline repair work on road", "Road Cracks"),
    ("Concrete road slab has large diagonal crack near market", "Road Cracks"),
    ("Joint sealant missing between road slabs causing cracking", "Road Cracks"),
    ("Heavy cracking on approach road to railway overbridge", "Road Cracks"),
    ("Road crack near temple entrance growing wider and dangerous", "Road Cracks"),
    ("Complete network of cracks on road surface near college gate", "Road Cracks"),
    ("Road splitting into pieces along the bus route stretch", "Road Cracks"),
    ("Asphalt cracking and peeling off on internal colony road", "Road Cracks"),
    ("Road crack caused by tree root growth underneath pavement", "Road Cracks"),
    ("Cracks making two-wheeler riding very risky in this stretch", "Road Cracks"),
    ("Footpath concrete cracked and uneven near government office", "Road Cracks"),

    # ══ Other ════════════════════════════════════════════════════════════════
    ("Stray dogs attacking pedestrians near market area", "Other"),
    ("Illegal parking blocking narrow lane near school", "Other"),
    ("Noise pollution from construction site at night", "Other"),
    ("Broken park bench needs urgent repair", "Other"),
    ("Encroachment on footpath by roadside vendor", "Other"),
    ("Missing manhole cover near residential area is dangerous", "Other"),
    ("Tree fallen on road blocking traffic movement", "Other"),
    ("Unauthorized construction blocking emergency access road", "Other"),
    ("Public toilet not functional and in very bad condition", "Other"),
    ("Damaged traffic signal causing accidents at intersection", "Other"),
    ("Blood donation camp required urgently in our area", "Other"),
    ("Lost child found near bus stop please help identify", "Other"),
    ("Stray cattle on highway causing accidents", "Other"),
    ("Footpath tiles broken and dangerous for elderly", "Other"),
    ("Abandoned vehicle blocking entrance to society", "Other"),
    ("Graffiti and vandalism on public wall near school", "Other"),
    ("Overcrowding at bus stop no shelter during rains", "Other"),
    ("Noise from loudspeakers at odd hours causing disturbance", "Other"),
    ("Illegal advertisement banner blocking visibility at crossing", "Other"),
    ("Playground swings broken children getting injured while playing", "Other"),
    ("Stray monkey menace in residential colony people scared to go out", "Other"),
    ("Public drinking water fountain not working in park since months", "Other"),
    ("Dangerous open well in empty plot not covered or fenced", "Other"),
    ("Traffic congestion at school time no traffic police present", "Other"),
    ("Unauthorized hawkers blocking entire market footpath", "Other"),
    ("Community hall in bad condition ceiling leaking during functions", "Other"),
    ("Mosquito fogging not done in our area despite dengue cases", "Other"),
    ("Dangerous tree leaning over playground might fall on children", "Other"),
    ("Public library closed for repairs since six months no updates", "Other"),
    ("Bee hive on electric pole near bus stop posing risk to commuters", "Other"),
    ("Loud music from commercial establishment disturbing residents nightly", "Other"),
    ("Stray pigs roaming in residential area creating hygiene issues", "Other"),
    ("Boundary wall of government school collapsed needs rebuilding", "Other"),
    ("No proper signage at sharp turn on hill road accidents frequent", "Other"),
    ("Broken CCTV camera at public parking lot theft cases increasing", "Other"),
    ("Senior citizen bench in park vandalised needs replacement urgently", "Other"),
    ("Public bus stop shelter roof blown off during storm not repaired", "Other"),
    ("Speed limit sign missing near school zone vehicles drive fast", "Other"),
    ("Abandoned construction site became dumping ground and hideout", "Other"),
    ("Community garden fence broken cattle enter and destroy plants daily", "Other"),
]

# ── Build balanced DataFrame ──────────────────────────────────────────────────
df = pd.DataFrame(RAW_DATA, columns=["text", "category"])

max_count = df["category"].value_counts().max()
balanced_dfs = []
for cat in df["category"].unique():
    subset = df[df["category"] == cat]
    if len(subset) < max_count:
        subset = resample(subset, replace=True, n_samples=max_count,
                          random_state=42)
    balanced_dfs.append(subset)
df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total training samples: {len(df)}")
print(df["category"].value_counts())

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["category"],
    test_size=0.2, random_state=42, stratify=df["category"]
)

# ── Pipeline: TF-IDF + Calibrated LinearSVC ───────────────────────────────────
# LinearSVC is faster and often better on small text data than Naive Bayes.
# CalibratedClassifierCV wraps it to provide predict_proba().
base_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ("clf", LinearSVC(C=1.0, max_iter=5000, random_state=42))
])

# CalibratedClassifierCV gives us probability estimates
pipeline = CalibratedClassifierCV(base_pipeline, cv=3)
pipeline.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
print("\n── Classification Report ──────────────────────────────")
print(classification_report(y_test, y_pred, digits=3))

print("── Confusion Matrix ───────────────────────────────────")
labels = sorted(df["category"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(base_pipeline, df["text"], df["category"],
                         cv=cv, scoring="f1_macro")
print(f"\n5-Fold CV Macro F1: {scores.mean():.3f} ± {scores.std():.3f}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)
with open("models/classifier.pkl", "wb") as f:
    pickle.dump(pipeline, f)
print("\n✓ Saved → models/classifier.pkl")

# ── Quick demo ────────────────────────────────────────────────────────────────
demo_inputs = [
    "Huge pothole near my house causing accidents",
    "Deep cracks across the entire road surface near bridge",
    "Garbage not collected since three days in locality",
    "Water pipeline burst flooding the entire road",
    "No electricity in colony since yesterday night",
    "Stray dogs attacking people near park area",
    "Road has deep cracks and asphalt is completely broken",
    "Overflowing dustbin creating terrible smell near school",
    "Streetlight not working for two months in our lane",
    "Tree fallen blocking road after last storm",
    "Major crack running across entire road surface near bridge",
]
print("\n── Demo predictions ───────────────────────────────────")
for text in demo_inputs:
    pred  = pipeline.predict([text])[0]
    proba = pipeline.predict_proba([text])[0].max()
    flag  = "⚠ needs review" if proba < 0.65 else ""
    print(f"  [{pred:<16}] {proba:.2f}  '{text[:55]}' {flag}")
