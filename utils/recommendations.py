"""
Disease Recommendation Module (proposal section 3.10).

A rule-based expert system. For each detected disease class, returns:
  - description: causal organism + observable symptoms
  - treatment: fungicide/biological treatment with dosage/timing
  - prevention: agronomic prevention practices
  - severity: "Green" (no action) | "Yellow" (moderate) | "Red" (immediate)

IMPORTANT: The text below is a starting structure only. Before using this
in a real advisory tool, have the dosage/fungicide details reviewed against
current NARC Nepal / FAO / CIMMYT guidance, since recommended products and
dosages change over time and by region. Treat these as placeholders to be
verified, not final agronomic advice.
"""

DISEASE_KNOWLEDGE_BASE = {
    "Healthy": {
        "description": "No visible disease symptoms detected on the leaf surface.",
        "treatment": "No treatment required.",
        "prevention": "Continue regular field monitoring, balanced fertilization, "
                       "and proper crop spacing to maintain plant health.",
        "severity": "Green",
    },
    "Common Rust": {
        "description": "Caused by the fungus Puccinia sorghi. Appears as small, "
                        "circular to elongated reddish-brown pustules scattered on "
                        "both leaf surfaces.",
        "treatment": "Apply a triazole or strobilurin-based fungicide at early "
                      "pustule appearance, following label dosage. Repeat per "
                      "product interval if pressure continues.",
        "prevention": "Plant resistant/tolerant maize varieties where available; "
                       "avoid excessive nitrogen application; ensure adequate row "
                       "spacing for airflow.",
        "severity": "Yellow",
    },
    "Gray Leaf Spot": {
        "description": "Caused by Cercospora zeae-maydis. Produces narrow, "
                        "rectangular, tan-to-gray lesions that run parallel to "
                        "leaf veins.",
        "treatment": "Apply a foliar fungicide (e.g., strobilurin or triazole "
                      "class) at first symptom onset, especially under humid "
                      "conditions.",
        "prevention": "Rotate crops away from maize for at least one season; "
                       "manage crop residue through tillage or removal; avoid "
                       "dense planting.",
        "severity": "Yellow",
    },
    "Northern Leaf Blight": {
        "description": "Caused by Exserohilum turcicum. Characterized by long, "
                        "cigar-shaped gray-green to tan lesions on leaves.",
        "treatment": "Apply recommended fungicide if lesions appear before "
                      "tasseling and conditions remain humid; prioritize fields "
                      "with history of severe infection.",
        "prevention": "Use resistant hybrids; rotate crops; manage residue from "
                       "previous infected maize crop.",
        "severity": "Yellow",
    },
    "Southern Leaf Blight": {
        "description": "Caused by Bipolaris maydis. Produces tan, elongated "
                        "lesions with parallel margins, often more severe in "
                        "warm, humid conditions than Northern Leaf Blight.",
        "treatment": "Apply fungicide promptly if detected during early vegetative "
                      "stages; remove and destroy heavily infected plant debris.",
        "prevention": "Avoid continuous maize cropping; plant resistant varieties; "
                       "ensure balanced soil fertility.",
        "severity": "Yellow",
    },
    "Southern Rust": {
        "description": "Caused by Puccinia polysora. Produces small, orange to "
                        "tan pustules, typically denser than Common Rust and "
                        "concentrated on the upper leaf surface.",
        "treatment": "Apply fungicide immediately upon detection, as Southern "
                      "Rust can spread rapidly under warm, humid conditions and "
                      "cause significant yield loss.",
        "prevention": "Monitor fields closely during warm/humid periods; use "
                       "resistant hybrids; avoid late planting in high-risk zones.",
        "severity": "Red",
    },
    "Banded Leaf and Sheath Blight": {
        "description": "Caused by Rhizoctonia solani. Produces irregular brown "
                        "banded lesions on leaf sheaths, often spreading upward "
                        "to leaves under high humidity.",
        "treatment": "Apply a recommended fungicide to affected sheaths and "
                      "surrounding plants; remove severely infected plants where "
                      "feasible.",
        "prevention": "Avoid water stagnation in fields; maintain adequate plant "
                       "spacing; rotate with non-host crops.",
        "severity": "Red",
    },
    "Maize Streak Virus": {
        "description": "A viral disease transmitted by leafhoppers, causing "
                        "narrow yellow-to-white streaks running parallel to the "
                        "leaf veins.",
        "treatment": "No direct chemical cure for the virus; control leafhopper "
                      "vector populations using approved insecticides to limit "
                      "spread to healthy plants.",
        "prevention": "Plant virus-resistant varieties; remove and destroy "
                       "infected plants early; control weeds that host "
                       "leafhoppers near field borders.",
        "severity": "Red",
    },
    "Brown Spot": {
        "description": "Caused by Physoderma maydis. Appears as small, round to "
                        "oval brown spots, often in bands across the leaf, "
                        "sometimes extending to stalks.",
        "treatment": "Apply fungicide if spots are spreading rapidly; ensure "
                      "field drainage is improved to reduce moisture buildup.",
        "prevention": "Avoid waterlogged fields; rotate crops; use balanced "
                       "fertilization to strengthen plant resistance.",
        "severity": "Yellow",
    },
    "Downy Mildew": {
        "description": "Caused by various Peronosclerospora species. Produces "
                        "chlorotic streaking and a downy white-to-gray fungal "
                        "growth, often causing stunted, distorted plants.",
        "treatment": "Remove and destroy infected plants immediately to limit "
                      "spread; apply systemic fungicide seed treatment for future "
                      "plantings.",
        "prevention": "Use certified disease-free seed; rotate crops; avoid "
                       "planting in fields with known history of downy mildew.",
        "severity": "Red",
    },
}

# Fallback used only if an unexpected label is ever produced
_DEFAULT_RECOMMENDATION = {
    "description": "Disease information not available for this class.",
    "treatment": "Please consult a local agricultural extension officer.",
    "prevention": "Maintain general good agronomic practices.",
    "severity": "Yellow",
}


def get_recommendation(disease_label: str) -> dict:
    """
    Looks up the structured recommendation for a given disease label.

    Args:
        disease_label: Must exactly match a key used in CLASS_NAMES (app.py).

    Returns:
        dict with keys: description, treatment, prevention, severity
    """
    return DISEASE_KNOWLEDGE_BASE.get(disease_label, _DEFAULT_RECOMMENDATION)
