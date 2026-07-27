"""
Disease Recommendation Module (Proposal Section 3.10)

A rule-based expert system mapping classification outputs to detailed agronomic
guidance tailored for Nepalese maize cultivation.

Features:
  - Robust String Normalization (handles case differences, spaces, and underscores).
  - Class Alias Mapping (maps generic model labels like 'Blight' to specific entries).
  - NARC / CIMMYT-aligned advisory guidelines.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ==============================================================================
# KNOWLEDGE BASE
# Keys are stored in lower-case with spaces for easy normalized matching.
# ==============================================================================
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, str]] = {
    "healthy": {
        "description": "No visible disease symptoms detected on the leaf surface.",
        "treatment": "No chemical application required.",
        "prevention": "Maintain regular field scouting, balanced NPK soil fertilization, "
                      "and optimal plant density (60 cm x 20 cm spacing) for airflow.",
        "severity": "Green",
    },
    "common rust": {
        "description": "Caused by Puccinia sorghi. Characterized by small, circular to elongated "
                       "cinnamon-brown pustules on both upper and lower leaf surfaces.",
        "treatment": "Apply Mancozeb 75% WP (2-2.5 g/L water) or a triazole/strobilurin fungicide "
                      "(e.g., Propiconazole at 1 mL/L) at first pustule appearance.",
        "prevention": "Plant resistant/tolerant hybrids (e.g., Rampur Hybrid-10); avoid excess nitrogen; "
                      "maintain proper row spacing.",
        "severity": "Yellow",
    },
    "gray leaf spot": {
        "description": "Caused by Cercospora zeae-maydis. Produces distinct, rectangular, "
                       "tan-to-gray lesions running strictly parallel to leaf veins.",
        "treatment": "Apply foliar fungicide like Azoxystrobin or Propiconazole (1-1.5 mL/L) at "
                      "symptom onset prior to tasseling, especially in humid hill microclimates.",
        "prevention": "Rotate crops with non-host legumes or mustard for 1-2 seasons; bury infected "
                      "stubble via deep tillage.",
        "severity": "Yellow",
    },
    "northern leaf blight": {
        "description": "Caused by Exserohilum turcicum. Characterized by large, elliptical, "
                       "cigar-shaped grayish-green to tan lesions on leaves.",
        "treatment": "Spray Mancozeb (2.5 g/L) or Tebuconazole (1 mL/L) if lesions appear before "
                      "tasseling and high humidity (>80%) persists.",
        "prevention": "Utilize NARC-recommended resistant maize varieties (e.g., Deuti, Sheetal); "
                      "rotate crops and manage crop residue.",
        "severity": "Yellow",
    },
    "southern leaf blight": {
        "description": "Caused by Bipolaris maydis. Produces small, elongated tan lesions with parallel "
                       "margins, spreading rapidly in warm, moist low-altitude/Terai regions.",
        "treatment": "Apply recommended foliar fungicides (Mancozeb or Zineb @ 2 g/L) upon early detection.",
        "prevention": "Avoid continuous maize mono-cropping; destroy infected crop debris post-harvest.",
        "severity": "Yellow",
    },
    "southern rust": {
        "description": "Caused by Puccinia polysora. Produces tiny, dense, orange-to-light-brown pustules "
                       "concentrated predominantly on the upper leaf surface.",
        "treatment": "Immediate application of systemic fungicides (e.g., Azoxystrobin + Difenoconazole) "
                      "is critical as Southern Rust spreads aggressively in high temperatures.",
        "prevention": "Scout fields regularly during warm monsoon months; avoid late-season planting.",
        "severity": "Red",
    },
    "banded leaf and sheath blight": {
        "description": "Caused by Rhizoctonia solani f. sp. sasakii. Displays prominent, irregular brown/white "
                       "banded lesions on sheaths and lower leaves.",
        "treatment": "Apply Validamycin 3% L (2 mL/L) or Carbendazim (1 g/L) directed at the lower leaf sheaths.",
        "prevention": "Ensure good field drainage to avoid standing water; rogue and burn severely infected plants.",
        "severity": "Red",
    },
    "maize streak virus": {
        "description": "Viral infection transmitted by Cicadulina leafhoppers, causing continuous "
                       "yellowish-white chlorotic streaks along leaf veins.",
        "treatment": "No direct cure for viral infection. Control vector leafhoppers using Imidacloprid (0.5 mL/L) "
                      "to limit transmission to healthy stands.",
        "prevention": "Plant virus-resistant seed stock; destroy infected rogue plants early; control weed vectors.",
        "severity": "Red",
    },
    "brown spot": {
        "description": "Caused by Physoderma maydis. Produces small, yellowish to reddish-brown spots in bands "
                       "across leaf blades, midribs, and sheaths.",
        "treatment": "Improve field drainage immediately. Apply Mancozeb (2 g/L) if lesions spread rapidly to upper canopy.",
        "prevention": "Avoid waterlogging; use balanced potassium fertilization to enhance stalk strength.",
        "severity": "Yellow",
    },
    "downy mildew": {
        "description": "Caused by Peronosclerospora species. Causes chlorotic leaf streaking accompanied by "
                       "a white downy growth on leaf undersides, leading to severe stunting.",
        "treatment": "Systemic rogueing: Pull out and burn infected plants immediately. Treat seeds with Metalaxyl "
                      "for future cropping cycles.",
        "prevention": "Use certified disease-free seed; maintain strict crop rotation regimes.",
        "severity": "Red",
    },
}

# ==============================================================================
# ALIAS MAPPINGS
# Maps alternative model string outputs to standard keys in the knowledge base.
# ==============================================================================
CLASS_ALIASES: Dict[str, str] = {
    "blight": "northern leaf blight",
    "gray_leaf_spot": "gray leaf spot",
    "common_rust": "common rust",
    "northern_leaf_blight": "northern leaf blight",
    "southern_leaf_blight": "southern leaf blight",
    "southern_rust": "southern rust",
    "banded_leaf_and_sheath_blight": "banded leaf and sheath blight",
    "maize_streak_virus": "maize streak virus",
    "brown_spot": "brown spot",
    "downy_mildew": "downy mildew",
}

_DEFAULT_RECOMMENDATION: Dict[str, str] = {
    "description": "Detailed agronomic information is currently being expanded for this classification.",
    "treatment": "Please consult a local agricultural extension officer (Krishi Gyan Kendra) for field assessment.",
    "prevention": "Maintain standard field sanitation, balanced fertilization, and routine monitoring.",
    "severity": "Yellow",
}


def normalize_label(raw_label: str) -> str:
    """
    Normalizes input label strings:
      - Converts to lowercase.
      - Replaces underscores and hyphens with spaces.
      - Strips whitespace.
    """
    if not raw_label or not isinstance(raw_label, str):
        return ""
    
    clean = raw_label.lower().replace("_", " ").replace("-", " ")
    return " ".join(clean.split())  # removes redundant internal spaces


def get_recommendation(disease_label: str) -> Dict[str, str]:
    """
    Retrieves recommendation details for a given disease classification label.

    Args:
        disease_label (str): Raw output string from classification pipeline
                             (e.g., "Gray_Leaf_Spot", "blight", "Common Rust").

    Returns:
        dict: Recommendation dict containing 'description', 'treatment', 
              'prevention', and 'severity'.
    """
    normalized_key = normalize_label(disease_label)

    if not normalized_key:
        logger.warning("Empty or invalid disease_label passed to get_recommendation().")
        return _DEFAULT_RECOMMENDATION.copy()

    # 1. Direct Lookup
    if normalized_key in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[normalized_key].copy()

    # 2. Alias Lookup (handles models outputting short labels like 'blight')
    mapped_key = CLASS_ALIASES.get(normalized_key)
    if mapped_key and mapped_key in DISEASE_KNOWLEDGE_BASE:
        logger.info("Mapped label alias '%s' -> '%s'", disease_label, mapped_key)
        return DISEASE_KNOWLEDGE_BASE[mapped_key].copy()

    # 3. Partial Fallback Search
    for key in DISEASE_KNOWLEDGE_BASE:
        if key in normalized_key or normalized_key in key:
            logger.info("Matched partial key '%s' for raw label '%s'", key, disease_label)
            return DISEASE_KNOWLEDGE_BASE[key].copy()

    # 4. Final Fallback
    logger.warning("Unmapped disease label received: '%s'. Returning fallback.", disease_label)
    return _DEFAULT_RECOMMENDATION.copy()