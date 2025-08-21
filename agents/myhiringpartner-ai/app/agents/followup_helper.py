import re
from typing import Dict, List, Any


def _is_blank(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        return len(val.strip()) == 0 or val.strip().lower() in {"n/a", "na", "none", "not specified"}
    if isinstance(val, (list, tuple, set)):
        return len(val) == 0
    return False


def get_missing_candidate_fields(resume_full: Dict[str, Any]) -> List[str]:
    """
    Inspect a resume record (as returned from BigQuery) and determine which candidate
    fields are missing or blank. This mirrors the resumes table schema in bq_schema_manager.

    Returns a list of human-friendly field labels to ask the candidate.
    """
    missing: List[str] = []

    # Flat fields
    if _is_blank(resume_full.get("linkedin_url")):
        missing.append("LinkedIn Profile URL")

    # Ensure candidate_location is treated as a string
    candidate_location = resume_full.get("candidate_location")
    if _is_blank(candidate_location) or not isinstance(candidate_location, str):
        missing.append("Current Location")

    if _is_blank(resume_full.get("candidate_email")):
        missing.append("Email Address")
    if _is_blank(resume_full.get("candidate_phone")):
        missing.append("Phone Number")

    # Ensure date_of_birth is only a year
    dob = resume_full.get("date_of_birth")
    if _is_blank(dob) or not re.match(r"^\d{4}$", str(dob).strip()):
        missing.append("Year of Birth (e.g., 1990)")

    if _is_blank(resume_full.get("total_years_experience")):
        missing.append("Total Years of Experience")

    # Education (REPEATED RECORD): degree, institution, field_of_study, graduation_year
    education = resume_full.get("education") or []
    has_degree = False
    has_inst = False
    has_field = False
    has_grad_year = False
    has_masters = False
    has_bachelors = False
    if isinstance(education, list):
        for edu in education:
            try:
                deg = (edu or {}).get("degree")
                if not _is_blank(deg):
                    has_degree = True
                    # classify degree level
                    d = str(deg).strip().lower()
                    if re.search(r"\b(masters?|m\.?sc|m\.?tech|mba|mca)\b", d):
                        has_masters = True
                    if re.search(r"\b(bachelors?|b\.?sc|b\.?tech|be|beng|ba|bcom)\b", d):
                        has_bachelors = True
                inst = (edu or {}).get("institution")
                if not _is_blank(inst):
                    has_inst = True
                fos = (edu or {}).get("field_of_study")
                if not _is_blank(fos):
                    has_field = True
                gy = (edu or {}).get("graduation_year")
                if gy not in (None, ""):
                    has_grad_year = True
            except Exception:
                continue
    if not has_degree:
        missing.append("Highest Degree (e.g., Bachelors, Masters)")
    if not has_inst:
        missing.append("Education Institution")
    if not has_field:
        missing.append("Education Field of Study")
    if not has_grad_year:
        missing.append("Graduation Year")
    # If highest degree appears to be Masters but no Bachelor's found, ask explicitly for Bachelor's details
    if has_masters and not has_bachelors:
        missing.append("Bachelor's Degree Details (Degree, Institution, Field of Study, Graduation Year)")

    # Certifications (REPEATED STRING)
    certs = resume_full.get("certifications")
    if _is_blank(certs):
        missing.append("Certifications (if any)")

    # Work preferences (RECORD)
    wp = resume_full.get("work_preferences") or {}
    wtr = None
    if isinstance(wp, dict):
        # willing_to_relocate (BOOLEAN)
        wtr = wp.get("willing_to_relocate")
        if wtr is None:
            # only ask if unknown; if explicitly True/False, we consider it provided
            missing.append("Willing to Relocate (Yes/No)")

        # preferred_work_models (REPEATED STRING)
        pwm = wp.get("preferred_work_models")
        if _is_blank(pwm):
            missing.append("Preferred Work Models (Onsite/Remote/Hybrid)")

        # preferred_employment_types (REPEATED STRING)
        pet = wp.get("preferred_employment_types")
        if _is_blank(pet):
            missing.append("Preferred Employment Types (Full-time/Contract/etc.)")

        # preferred_locations (REPEATED STRING)
        plocs = wp.get("preferred_locations")
        if _is_blank(plocs):
            missing.append("Preferred Work Locations")

        # salary_expectations (STRING)
        if _is_blank(wp.get("salary_expectations")):
            missing.append("Salary Expectations")

    # Languages (REPEATED RECORD): language, proficiency
    languages = resume_full.get("languages")
    if _is_blank(languages):
        missing.append("Languages and Proficiency")
    else:
        # If present, see if any entry lacks required subfields
        needs_lang = False
        for lang in (languages or []):
            try:
                if _is_blank((lang or {}).get("language")) or _is_blank((lang or {}).get("proficiency")):
                    needs_lang = True
                    break
            except Exception:
                continue
        if needs_lang:
            missing.append("Languages and Proficiency")

    # Skill proficiency (REPEATED RECORD): skill, level, years_experience
    sp = resume_full.get("skill_proficiency")
    if _is_blank(sp):
        missing.append("Skills with Level and Years of Experience")
    else:
        needs_sp = False
        for s in (sp or []):
            try:
                if _is_blank((s or {}).get("skill")) or _is_blank((s or {}).get("level")) or ((s or {}).get("years_experience") in (None, "")):
                    needs_sp = True
                    break
            except Exception:
                continue
        if needs_sp:
            missing.append("Skills with Level and Years of Experience")

    # Experience (REPEATED RECORD)
    exp = resume_full.get("experience")
    if _is_blank(exp):
        missing.append("Work Experience (Company, Role, Dates, Location, Technologies Used)")
    else:
        needs_exp = False
        for e in (exp or []):
            try:
                if (
                    _is_blank((e or {}).get("company")) or
                    _is_blank((e or {}).get("role")) or
                    ((e or {}).get("start_date") in (None, "")) or
                    ((e or {}).get("end_date") in (None, "")) or
                    ((e or {}).get("duration_months") in (None, "")) or
                    _is_blank((e or {}).get("location")) or
                    _is_blank((e or {}).get("technologies_used"))
                ):
                    needs_exp = True
                    break
            except Exception:
                continue
        if needs_exp:
            missing.append("Work Experience (Company, Role, Dates, Location, Technologies Used)")

    return missing
