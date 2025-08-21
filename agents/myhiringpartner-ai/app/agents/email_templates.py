from string import Template
from typing import List, Tuple, Optional


RECRUITER_EMAIL_TEMPLATE = Template("""Subject: Regarding Job Posting: ${job_title}

Hi ${recruiter_name},

Thank you for sharing the job opportunity for the ${job_title} position at ${end_client_name}.

${missing_fields_section}
${vendor_specific_section}
Thank you for your help!

Best regards,
MyHiringPartner.ai
""")


def render_candidate_followup_email(candidate_name: str, missing_fields: List[str], job_id: Optional[str] = None, editable_format: str = "form") -> Tuple[str, str]:
    """Return (subject, html) for a Gmail-compatible candidate follow-up email.

    Args:
        candidate_name: Name of the candidate
        missing_fields: List of missing field names
        job_id: Optional job ID
        editable_format: 'form', 'fillable_table', or 'simple_text'
    
    Returns:
        Tuple of (subject, html_body)
    """
    safe_name = (candidate_name or "there").strip()
    subject = "Quick details to complete your profile" + (f" (Job {job_id})" if job_id else "")

    # Normalization helper
    def norm(s: str) -> str:
        return (s or "").strip().lower()

    # Expand the human-friendly missing items into concrete schema fields grouped by section
    sections: dict[str, List[str]] = {
        "Contact": [],
        "Education": [],
        "Certifications": [],
        "Work Preferences": [],
        "Languages": [],
        "Skill Proficiency": [],
        "Experience": [],
        "Other": [],
    }

    for item in missing_fields:
        k = norm(item)
        if k in {"linkedin profile url", "linkedin", "linkedin url"}:
            sections["Contact"].append("linkedin_url")
        elif k in {"current location", "location", "candidate location"}:
            sections["Contact"].append("candidate_location")
        elif k == "email address":
            sections["Contact"].append("candidate_email")
        elif k == "phone number":
            sections["Contact"].append("candidate_phone")
        elif k == "date of birth":
            sections["Contact"].append("date_of_birth (YYYY-MM-DD)")
        elif k == "total years of experience":
            sections["Contact"].append("total_years_experience")

        elif k.startswith("highest degree") or k == "education institution" or k == "education field of study" or k == "graduation year":
            if k.startswith("highest degree"):
                sections["Education"].append("education[].degree")
            if k == "education institution":
                sections["Education"].append("education[].institution")
            if k == "education field of study":
                sections["Education"].append("education[].field_of_study")
            if k == "graduation year":
                sections["Education"].append("education[].graduation_year")
        elif k.startswith("bachelor's degree details"):
            sections["Education"].extend([
                "education[].(Bachelor) degree",
                "education[].(Bachelor) institution",
                "education[].(Bachelor) field_of_study",
                "education[].(Bachelor) graduation_year",
            ])

        elif k == "certifications (if any)" or k == "certifications":
            sections["Certifications"].append("certifications[]")

        elif k == "willing to relocate (yes/no)":
            sections["Work Preferences"].append("work_preferences.willing_to_relocate")
        elif k.startswith("preferred work models"):
            sections["Work Preferences"].append("work_preferences.preferred_work_models[]")
        elif k.startswith("preferred employment types"):
            sections["Work Preferences"].append("work_preferences.preferred_employment_types[]")
        elif k.startswith("preferred work locations"):
            sections["Work Preferences"].append("work_preferences.preferred_locations[]")
        elif k == "salary expectations":
            sections["Work Preferences"].append("work_preferences.salary_expectations")

        elif k == "languages and proficiency":
            sections["Languages"].extend([
                "languages[].language",
                "languages[].proficiency",
            ])

        elif k == "skills with level and years of experience":
            sections["Skill Proficiency"].extend([
                "skill_proficiency[].skill",
                "skill_proficiency[].level",
                "skill_proficiency[].years_experience",
            ])

        elif k.startswith("work experience"):
            sections["Experience"].extend([
                "experience[].company",
                "experience[].role",
                "experience[].start_date",
                "experience[].end_date",
                "experience[].duration_months",
                "experience[].location",
                "experience[].technologies_used",
            ])
        else:
            sections["Other"].append(item)

    # Choose rendering method based on format
    if editable_format == "form":
        sections_html = render_form_sections(sections)
    elif editable_format == "fillable_table":
        sections_html = render_fillable_table_sections(sections)
    else:  # simple_text
        sections_html = render_simple_text_sections(sections)

    html = f"""
    <div style="font-family:Arial,Helvetica,sans-serif;font-size:14px;line-height:1.6;color:#111827;">
      <p>Hi {safe_name},</p>
      <p>Thanks for sharing your resume. To complete your profile and proceed efficiently, please help with the following details{(' for Job ' + str(job_id)) if job_id else ''}. Please fill in the information below and reply to this email.</p>
      {sections_html}
      <p>You can reply to this email with the filled information. Thank you!</p>
      <p>Best regards,<br/>MyHiringPartner AI</p>
    </div>
    """

    return subject, html


def pretty_label(field: str) -> str:
    """Convert a schema-ish field like 'work_preferences.preferred_work_models[]'
    into a human-friendly label like 'Work Preferences – Preferred Work Models (multiple)'.

    Only affects the visible label; IDs and name attributes remain unchanged.
    """
    if not field:
        return ""

    multiple = "[]" in field
    # Remove array markers for display
    display = field.replace("[]", "")
    # Keep any explicit hint text like (YYYY-MM-DD) intact
    parts = [p.strip() for p in display.split(".")]
    pretty_parts: List[str] = []
    for p in parts:
        # Preserve content in parentheses, title-case the rest
        # Split into main and any parenthetical suffixes
        if "(" in p and ")" in p:
            main = p[: p.index("(")].replace("_", " ").strip()
            suffix = p[p.index("(") :].strip()
            pretty_main = main.title()
            pretty_parts.append((pretty_main + " " + suffix).strip())
        else:
            pretty_parts.append(p.replace("_", " ").strip().title())

    # Join parts with an en dash for readability when there are hierarchical sections
    if len(pretty_parts) > 1:
        label = "– ".join(pretty_parts)
    else:
        label = pretty_parts[0]
    # Append multiple hint
    if multiple:
        label = f"{label} (multiple)"
    return label

def render_form_sections(sections: dict[str, List[str]]) -> str:
    """Render sections as HTML form elements (most editable)."""
    parts = []
    
    for section_name, fields in sections.items():
        if not fields:
            continue
            
        form_fields = []
        for field in fields:
            field_id = field.replace(" ", "_").replace(".", "_").replace("[", "_").replace("]", "_")
            
            if "yes/no" in field.lower() or "willing_to_relocate" in field:
                form_fields.append(f"""
                <div style="margin:8px 0;">
                  <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                  <select name="{field_id}" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;">
                    <option value="">Please select...</option>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>
                """)
            elif "date" in field.lower():
                form_fields.append(f"""
                <div style="margin:8px 0;">
                  <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                  <input type="date" name="{field_id}" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;" />
                </div>
                """)
            elif "email" in field.lower():
                form_fields.append(f"""
                <div style="margin:8px 0;">
                  <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                  <input type="email" name="{field_id}" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;" placeholder="your.email@example.com" />
                </div>
                """)
            elif "phone" in field.lower():
                form_fields.append(f"""
                <div style="margin:8px 0;">
                  <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                  <input type="tel" name="{field_id}" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;" placeholder="+1 (555) 123-4567" />
                </div>
                """)
            elif "url" in field.lower() or "linkedin" in field.lower():
                form_fields.append(f"""
                <div style="margin:8px 0;">
                  <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                  <input type="url" name="{field_id}" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;" placeholder="https://linkedin.com/in/yourprofile" />
                </div>
                """)
            else:
                # Text area for longer fields, input for shorter ones
                if any(x in field.lower() for x in ["experience", "technologies", "skills"]):
                    form_fields.append(f"""
                    <div style="margin:8px 0;">
                      <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                      <textarea name="{field_id}" rows="3" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;resize:vertical;"></textarea>
                    </div>
                    """)
                else:
                    form_fields.append(f"""
                    <div style="margin:8px 0;">
                      <label style="display:block;font-weight:600;margin-bottom:4px;">{pretty_label(field)}:</label>
                      <input type="text" name="{field_id}" style="width:100%;padding:8px;border:1px solid #d1d5db;border-radius:4px;" />
                    </div>
                    """)
        
        if form_fields:
            fields_html = "".join(form_fields)
            parts.append(f"""
            <div style="margin:20px 0;padding:16px;border:1px solid #e5e7eb;border-radius:8px;background:#f9fafb;">
              <h3 style="margin:0 0 12px 0;font-size:16px;color:#111827;">{section_name}</h3>
              {fields_html}
            </div>
            """)
    
    return "".join(parts)


def render_fillable_table_sections(sections: dict[str, List[str]]) -> str:
    """Render sections as tables with input fields in cells."""
    parts = []
    
    for section_name, fields in sections.items():
        if not fields:
            continue
            
        rows = []
        for field in fields:
            field_id = field.replace(" ", "_").replace(".", "_").replace("[", "_").replace("]", "_")
            
            if "yes/no" in field.lower():
                input_html = f'<select name="{field_id}" style="width:100%;padding:4px;"><option value="">Select...</option><option value="Yes">Yes</option><option value="No">No</option></select>'
            elif "date" in field.lower():
                input_html = f'<input type="date" name="{field_id}" style="width:100%;padding:4px;" />'
            elif "email" in field.lower():
                input_html = f'<input type="email" name="{field_id}" style="width:100%;padding:4px;" placeholder="email@example.com" />'
            elif any(x in field.lower() for x in ["experience", "technologies", "skills"]):
                input_html = f'<textarea name="{field_id}" rows="2" style="width:100%;padding:4px;resize:vertical;"></textarea>'
            else:
                input_html = f'<input type="text" name="{field_id}" style="width:100%;padding:4px;" />'
            
            rows.append(f"""
            <tr>
              <td style="border:1px solid #d1d5db;padding:8px;background:#f9fafb;font-weight:600;width:40%;">{pretty_label(field)}</td>
              <td style="border:1px solid #d1d5db;padding:8px;width:60%;">{input_html}</td>
            </tr>
            """)
        
        if rows:
            rows_html = "".join(rows)
            parts.append(f"""
            <h3 style="margin:16px 0 8px 0;font-size:16px;color:#111827;">{section_name}</h3>
            <table border="0" cellpadding="0" cellspacing="0" width="100%" style="border-collapse:collapse;max-width:820px;border:1px solid #e5e7eb;margin:6px 0 16px 0;">
              <thead>
                <tr>
                  <th align="left" style="padding:8px;border:1px solid #d1d5db;background:#f3f4f6;">Field</th>
                  <th align="left" style="padding:8px;border:1px solid #d1d5db;background:#f3f4f6;">Your Details</th>
                </tr>
              </thead>
              <tbody>
                {rows_html}
              </tbody>
            </table>
            """)
    
    return "".join(parts)


def render_simple_text_sections(sections: dict[str, List[str]]) -> str:
    """Render sections as simple text format for easy copy-paste editing."""
    parts = []
    
    for section_name, fields in sections.items():
        if not fields:
            continue
        
        field_lines = []
        for field in fields:
            field_lines.append(f"{pretty_label(field)}: ___________________")
        
        fields_text = "<br/>".join(field_lines)
        parts.append(f"""
        <div style="margin:20px 0;padding:16px;border:1px solid #e5e7eb;border-radius:4px;background:#f9fafb;">
          <h3 style="margin:0 0 12px 0;font-size:16px;color:#111827;">{section_name}</h3>
          <div style="font-family:monospace;font-size:13px;line-height:1.8;">
            {fields_text}
          </div>
        </div>
        """)
    
    return "".join(parts)
