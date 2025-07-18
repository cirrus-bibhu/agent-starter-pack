from string import Template


RECRUITER_EMAIL_TEMPLATE = Template("""Subject: Regarding Job Posting: ${job_title}

Hi ${recruiter_name},

Thank you for sharing the job opportunity for the ${job_title} position at ${end_client_name}.

${missing_fields_section}
${vendor_specific_section}
Thank you for your help!

Best regards,
MyHiringPartner.ai
""")
