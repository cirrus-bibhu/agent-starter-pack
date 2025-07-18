from typing import Dict, Any

def get_missing_fields(job_data: Dict[str, Any]) -> Dict[str, list[str]]:
    """Checks a job data dictionary for essential missing fields."""
    missing_fields = {
        'basic_info': [],
        'vendor_info': []
    }

    def is_field_truly_missing(value, is_boolean=False):
        if is_boolean:
            return value is None
        return value is None or str(value).strip().lower() in ['', 'not specified', 'none']

    # Check basic job information
    if is_field_truly_missing(job_data.get('is_relocation_allowed'), is_boolean=True):
        missing_fields['basic_info'].append('is_relocation_allowed')
    if is_field_truly_missing(job_data.get('is_remote'), is_boolean=True):
        missing_fields['basic_info'].append('is_remote')

    # Check vendor information based on customer type
    customer_type = job_data.get('customer_type', '').lower()

    if customer_type == 'prime_vendor':
        if is_field_truly_missing(job_data.get('end_client_name')):
            missing_fields['vendor_info'].append('end_client_name')
    elif customer_type == 'sub_vendor':
        if is_field_truly_missing(job_data.get('end_client_name')):
            missing_fields['vendor_info'].append('end_client_name')
        if is_field_truly_missing(job_data.get('prime_vendor_name')):
            missing_fields['vendor_info'].append('prime_vendor_name')

    return {k: v for k, v in missing_fields.items() if v}
