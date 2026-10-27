"""Module for extracting LinkedIn profile data from local sample data."""

import json
import logging
from typing import Dict, Optional, Any, List

import config

logger = logging.getLogger(__name__)


def list_available_profiles() -> List[str]:
    """
    List all available sample LinkedIn profile keys.

    Returns:
        List of profile key names.
    """
    try:
        with open(config.SAMPLE_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.keys())
    except Exception as e:
        logger.error(f"Error listing profiles: {e}")
        return config.AVAILABLE_PROFILES


def get_profile_label_map() -> Dict[str, str]:
    """
    Map human-readable dropdown labels to profile keys.
    """
    profiles = list_available_profiles()
    label_map = {}
    for p in profiles:
        label = config.PROFILE_LABELS.get(p, p.replace("_", " ").title())
        label_map[label] = p
    return label_map


def extract_linkedin_profile(
    linkedin_profile_url: str,
    api_key: Optional[str] = None,
    mock: bool = False,
) -> Dict[str, Any]:
    """
    Extract LinkedIn profile data from local sample data.

    The `linkedin_profile_url` can be:
    - A human-readable label (e.g. 'Satya Nadella — Chairman & CEO, Microsoft')
    - A profile key name (e.g. 'satya_nadella', 'jensen_huang')
    - A LinkedIn URL containing the public identifier
    - Left empty to use the first available profile

    Returns:
        Dictionary containing the LinkedIn profile data.
    """
    try:
        with open(config.SAMPLE_DATA_FILE, "r", encoding="utf-8") as f:
            all_profiles = json.load(f)

        profile_key = None

        if linkedin_profile_url:
            cleaned_input = linkedin_profile_url.strip()

            # 1. Check exact key match
            if cleaned_input in all_profiles:
                profile_key = cleaned_input

            # 2. Check label map match
            if not profile_key:
                label_map = get_profile_label_map()
                if cleaned_input in label_map:
                    profile_key = label_map[cleaned_input]

            # 3. Check public_identifier match
            if not profile_key:
                for k, v in all_profiles.items():
                    if v.get("public_identifier", "").lower() == cleaned_input.lower():
                        profile_key = k
                        break

            # 4. Check substring match in keys or names or URLs
            if not profile_key:
                lower_in = cleaned_input.lower().replace("-", "_")
                for k, v in all_profiles.items():
                    if k in lower_in or v.get("full_name", "").lower() in lower_in or lower_in in k:
                        profile_key = k
                        break

        # Fallback to first profile
        if not profile_key or profile_key not in all_profiles:
            profile_key = list(all_profiles.keys())[0]
            logger.info(f"Defaulting to profile key: {profile_key}")

        profile_data = all_profiles[profile_key].copy()

        # Clean out empty values
        profile_data = {
            k: v
            for k, v in profile_data.items()
            if v not in ([], "", None)
            and k not in ["people_also_viewed", "certifications"]
        }

        logger.info(
            f"Loaded local profile: {profile_data.get('full_name', profile_key)} (key={profile_key})"
        )
        return profile_data

    except Exception as e:
        logger.exception(f"Unexpected error while extracting profile: {e}")
        return {}
