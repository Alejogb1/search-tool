#!/usr/bin/env python
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This example generates keyword ideas from a list of seed keywords."""

import sys
from google.ads.googleads.client import GoogleAdsClient as AdsClient
from google.ads.googleads.errors import GoogleAdsException

class GoogleAdsClient:
    _DEFAULT_LOCATION_IDS = ["1023191"]  # New York, NY
    #_DEFAULT_LOCATION_IDS = ["20010"]  # Buenos Aires, BA
    #_DEFAULT_LANGUAGE_ID = "1003"  # Spanish

    _DEFAULT_LANGUAGE_ID = "1000"  # English
    def __init__(self, config_path='google-ads.yaml'):
        self.client = AdsClient.load_from_storage(config_path, version="v20")

    def generate_keyword_ideas(self, customer_id, keyword_texts, page_url=None, location_ids=None, language_id=None):
        location_ids = location_ids or self._DEFAULT_LOCATION_IDS
        language_id = language_id or self._DEFAULT_LANGUAGE_ID

        keyword_plan_idea_service = self.client.get_service("KeywordPlanIdeaService")
        keyword_plan_network = self.client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
        
        location_rns = self._map_locations_ids_to_resource_names(location_ids)
        language_rn = self.client.get_service("GoogleAdsService").language_constant_path(language_id)

        if not (keyword_texts or page_url):
            raise ValueError("At least one of keywords or page URL is required.")

        request = self.client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = customer_id
        request.language = language_rn
        request.geo_target_constants = location_rns
        request.include_adult_keywords = False
        request.keyword_plan_network = keyword_plan_network

        if not keyword_texts and page_url:
            request.url_seed.url = page_url
        elif keyword_texts and not page_url:
            request.keyword_seed.keywords.extend(keyword_texts)
        elif keyword_texts and page_url:
            request.keyword_and_url_seed.url = page_url
            request.keyword_and_url_seed.keywords.extend(keyword_texts)

        try:
            keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
            return [self._format_idea(idea) for idea in keyword_ideas]
        except GoogleAdsException as ex:
            self._handle_google_ads_exception(ex)
            return []

    def _format_idea(self, idea):
        competition_value = idea.keyword_idea_metrics.competition.name
        return {
            "text": idea.text,
            "avg_monthly_searches": idea.keyword_idea_metrics.avg_monthly_searches,
            "competition": competition_value
        }

    def _map_locations_ids_to_resource_names(self, location_ids):
        build_resource_name = self.client.get_service("GeoTargetConstantService").geo_target_constant_path
        return [build_resource_name(location_id) for location_id in location_ids]

    def _handle_google_ads_exception(self, ex):
        print(
            f'Request with ID "{ex.request_id}" failed with status '
            f'"{ex.error.code().name}" and includes the following errors:'
        )
        for error in ex.failure.errors:
            print(f'\tError with message "{error.message}".')
            if error.location:
                for field_path_element in error.location.field_path_elements:
                    print(f"\t\tOn field: {field_path_element.field_name}")
        sys.exit(1)
